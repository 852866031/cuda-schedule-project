/* Orion-lite kernel gate: one LD_PRELOAD shim, two roles, one /dev/shm page.
 *
 *   COLOC_ROLE=hp  (the vLLM decode engine)  -- publish "GPU busy" into shm:
 *       every intercepted launch bumps last_launch_ns and (throttled) records a
 *       pooled CUDA event on the launching stream; a background thread retires
 *       events and writes  busy = events pending || launch < RECENT_NS ago.
 *       Completion-awareness matters because vLLM replays whole decode steps as
 *       single cudaGraphLaunch calls: the API call is ~us, the GPU work ~20 ms.
 *
 *   COLOC_ROLE=be  (the fine-tune trainer) -- gate: every intercepted launch
 *       spins (usleep) while the page says busy. If the hp heartbeat goes stale
 *       (engine died / not started), the gate opens so the trainer never hangs.
 *
 * Interposed: cudaLaunchKernel, cudaLaunchKernelExC, cudaGraphLaunch. That covers
 * torch (aten/cublas) and vLLM's graph replays; eager Triton kernels launched via
 * the driver API are missed -- accepted for v0 and checked empirically against
 * DCGM SM-activity.
 *
 * Build:  gcc -O2 -shared -fPIC -o gate_shim.so gate_shim.c -ldl -lpthread
 * No CUDA headers or libs: real entry points come from dlsym.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define SHM_NAME "/coloc_hp_busy"
#define MAGIC 0x434f4c4f43ll
#define POOL 256              /* event pool / ring size                       */
#define PROBE_NS 200000ll     /* record at most one probe event per 200 us    */
#define RECENT_NS 300000ll    /* "busy" for 300 us after any launch           */
#define HB_STALE_NS 1500000000ll /* be: ignore busy if hp heartbeat older     */

typedef struct { unsigned x, y, z; } dim3_;

struct page {
    volatile int64_t magic, heartbeat_ns, busy, last_launch_ns,
                     launches, probes, be_gated_ns;
};

static struct page *pg;
static int role_hp = 0, role_be = 0, active = 0;

static int64_t now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

/* ---- real CUDA entry points, resolved lazily ------------------------------ */
typedef int (*launch_fn)(const void *, dim3_, dim3_, void **, size_t, void *);
typedef int (*launch_exc_fn)(const void *, const void *, void **);
typedef int (*graph_fn)(void *, void *);
typedef int (*ev_create_fn)(void **, unsigned);
typedef int (*ev_record_fn)(void *, void *);
typedef int (*ev_query_fn)(void *);
static launch_fn real_launch;
static launch_exc_fn real_launch_exc;
static graph_fn real_graph;
static ev_create_fn ev_create;
static ev_record_fn ev_record;
static ev_query_fn ev_query;

/* ---- hp state ------------------------------------------------------------- */
static void *events[POOL];
static void *ring[POOL];          /* recorded, not yet retired */
static int ring_head, ring_tail;  /* [head, tail) pending      */
static int64_t last_probe;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_t poller;

static void *poll_loop(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&mu);
        while (ring_head != ring_tail) {
            /* cudaSuccess=0 done; 600 cudaErrorNotReady */
            if (ev_query && ev_query(ring[ring_head % POOL]) == 0)
                ring_head++;
            else
                break;
        }
        int pending = ring_tail - ring_head;
        pthread_mutex_unlock(&mu);
        int64_t t = now_ns();
        pg->busy = pending > 0 || (t - pg->last_launch_ns) < RECENT_NS;
        pg->probes = pending;
        pg->heartbeat_ns = t;
        usleep(100);
    }
    return NULL;
}

static void hp_note_launch(void *stream) {
    int64_t t = now_ns();
    pg->last_launch_ns = t;
    __atomic_add_fetch((int64_t *)&pg->launches, 1, __ATOMIC_RELAXED);
    if (t - last_probe < PROBE_NS)
        return;
    if (!ev_record) {
        ev_record = (ev_record_fn)dlsym(RTLD_DEFAULT, "cudaEventRecord");
        ev_query = (ev_query_fn)dlsym(RTLD_DEFAULT, "cudaEventQuery");
        ev_create = (ev_create_fn)dlsym(RTLD_DEFAULT, "cudaEventCreateWithFlags");
        if (!ev_record || !ev_query || !ev_create) return;
    }
    pthread_mutex_lock(&mu);
    if (ring_tail - ring_head < POOL) {
        int slot = ring_tail % POOL;
        if (!events[slot])
            ev_create(&events[slot], 2 /* cudaEventDisableTiming */);
        if (events[slot] && ev_record(events[slot], stream) == 0) {
            ring[slot] = events[slot];
            ring_tail++;
            last_probe = t;
        }
    }
    pthread_mutex_unlock(&mu);
}

/* ---- be gate -------------------------------------------------------------- */
static void be_gate(void) {
    if (pg->magic != MAGIC) return;
    int64_t t0 = now_ns();
    while (pg->busy && (now_ns() - pg->heartbeat_ns) < HB_STALE_NS)
        usleep(50);
    __atomic_add_fetch((int64_t *)&pg->be_gated_ns, now_ns() - t0,
                       __ATOMIC_RELAXED);
}

/* ---- init ----------------------------------------------------------------- */
static void setup(void) __attribute__((constructor));
static void setup(void) {
    const char *role = getenv("COLOC_ROLE");
    if (!role) return;
    role_hp = strcmp(role, "hp") == 0;
    role_be = strcmp(role, "be") == 0;
    if (!role_hp && !role_be) return;

    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (fd < 0) return;
    if (ftruncate(fd, sizeof(struct page)) != 0) { close(fd); return; }
    pg = mmap(NULL, sizeof(struct page), PROT_READ | PROT_WRITE, MAP_SHARED,
              fd, 0);
    close(fd);
    if (pg == MAP_FAILED) { pg = NULL; return; }
    if (role_hp) {
        memset((void *)pg, 0, sizeof(struct page));
        pg->magic = MAGIC;
        pthread_create(&poller, NULL, poll_loop, NULL);
    }
    active = 1;
    fprintf(stderr, "[gate_shim] active, role=%s pid=%d\n", role, getpid());
}

/* ---- interposed entry points ---------------------------------------------- */
int cudaLaunchKernel(const void *fn, dim3_ g, dim3_ b, void **args,
                     size_t shm, void *stream) {
    if (!real_launch)
        real_launch = (launch_fn)dlsym(RTLD_NEXT, "cudaLaunchKernel");
    if (active && role_be) be_gate();
    int r = real_launch(fn, g, b, args, shm, stream);
    if (active && role_hp && r == 0) hp_note_launch(stream);
    return r;
}

int cudaLaunchKernelExC(const void *cfg, const void *fn, void **args) {
    if (!real_launch_exc)
        real_launch_exc = (launch_exc_fn)dlsym(RTLD_NEXT, "cudaLaunchKernelExC");
    if (active && role_be) be_gate();
    int r = real_launch_exc(cfg, fn, args);
    /* cudaLaunchConfig_t's first member is the stream */
    if (active && role_hp && r == 0 && cfg)
        hp_note_launch(*(void *const *)cfg);
    return r;
}

int cudaGraphLaunch(void *graph_exec, void *stream) {
    if (!real_graph)
        real_graph = (graph_fn)dlsym(RTLD_NEXT, "cudaGraphLaunch");
    if (active && role_be) be_gate();
    int r = real_graph(graph_exec, stream);
    if (active && role_hp && r == 0) hp_note_launch(stream);
    return r;
}
