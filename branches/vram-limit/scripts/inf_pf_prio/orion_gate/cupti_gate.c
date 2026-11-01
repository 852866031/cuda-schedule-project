/* Kernel-level idle-window gate via CUPTI callbacks (the "one level below
 * cuBLAS" interception).
 *
 * Why this exists: LD_PRELOAD interposition of the CUDA *runtime* API (Orion's
 * mechanism, and colocation/colocator's) cannot see cuBLAS/cuDNN/Triton kernel
 * launches -- those libraries obtain *driver*-API entry points through
 * cuGetProcAddress and never cross an interposable symbol. CUPTI's callback API
 * hooks inside libcuda itself: every driver-level launch fires a synchronous
 * callback on the launching thread, whoever the caller is. Blocking in the
 * ENTER callback therefore delays exactly that launch -- a universal, per-kernel
 * gate with no per-library wrappers.
 *
 * Role: LD_PRELOAD into the *trainer* only (COLOC_ROLE=be). The decode engine
 * keeps publishing its busy window via orion_gate/hp_patch (unchanged).
 *
 * Policy per intercepted launch, on the launching thread:
 *   1. bounded lookahead: if > COLOC_MAXPEND credit events are unretired,
 *      synchronize on the oldest (caps the async backlog that would otherwise
 *      drain into the next decode step);
 *   2. wait while the shm busy flag is set (heartbeat-stale => open);
 *   3. let the launch proceed; every COLOC_K-th launch records a credit event
 *      on the default stream.
 *
 * Build: gcc -O2 -shared -fPIC -I$CUPTI/include cupti_gate.c -o cupti_gate.so \
 *            -L$CUPTI/lib -lcupti -ldl -lpthread -Wl,-rpath,$CUPTI/lib
 */
#define _GNU_SOURCE
#include <cupti.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define SHM_NAME "/coloc_hp_busy"
#define MAGIC 0x434f4c4f43ll
#define HB_STALE_NS 1500000000ll
#define NEVENTS 64

struct page {
    volatile int64_t magic, heartbeat_ns, busy, last_ns, steps,
                     be_launches, be_gated_ns;
};

static struct page *pg;
static int cfg_k = 8, cfg_maxpend = 3;
static __thread int inhook;          /* reentry guard for our own CUDA calls */

typedef int (*ev_create_fn)(void **, unsigned);
typedef int (*ev_record_fn)(void *, void *);
typedef int (*ev_query_fn)(void *);
typedef int (*ev_sync_fn)(void *);
static ev_create_fn ev_create;
static ev_record_fn ev_record;
static ev_query_fn ev_query;
static ev_sync_fn ev_sync;

static void *events[NEVENTS];
static int ring_head, ring_tail;     /* [head, tail) pending credit events */
static long long n_launch, n_gate_hits, n_fence_syncs;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

static int64_t now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

static int cuda_syms(void) {
    if (ev_record) return 1;
    ev_create = (ev_create_fn)dlsym(RTLD_DEFAULT, "cudaEventCreateWithFlags");
    ev_record = (ev_record_fn)dlsym(RTLD_DEFAULT, "cudaEventRecord");
    ev_query = (ev_query_fn)dlsym(RTLD_DEFAULT, "cudaEventQuery");
    ev_sync = (ev_sync_fn)dlsym(RTLD_DEFAULT, "cudaEventSynchronize");
    return ev_record && ev_query && ev_sync && ev_create;
}

static void gate_one_launch(void) {
    if (!pg || pg->magic != MAGIC || !cuda_syms())
        return;
    __atomic_add_fetch((int64_t *)&pg->be_launches, 1, __ATOMIC_RELAXED);
    /* 1. bounded lookahead */
    pthread_mutex_lock(&mu);
    while (ring_head != ring_tail && ev_query(events[ring_head % NEVENTS]) == 0)
        ring_head++;
    if (ring_tail - ring_head > cfg_maxpend) {
        void *oldest = events[ring_head % NEVENTS];
        pthread_mutex_unlock(&mu);
        ev_sync(oldest);
        n_fence_syncs++;
        pthread_mutex_lock(&mu);
        while (ring_head != ring_tail &&
               ev_query(events[ring_head % NEVENTS]) == 0)
            ring_head++;
    }
    pthread_mutex_unlock(&mu);

    /* 2. yield while the decode engine is mid-step */
    if (pg->busy && now_ns() - pg->heartbeat_ns < HB_STALE_NS) {
        int64_t t0 = now_ns();
        n_gate_hits++;
        struct timespec nap = {0, 100000};   /* 100 us */
        while (pg->busy && now_ns() - pg->heartbeat_ns < HB_STALE_NS)
            nanosleep(&nap, NULL);
        __atomic_add_fetch((int64_t *)&pg->be_gated_ns, now_ns() - t0,
                           __ATOMIC_RELAXED);
    }

    /* 3. credit event every cfg_k launches (default stream) */
    if (n_launch % cfg_k == 0) {
        pthread_mutex_lock(&mu);
        if (ring_tail - ring_head < NEVENTS) {
            int slot = ring_tail % NEVENTS;
            if (!events[slot])
                ev_create(&events[slot], 2 /* disable timing */);
            if (events[slot] && ev_record(events[slot], NULL) == 0)
                ring_tail++;
        }
        pthread_mutex_unlock(&mu);
    }
}

static void CUPTIAPI on_callback(void *ud, CUpti_CallbackDomain domain,
                                 CUpti_CallbackId cbid,
                                 const CUpti_CallbackData *cb) {
    (void)ud; (void)domain; (void)cbid;
    if (inhook || cb->callbackSite != CUPTI_API_ENTER || !cb->functionName)
        return;
    if (!strstr(cb->functionName, "Launch"))
        return;
    inhook = 1;
    n_launch++;                       /* counts ALL intercepted launches */
    gate_one_launch();
    inhook = 0;
}

static void report(void) {
    fprintf(stderr,
            "[cupti_gate] launches intercepted=%lld gate_hits=%lld "
            "fence_syncs=%lld gated_total_ms=%.1f\n",
            n_launch, n_gate_hits, n_fence_syncs,
            pg ? pg->be_gated_ns / 1e6 : 0.0);
}

static void setup(void) __attribute__((constructor));
static void setup(void) {
    const char *role = getenv("COLOC_ROLE");
    if (!role || strcmp(role, "be") != 0)
        return;
    const char *k = getenv("COLOC_K");
    const char *m = getenv("COLOC_MAXPEND");
    if (k) cfg_k = atoi(k);
    if (m) cfg_maxpend = atoi(m);

    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (fd < 0) return;
    if (ftruncate(fd, sizeof(struct page)) != 0) { close(fd); return; }
    pg = mmap(NULL, sizeof(struct page), PROT_READ | PROT_WRITE, MAP_SHARED,
              fd, 0);
    close(fd);
    if (pg == MAP_FAILED) { pg = NULL; return; }

    CUpti_SubscriberHandle sub;
    if (cuptiSubscribe(&sub, (CUpti_CallbackFunc)on_callback, NULL)
        != CUPTI_SUCCESS) {
        fprintf(stderr, "[cupti_gate] cuptiSubscribe failed (another profiler "
                        "attached?) -- running UNGATED\n");
        return;
    }
    cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_DRIVER_API);
    cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_RUNTIME_API);
    atexit(report);
    fprintf(stderr, "[cupti_gate] active: driver+runtime launch callbacks, "
                    "K=%d maxpend=%d, pid=%d\n", cfg_k, cfg_maxpend, getpid());
}
