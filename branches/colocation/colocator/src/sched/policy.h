// Scheduling policy interface — the extension point for Orion/REEF-style
// logic. The scheduler loop peeks the head record of every client queue and
// asks the policy which one to launch next; mechanism (queues, streams,
// replay) stays in scheduler.cpp, policy is only this decision.
#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>

#include "../common/records.h"

// What the scheduler exposes to policies. Extend as future policies need
// more (e.g. op duration profiles, live GPU busy time from the observer).
struct SchedState {
    int num_clients = 0;
    std::vector<uint64_t> issued;     // ops issued so far, per client
    // EXACT count of stream ops issued but not yet finished on the GPU,
    // per client — from the per-op completion-event cursor (Orion-style):
    // one cudaEvent is recorded after every issued stream op, and the
    // scheduler loop advances a cursor with cudaEventQuery each iteration.
    std::vector<uint64_t> in_flight;
};

struct Policy {
    virtual ~Policy() = default;
    // heads[i] is client i's oldest queued record, or nullptr if its queue is
    // empty. Return the client whose head to execute now, or -1 to idle.
    virtual int pick_next(const std::vector<FuncRecord*>& heads, const SchedState& s) = 0;
    virtual const char* name() const = 0;
};

// FCFS across clients: run whichever queued op was intercepted earliest.
// Within one client, queue order already enforces program order.
struct FcfsPolicy : Policy {
    int pick_next(const std::vector<FuncRecord*>& heads, const SchedState&) override {
        int best = -1;
        uint64_t best_t = ~0ull;
        for (int i = 0; i < (int)heads.size(); i++) {
            if (heads[i] && heads[i]->t_intercept_ns < best_t) {
                best_t = heads[i]->t_intercept_ns;
                best = i;
            }
        }
        return best;
    }
    const char* name() const override { return "fcfs"; }
};

// FCFS + in-flight cap: a client whose stream already holds `threshold` or
// more unfinished ops gets its head HELD (not issued) until the event cursor
// shows completions — so in-flight never exceeds `threshold`. First policy to
// use SchedState.in_flight; it bounds how deep any one client can stack the
// hardware queue. Ops that add no stream work (malloc/free/streamSync) are
// always eligible: holding them would stall the client without protecting the
// GPU from anything.
struct ThrottledFcfsPolicy : Policy {
    uint64_t threshold;
    char name_buf[40];
    explicit ThrottledFcfsPolicy(uint64_t thr) : threshold(thr) {
        snprintf(name_buf, sizeof(name_buf), "throttled-fcfs(cap %llu)",
                 (unsigned long long)thr);
    }

    static bool adds_stream_work(OpKind k) {
        return k == OpKind::KernelLaunch || k == OpKind::Memcpy ||
               k == OpKind::MemcpyAsync || k == OpKind::Memset ||
               k == OpKind::MemsetAsync || k == OpKind::CublasGemmEx ||
               k == OpKind::CublasSgemm;  // replayed cuBLAS calls enqueue kernels
    }

    int pick_next(const std::vector<FuncRecord*>& heads, const SchedState& s) override {
        int best = -1;
        uint64_t best_t = ~0ull;
        for (int i = 0; i < (int)heads.size(); i++) {
            if (!heads[i]) continue;
            if (adds_stream_work(heads[i]->kind) && s.in_flight[i] >= threshold)
                continue;  // hold: too much of this client's work already pending
            if (heads[i]->t_intercept_ns < best_t) {
                best_t = heads[i]->t_intercept_ns;
                best = i;
            }
        }
        return best;  // -1 = everyone ineligible: idle until completions arrive
    }
    const char* name() const override { return name_buf; }
};
