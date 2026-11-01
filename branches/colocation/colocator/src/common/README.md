# src/common/ — shared contract between the two libraries

`records.h` is the single header both `libcolocator.so` and `libsched.so`
compile against. It holds:

- `OpKind` — every interposed CUDA call.
- `FuncRecord` — one intercepted operation: op tag, per-client `op_id`,
  `t_intercept_ns`, the saved call arguments, and the
  `state` machine (`QUEUED -> ISSUED -> DONE`) the client spins on.
- `ClientSlot` + the extern globals (`g_col_clients`, `g_col_num_clients`,
  `g_col_mode`) — **defined** in intercept.cpp, referenced by scheduler.cpp,
  resolved by the dynamic linker at load time (libcolocator is preloaded into
  global scope, so no dlsym wiring is needed — simpler than Orion).

Read the header comment in `records.h` first: it states the
lifetime/threading contract (records live on the client's stack; the client
may only proceed once the scheduler no longer needs the storage), which is
the invariant everything else leans on.
