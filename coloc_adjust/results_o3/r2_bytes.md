# R2: bytes per token of every tenant shape and variant (alone, counter calibration 1 % = 17.79 GB/s)

| run | kind | shape | SMs | tokens/s | GB/s while running | MB per token | vs usual | SM util % | W | loss | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base_cfg2 | usual | 4 micro-batches (usual) | 32 | 11628 | 156.6 | 13.47 | +0 % | 15.57 | 222 | 11.951 | ok |
| base_cfg0 | usual | 1 micro-batch | 32 | 14543 | 226.5 | 15.58 | +16 % | 17.86 | 251 | 11.946 | ok |
| base_cfg1 | usual | 2 micro-batches | 32 | 13416 | 180.8 | 13.48 | +0 % | 16.74 | 235 | 11.948 | ok |
| base_cfg3 | usual | 8 micro-batches | 32 | 9579 | 142.5 | 14.88 | +10 % | 13.59 | 212 | 11.954 | ok |
| base_cfg4 | usual | 8 mb + activations in host RAM | 32 | 3246 | 49.3 | 15.19 | +13 % | 5.85 | 143 | 12.045 | ok |
| base_cfg5 | usual | + 12 layers in host RAM | 32 | 2566 | 42.5 | 16.57 | +23 % | 4.64 | 132 | 12.085 | ok |
| base_cfg6 | usual | + 20 layers in host RAM | 32 | 2364 | 41.6 | 17.61 | +31 % | 4.27 | 132 | 12.112 | ok |
| top8 | less training | 4 micro-batches (usual) | 32 | 17743 | 179.2 | 10.10 | -25 % | 16.20 | 226 | 11.962 | ok |
| top4 | less training | 4 micro-batches (usual) | 32 | 20198 | 186.7 | 9.24 | -31 % | 16.43 | 230 | 11.968 | ok |
| losschunk | same training | 4 micro-batches (usual) | 32 | 7706 | 325.5 | 42.23 | +214 % | 16.13 | 262 | 11.981 | ok |
| compile | same training | 4 micro-batches (usual) | 32 | 14457 | 114.9 | 7.95 | -41 % | 16.14 | 224 | 11.945 | ok |
| compile_losschunk | same training | 4 micro-batches (usual) | 32 | – | – | – | – |  |  |  | exit 1; only 0 steps |
| eager | same training (check) | 4 micro-batches (usual) | 32 | 10096 | 208.5 | 20.66 | +53 % | 15.94 | 238 | 11.953 | ok |
| ckpt | same training | 4 micro-batches (usual) | 32 | 8857 | 115.8 | 13.08 | -3 % | 15.16 | 213 | 11.954 | ok |
| fwd | reference, no training | 4 micro-batches (usual) | 32 | 25010 | 93.2 | 3.73 | -72 % | 16.09 | 214 | 13.645 | ok |
| base_full | usual | 4 micro-batches (usual) | 170 | 28549 | 408.4 | 14.30 | +6 % | 58.28 | 369 | 11.940 | ok |
| top4_full | less training | 4 micro-batches (usual) | 170 | 54236 | 556.6 | 10.26 | -24 % | 70.04 | 409 | 11.949 | ok |
| losschunk_full | same training | 4 micro-batches (usual) | 170 | 17801 | 749.3 | 42.09 | +212 % | 64.18 | 401 | 11.979 | ok |
