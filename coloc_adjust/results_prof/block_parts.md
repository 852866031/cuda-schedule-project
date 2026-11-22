# Parts of the 24 blocks: bytes per micro-batch of the usual tenant (2026-11-22)

Tenant alone, green context of 32 SMs, 12 micro-batches of 2 x 512 tokens; a mark (with a wait for the GPU) before every part, so the micro-batch runs slower than in the tenant. One micro-batch: 14.29 GB, of it the 24 blocks 6.42 GB.

| part (summed over the 24 blocks) | GB per micro-batch | share of the blocks | ms |
|---|---|---|---|
| fwd:norm1 | 0.087 | 1 % | 1.6 |
| fwd:qkv_proj | 0.344 | 5 % | 7.2 |
| fwd:attention | 0.155 | 2 % | 3.6 |
| fwd:o_proj+residual | 0.291 | 5 % | 3.8 |
| fwd:norm2 | 0.081 | 1 % | 1.5 |
| fwd:mlp+residual | 1.689 | 26 % | 16.8 |
| bwd:mlp+residual | 2.061 | 32 % | 17.3 |
| bwd:norm2 | 0.326 | 5 % | 2.2 |
| bwd:o_proj+residual | 0.256 | 4 % | 3.9 |
| bwd:attention | 0.296 | 5 % | 5.3 |
| bwd:qkv_proj | 0.652 | 10 % | 9.7 |
| bwd:norm1 | 0.178 | 3 % | 2.2 |
| embedding | 0.020 | – | 1.0 |
| output layer and loss | 7.855 | – | 18.0 |
