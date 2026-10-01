# Measured results

All numbers come from the logs in [`results/`](results/). Hardware: Apple M2 laptop GPU (PyTorch MPS).
Data: Tiny Shakespeare, 576K BPE tokens (vocab 512), with the last 10% of the corpus held out for validation.

## Main run

`python -m slm.train --steps 5000` (4 layers, 4 heads, C=256, T=128, dropout 0.1, lr 1e-3, batch 32)

| | value |
|---|---|
| parameters | 3.29M |
| val loss at step 0 | 6.26 (ln 512 = 6.24, i.e. uniform) |
| best val loss | **2.793** at step 2,750 (perplexity 16.3 per BPE token) |
| final train / val loss | 1.66 / 2.84 at step 5,000 (overfitting after ~2,750) |

The best-validation checkpoint is kept, not the last one.

## Sweep

`python -m slm.sweep --steps 1500`. One axis varied around the baseline (T=128, C=256, lr=1e-3), with the same seed and the same fixed eval batches. **Best val loss within 1,500 steps:**

| axis | value | best val loss | params |
|---|---|---:|---:|
| context length | 32 | 3.007 | 3.29M |
| | 64 | 2.891 | 3.29M |
| | 128 (baseline) | 2.818 | 3.29M |
| | 256 | 2.870 | 3.29M |
| embedding size | 128 | 3.154 | 0.86M |
| | 256 (baseline) | 2.818 | 3.29M |
| | 384 | **2.776** | 7.30M |
| learning rate | 3e-4 | 3.089 | 3.29M |
| | 1e-3 (baseline) | 2.818 | 3.29M |
| | 3e-3 | 2.792 | 3.29M |

Caveat: most runs were still improving at step 1,500, so this ranks configurations at a fixed
short budget, not at convergence. A short budget favors faster learners (the higher learning rate).
For comparison, the 5,000-step baseline reached 2.793, the same as lr=3e-3 at 1,500 steps.
