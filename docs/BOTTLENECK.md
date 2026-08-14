# Core Bottleneck: Pens 3, 7, 8

## Per-Pen Ink Statistics (from training data analysis)
| Pen | Intensity | Stroke Width | R-G Diff | Saturation |
|-----|-----------|-------------|----------|------------|
| 6   | 64 (darkest) | 2.21 | — | — |
| 2   | 83        | 2.11        | —        | —          |
| 1   | 112       | 2.16        | —        | —          |
| **3** | **131.5** | **2.00** | **10.48** | **21.17** |
| **7** | **131.3** | **2.00** | **10.01** | **20.30** |
| **8** | **132.2** | **1.78** | **7.64**  | **16.16** |
| 4   | 148       | 1.73        | —        | —          |
| 5   | 177 (lightest) | 1.69  | —        | —          |

## Why This Matters
- Pens 1, 2, 4, 5, 6 are easily separable by intensity alone
- Pens 3, 7, 8 have nearly identical intensity (~131-132), similar width (~1.78-2.0)
- The only subtle differences: R-G channel diff and saturation
- Pen 8 is slightly distinguishable (lower saturation, thinner stroke)
- Pens 3 vs 7 are the hardest pair — virtually identical on ALL metrics
- Model is likely ~98%+ on easy pens, ~75-80% on pens 3/7/8
- This confusion alone caps overall accuracy below 0.90

## What Might Separate Them
- Micro-texture / ink bleed patterns
- Higher resolution needed (v5: 336px)
- Texture features (v5: Gabor + LBP)
- Fiber-level ink absorption differences
