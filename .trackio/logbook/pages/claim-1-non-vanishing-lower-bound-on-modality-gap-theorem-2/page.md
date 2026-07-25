# Claim 1: Non-vanishing lower bound on modality gap (Theorem 2)


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e4f9a80d60e8", "created_at": "2026-07-25T04:40:08+00:00", "title": "Claim 1: Non-vanishing lower bound on modality gap (Theorem 2)"}
-->
## Claim 1: Non-vanishing lower bound on modality gap (Theorem 2)

**Paper statement**: Theorem 2 proves that ||R - W*||_F^2 >= sum_{y in Y} 2(1 - ||Proj_S(r_y)||_2), establishing that text-derived prototypes R and optimal visual prototypes W* cannot be fully aligned when text prototypes contain components outside the visual feature subspace S.

### Numerical Audit Method
We implement the inequality directly:
1. Generate random visual features z_i and compute subspace S = span{z_1, ..., z_N}
2. Compute optimal visual prototypes W* via Theorem 1 fixed-point iteration
3. Generate text prototypes R with controlled fraction outside S
4. Verify LHS >= RHS for mixed, tight (R in S), and loose (R outside S) cases

### Results (5 random seeds)

| Case | LHS (||R-W*||^2) | RHS (lower bound) | LHS >= RHS? |
|------|------------------|--------------------|-------------|
| Mixed (15% out) | 19.70 avg | 1.63 avg | PASS |
| Tight (R in S) | 20.45 avg | 0.00 avg | PASS |
| Loose (R out S) | 20.00 avg | 20.00 avg | PASS |

All 5 trials pass. Key findings:
- When text prototypes lie entirely in S (tight case), the lower bound vanishes as expected
- When text prototypes have components outside S, the bound is non-vanishing
- W* perpendicular component is ~1e-15 (machine precision), confirming W* lies in S

### Conclusion
**VERIFIED**. Theorem 2 holds numerically across all tested configurations. The modality gap is structural and cannot be eliminated by prompt engineering alone (since text prototypes generally have components outside S).

### Code
- Audit script: https://github.com/hyw1008/ICML2026/blob/main/audit_theorem2.py
- arXiv: https://arxiv.org/abs/2605.26661 (Section 3, Theorem 2, Eqs 5-6)
- GitHub: https://github.com/hyw1008/ICML2026
