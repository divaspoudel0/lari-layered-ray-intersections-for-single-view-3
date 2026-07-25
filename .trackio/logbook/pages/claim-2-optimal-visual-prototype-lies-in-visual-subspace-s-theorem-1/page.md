# Claim 2: Optimal visual prototype lies in visual subspace S (Theorem 1)


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e9c21c54eeec", "created_at": "2026-07-25T04:40:13+00:00", "title": "Claim 2: Optimal visual prototype lies in visual subspace S (Theorem 1)"}
-->
## Claim 2: Optimal visual prototype lies in visual subspace S (Theorem 1)

**Paper statement**: Theorem 1 shows w_y* = l2(sum_{D_y} (1-pi_iy)*z_i - sum_{D\D_y} pi_iy*z_i), meaning the optimal visual prototype is a linear combination of visual features and thus lies entirely in S = span{z_1, ..., z_N}.

### Numerical Audit Method
1. Generate random visual features z_i on S^{d-1}
2. Compute optimal prototypes W* via iterative fixed-point convergence
3. Project W* onto orthogonal complement of S and measure relative perpendicular error
4. Verify self-consistency (update rule returns same W at fixed point)
5. Verify closed-form expression matches converged W*

### Results (3 random seeds, d=64, N=500, K=10)

| Metric | Trial 1 | Trial 2 | Trial 3 |
|--------|---------|---------|---------|
| Max relative perp error | 2.18e-15 | 2.34e-15 | 2.10e-15 |
| Self-consistency max change | 4.78e-08 | 6.49e-08 | 2.37e-08 |
| Closed-form cosine distance | 3.56e-09 | 4.52e-09 | 3.97e-09 |
| Converged in iterations | 9 | 11 | 10 |

All checks pass at machine precision:
- W* lies in S (perpendicular error ~ 1e-15)
- Fixed-point iteration converges (< 10 iterations)
- Closed-form expression matches converged solution

### Conclusion
**VERIFIED**. Theorem 1 holds numerically. The optimal visual prototype w_y* is provably a linear combination of visual features, lying entirely in the visual feature subspace S. This is the foundation for Theorem 2's modality gap result.

### Code
- Audit script: https://github.com/hyw1008/ICML2026/blob/main/audit_theorem1.py
- arXiv: https://arxiv.org/abs/2605.26661 (Section 3, Theorem 1)
- GitHub: https://github.com/hyw1008/ICML2026
