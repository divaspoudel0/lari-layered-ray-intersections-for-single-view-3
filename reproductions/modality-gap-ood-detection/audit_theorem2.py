"""
Numerical Audit of Theorem 2 (Modality Gap Lower Bound)

Theorem 2 states:
  ||R - W*||_F^2 >= sum_{y in Y} 2(1 - ||Proj_S(r_y)||_2)

This audit verifies the inequality holds numerically across different configurations.
"""

import numpy as np

np.random.seed(42)

def orthogonal_projector(S_basis):
    U, S, Vt = np.linalg.svd(S_basis.T, full_matrices=False)
    tol = max(S_basis.shape) * S[0] * 1e-10
    rank = np.sum(S > tol)
    Q = U[:, :rank]
    return Q @ Q.T

def project_onto_subspace(v, P_S):
    return P_S @ v

def compute_optimal_prototypes(z, y_labels, kappa=0.1):
    N, d = z.shape
    classes = np.unique(y_labels)
    K = len(classes)
    W = np.zeros((d, K))
    for k, c in enumerate(classes):
        mask = y_labels == c
        W[:, k] = z[mask].mean(axis=0)
        W[:, k] /= np.linalg.norm(W[:, k]) + 1e-8
    for iteration in range(100):
        W_old = W.copy()
        for k, c in enumerate(classes):
            mask = y_labels == c
            neg_mask = ~mask
            logits = z @ W / kappa
            logits -= logits.max(axis=1, keepdims=True)
            pi = np.exp(logits)
            pi = pi / pi.sum(axis=1, keepdims=True)
            pos_sum = ((1 - pi[mask, k])[:, None] * z[mask]).sum(axis=0)
            neg_sum = (pi[neg_mask, k][:, None] * z[neg_mask]).sum(axis=0)
            w_raw = pos_sum - neg_sum
            W[:, k] = w_raw / (np.linalg.norm(w_raw) + 1e-8)
        if np.max(np.abs(W - W_old)) < 1e-6:
            break
    return W

def numerical_audit_theorem2(d=64, N=30, K=10, n_text_dims_outside_S=10):
    print("=" * 70)
    print("NUMERICAL AUDIT OF THEOREM 2: Modality Gap Lower Bound")
    print("=" * 70)
    print(f"Parameters: d={d}, N={N}, K={K}, extra_dims={n_text_dims_outside_S}")
    print()
    z_raw = np.random.randn(N, d)
    z = z_raw / np.linalg.norm(z_raw, axis=1, keepdims=True)
    y_labels = np.tile(np.arange(K), N // K)
    np.random.shuffle(y_labels)
    P_S = orthogonal_projector(z)
    P_S_perp = np.eye(d) - P_S
    rank_S = np.linalg.matrix_rank(P_S)
    print(f"Visual subspace S has rank {rank_S} (out of d={d})")
    W_star = compute_optimal_prototypes(z, y_labels, kappa=1.0)
    print(f"W* shape: {W_star.shape}, norms: {np.linalg.norm(W_star, axis=0)[:3]}...")
    W_star_proj = P_S @ W_star
    W_star_perp = P_S_perp @ W_star
    error_in_S = np.linalg.norm(W_star_perp) / (np.linalg.norm(W_star) + 1e-8)
    print(f"W* perpendicular component: {error_in_S:.2e} (should be ~0)")
    R_in_S = P_S @ np.random.randn(d, K)
    R_in_S = R_in_S / np.linalg.norm(R_in_S, axis=0, keepdims=True)
    R_out_S = P_S_perp @ np.random.randn(d, K)
    R_out_S = R_out_S / (np.linalg.norm(R_out_S, axis=0, keepdims=True) + 1e-8)
    alpha = n_text_dims_outside_S / d
    R = np.sqrt(1 - alpha) * R_in_S + np.sqrt(alpha) * R_out_S
    R = R / np.linalg.norm(R, axis=0, keepdims=True)
    print(f"\nText prototypes R: mixed with {alpha*100:.1f}% energy outside S")
    lhs = np.linalg.norm(R - W_star, 'fro') ** 2
    rhs = 0.0
    for k in range(K):
        proj_norm = np.linalg.norm(project_onto_subspace(R[:, k], P_S))
        rhs += 2 * (1 - proj_norm)
    print(f"\n--- Theorem 2 Inequality Check ---")
    print(f"LHS: ||R - W*||_F^2 = {lhs:.6f}")
    print(f"RHS: sum 2(1 - ||Proj_S(r_y)||_2) = {rhs:.6f}")
    print(f"LHS >= RHS? {lhs >= rhs - 1e-8}  (margin: {lhs - rhs:.2e})")
    rhs_eq6 = 0.0
    for k in range(K):
        perp_norm = np.linalg.norm(P_S_perp @ R[:, k])
        rhs_eq6 += 2 * (1 - np.sqrt(1 - perp_norm**2))
    print(f"\nEquivalent form (Eq 6):")
    print(f"RHS (Eq 6): sum 2(1 - sqrt(1 - ||Proj_S_perp(r_y)||^2)) = {rhs_eq6:.6f}")
    print(f"LHS >= RHS (Eq 6)? {lhs >= rhs_eq6 - 1e-8}  (margin: {lhs - rhs_eq6:.2e})")
    print(f"\n--- Tight Case: Text prototypes entirely in S ---")
    R_tight = P_S @ np.random.randn(d, K)
    R_tight = R_tight / np.linalg.norm(R_tight, axis=0, keepdims=True)
    lhs_tight = np.linalg.norm(R_tight - W_star, 'fro') ** 2
    rhs_tight = 0.0
    for k in range(K):
        proj_norm = np.linalg.norm(project_onto_subspace(R_tight[:, k], P_S))
        rhs_tight += 2 * (1 - proj_norm)
    print(f"LHS: {lhs_tight:.6f}, RHS: {rhs_tight:.6f}")
    print(f"RHS is ~0 as expected: {rhs_tight < 1e-6}")
    print(f"Inequality still holds: {lhs_tight >= rhs_tight - 1e-8}")
    print(f"\n--- Loose Case: Text prototypes entirely outside S ---")
    R_loose = P_S_perp @ np.random.randn(d, K)
    R_loose = R_loose / np.linalg.norm(R_loose, axis=0, keepdims=True)
    lhs_loose = np.linalg.norm(R_loose - W_star, 'fro') ** 2
    rhs_loose = 0.0
    for k in range(K):
        proj_norm = np.linalg.norm(project_onto_subspace(R_loose[:, k], P_S))
        rhs_loose += 2 * (1 - proj_norm)
    print(f"LHS: {lhs_loose:.6f}, RHS: {rhs_loose:.6f}")
    print(f"RHS is large as expected: {rhs_loose > 1.0}")
    print(f"Inequality holds: {lhs_loose >= rhs_loose - 1e-8}")
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, l, r in [("Mixed", lhs, rhs), ("Tight", lhs_tight, rhs_tight),
                         ("Loose", lhs_loose, rhs_loose)]:
        passed = l >= r - 1e-6
        all_pass = all_pass and passed
        print(f"  {name}: LHS={l:.4f} >= RHS={r:.4f} -> {'PASS' if passed else 'FAIL'}")
    print(f"\nOverall: {'ALL TESTS PASS' if all_pass else 'SOME TESTS FAILED'}")
    return all_pass

if __name__ == "__main__":
    results = []
    for trial in range(5):
        np.random.seed(trial * 1000 + 42)
        results.append(numerical_audit_theorem2())
    print("\n" + "=" * 70)
    print(f"MULTI-TRIAL SUMMARY: {sum(results)}/{len(results)} trials passed")
    print("=" * 70)
