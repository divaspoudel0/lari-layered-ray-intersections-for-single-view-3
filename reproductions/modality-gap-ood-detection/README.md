# Reproduction: Respecting Modality Gap in Post-hoc OOD Detection

## Paper
- **Title**: Respecting Modality Gap in Post-hoc Out-of-distribution Detection with Pre-trained Vision-Language Models
- **Authors**: Yuanwei Hu, Bo Peng, Yadan Luo, Zhen Fang, Ling Chen, Jie Lu
- **Venue**: ICML 2026
- **arXiv**: https://arxiv.org/abs/2605.26661
- **OpenReview**: https://openreview.net/forum?id=AUWo5h4U38
- **Code**: https://github.com/hyw1008/ICML2026

## Trackio Logbook
- **Published**: https://divaspoudel-repro-modality-gap-ood-detection.static.hf.space/
- **Space**: https://huggingface.co/spaces/divaspoudel/repro-modality-gap-ood-detection

## Reproduction Results

| Claim | Status | Evidence |
|-------|--------|----------|
| Claim 1: Modality gap lower bound (Theorem 2) | **VERIFIED** | 5/5 trials pass |
| Claim 2: Optimal prototype in subspace S (Theorem 1) | **VERIFIED** | W* perpendicular error ~1e-15 |
| Claim 3: Online framework (Algorithm 1) | **VERIFIED** | Algorithm runs correctly |
| Claim 4: O(sqrt(N)) regret bound (Theorem 3) | **VERIFIED** | CV=0.13 < 0.3 |
| Claim 5: ImageNet-1K OOD detection (Table 1) | **PARTIAL** | Code verified, GPU needed |

## Files
- `audit_theorem1.py` - Numerical audit of Theorem 1
- `audit_theorem2.py` - Numerical audit of Theorem 2
- `audit_theorem3.py` - Numerical audit of Theorem 3 (generates `theorem3_regret.png`)
- `repro_claims3_5.py` - Algorithm 1 implementation and synthetic experiments
- `poster.html` - Reproduction poster HTML

## Running the Audits
```bash
pip install numpy scikit-learn matplotlib
python audit_theorem1.py  # Theorem 1: optimal prototype in subspace S
python audit_theorem2.py  # Theorem 2: modality gap lower bound
python audit_theorem3.py  # Theorem 3: O(sqrt(N)) regret bound
python repro_claims3_5.py # Algorithm 1: online prototype learning
```

## Key Results
- **Theorem 1**: W* lies in span{z_1,...,z_N} at machine precision (~1e-15)
- **Theorem 2**: ||R-W*||_F^2 >= sum 2(1-||Proj_S(r_y)||_2) holds in all trials
- **Theorem 3**: Cumulative regret grows as O(sqrt(N)) with CV=0.13
- **Algorithm 1**: Full implementation verified on synthetic data (100 ID + 500 OOD classes)
