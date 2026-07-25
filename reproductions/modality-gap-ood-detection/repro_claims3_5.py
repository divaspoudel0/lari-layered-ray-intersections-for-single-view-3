"""
Self-contained reproduction of Claims 3 & 5:
Algorithm 1: Online Prototype Learning for OOD Detection (simplified version).
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def make_synthetic_data(num_id=100, num_ood=500, samples_per_class=5, d=512):
    """Generate synthetic visual features and text prototypes."""
    id_centroids = F.normalize(torch.randn(num_id, d), dim=1)
    id_features = []
    id_labels = []
    for c in range(num_id):
        noise = torch.randn(samples_per_class, d) * 0.3
        feats = F.normalize(id_centroids[c:c+1] + noise, dim=1)
        id_features.append(feats)
        id_labels.extend([c] * samples_per_class)
    id_features = torch.cat(id_features)
    id_labels = torch.tensor(id_labels)
    ood_features = F.normalize(torch.randn(samples_per_class * num_ood, d), dim=1)
    ood_labels = torch.full((len(ood_features),), -1, dtype=torch.long)
    id_text = F.normalize(torch.randn(num_id, d), dim=1)
    ood_text = F.normalize(torch.randn(num_ood, d), dim=1)
    return {
        'id_features': id_features, 'id_labels': id_labels,
        'ood_features': ood_features, 'ood_labels': ood_labels,
        'id_text': id_text, 'ood_text': ood_text,
        'num_id': num_id, 'num_ood': num_ood, 'd': d,
    }


def run_algorithm1(data, rho=0.1, kappa=0.05, tau=0.01, beta=0.95, num_runs=3):
    """Run Algorithm 1: Online Prototype Learning for OOD Detection."""
    id_feat = data['id_features']
    id_labels = data['id_labels']
    ood_feat = data['ood_features']
    ood_labels = data['ood_labels']
    id_text = data['id_text']
    ood_text = data['ood_text']
    num_id = data['num_id']
    num_ood = data['num_ood']
    d = data['d']
    all_aurocs, all_fpr95s = [], []
    for run in range(num_runs):
        W_id = id_text.t().clone()
        W_ood = ood_text.t().clone()
        c_pos, c_neg = 0, 0
        all_feat = torch.cat([id_feat, ood_feat])
        all_lab = torch.cat([id_labels, ood_labels])
        perm = torch.randperm(len(all_feat))
        all_feat = all_feat[perm]
        all_lab = all_lab[perm]
        scores, true_labels = [], []
        N = len(all_feat)
        for i in range(N):
            x = all_feat[i]
            y_true = all_lab[i].item()
            W_all = torch.cat([W_id, W_ood], dim=1)
            text_logits = x @ W_all / tau
            text_probs = F.softmax(text_logits, dim=0)
            id_prob = text_probs[:num_id].sum().item()
            is_pos = id_prob >= beta
            is_neg = id_prob <= (1 - beta)
            if is_pos:
                c_pos += 1
            if is_neg:
                c_neg += 1
            if is_pos and c_pos > 0:
                vis_logits = x @ W_id / kappa
                vis_probs = F.softmax(vis_logits, dim=0)
                text_slice = text_probs[:num_id] / (text_probs[:num_id].sum() + 1e-8)
                vis_slice = vis_probs / (vis_probs.sum() + 1e-8)
                eta = rho / math.sqrt(c_pos)
                grad = torch.outer(x, vis_slice - text_slice) / kappa
                W_id = W_id - eta * grad
                W_id = F.normalize(W_id, dim=0)
            if is_neg and c_neg > 0:
                vis_logits = x @ W_ood / kappa
                vis_probs = F.softmax(vis_logits, dim=0)
                text_slice = text_probs[num_id:] / (text_probs[num_id:].sum() + 1e-8)
                vis_slice = vis_probs / (vis_probs.sum() + 1e-8)
                eta = rho / math.sqrt(c_neg)
                grad = torch.outer(x, vis_slice - text_slice) / kappa
                W_ood = W_ood - eta * grad
                W_ood = F.normalize(W_ood, dim=0)
            score_id = x @ W_id / tau
            score_ood = x @ W_ood / tau
            score_groups = min(20, num_ood)
            ood_dim = num_ood - (num_ood % score_groups)
            score_ood_trimmed = score_ood[:ood_dim]
            group_scores = []
            for chunk in score_ood_trimmed.chunk(score_groups, dim=0):
                cat = torch.cat([score_id, chunk], dim=0)
                s = F.softmax(cat, dim=0)[:num_id].sum().item()
                group_scores.append(s)
            final_score = np.mean(group_scores)
            scores.append(final_score)
            true_labels.append(0 if y_true >= 0 else 1)
        scores = np.array(scores)
        true_labels = np.array(true_labels)
        auroc = roc_auc_score(true_labels, -scores) * 100
        threshold = np.percentile(scores[true_labels == 0], 5)
        fpr95 = np.mean(scores[true_labels == 1] < threshold) * 100
        all_aurocs.append(auroc)
        all_fpr95s.append(fpr95)
        print(f"  Run {run+1}: AUROC={auroc:.2f}%, FPR95={fpr95:.2f}%")
    return all_aurocs, all_fpr95s


if __name__ == "__main__":
    print("=" * 70)
    print("Claim 3+5: Algorithm 1 Online Prototype Learning (Synthetic Data)")
    print("=" * 70)
    print("Config: num_id=100, num_ood=500, d=512, rho=0.1, kappa=0.05, tau=0.01")
    print()
    data = make_synthetic_data(num_id=100, num_ood=500, samples_per_class=5, d=512)
    aurocs, fpr95s = run_algorithm1(data, rho=0.1, kappa=0.05, tau=0.01, beta=0.95, num_runs=3)
    print(f"\n--- Summary (3 runs) ---")
    print(f"AUROC:  {np.mean(aurocs):.2f}% +/- {np.std(aurocs):.2f}%")
    print(f"FPR95:  {np.mean(fpr95s):.2f}% +/- {np.std(fpr95s):.2f}%")
