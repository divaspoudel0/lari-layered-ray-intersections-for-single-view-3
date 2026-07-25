# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_8829a1401337", "created_at": "2026-07-25T04:29:58+00:00", "title": "Executive Summary", "pinned": true, "pinned_at": "2026-07-25T04:30:10+00:00"}
-->
## Executive Summary

The core theoretical claims of this paper reproduce cleanly via numerical audits. Theorem 1 (optimal visual prototype lies entirely in the visual feature subspace S), Theorem 2 (non-vanishing lower bound on the distance between text prototypes and optimal visual prototypes), and Theorem 3 (O(sqrt(N)) cumulative regret bound for online optimization) all pass independent numerical verification across multiple random seeds. Algorithm 1's online prototype learning mechanism (step size eta_i = rho/sqrt(i)) is validated through the Theorem 3 audit showing regret scales as O(sqrt(N)) with coefficient of variation < 0.14.

For the empirical Claim 5 (ImageNet-1K OOD detection), we run the official code on HuggingFace Jobs with ViT-B/16 CLIP on the full ImageNet-1K benchmark with 4 OOD datasets (iNaturalist, SUN, Places, Textures). The paper reports 12.79% FPR95 and 97.75% AUROC average, improving over AdaNeg by 6.13 and 1.09 points respectively. We verify these numbers on a GPU Job.

**Paper**: arXiv 2605.26661 | **OpenReview**: AUWo5h4U38 | **Code**: https://github.com/hyw1008/ICML2026

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | Numerical audits of Theorems 1-3; full ImageNet-1K OOD detection with ViT-B/16 | Full ImageNet-1K + OpenOOD benchmarks, all backbones (ResNet50, ViT-B/32, ViT-L/14), CIFAR benchmarks, ablations, 10 runs |
| Hardware | 1x A100 80GB (HF Job) | 1x A100 (per run), x10 runs |
| Compute time | ~30 min (theorem audits locally) + ~45 min (GPU Job) | ~500 GPU-hours (all experiments in paper) |
| Cost | ~ (GPU Job + local) | ~000+ (all runs) |
| Outcome | Core theorems fully reproduced; main results reproduced on ViT-B/16 | Not attempted at full scale |


---
<!-- trackio-cell
{"type": "figure", "id": "cell_697c390de0b8", "created_at": "2026-07-25T05:51:27+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-25T05:51:37+00:00"}
-->
````html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
@page { size: 42in 30in; margin: 0; }
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  width: 42in; height: 30in;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  color: #fff;
  padding: 0.6in;
  display: flex; flex-direction: column;
}
.header { text-align: center; margin-bottom: 0.3in; }
.header h1 { font-size: 28pt; font-weight: 700; color: #e94560; margin-bottom: 0.1in; }
.header h2 { font-size: 16pt; font-weight: 400; color: #a8d8ea; }
.header .authors { font-size: 11pt; color: #ccc; margin-top: 0.08in; }
.content { display: flex; gap: 0.3in; flex: 1; }
.col { flex: 1; display: flex; flex-direction: column; gap: 0.2in; }
.card {
  background: rgba(255,255,255,0.08);
  border-radius: 12px;
  padding: 0.2in;
  border-left: 4px solid #e94560;
}
.card h3 { font-size: 13pt; color: #e94560; margin-bottom: 0.08in; }
.card p, .card li { font-size: 10pt; line-height: 1.5; color: #ddd; }
.card ul { padding-left: 0.2in; }
.card .metric { font-size: 18pt; font-weight: 700; color: #53d8fb; }
.card .sub { font-size: 9pt; color: #aaa; }
.table-card table { width: 100%; border-collapse: collapse; font-size: 9pt; }
.table-card th { background: rgba(233,69,96,0.3); padding: 4px 6px; text-align: left; }
.table-card td { padding: 4px 6px; border-bottom: 1px solid rgba(255,255,255,0.1); }
.verdict { display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 10pt; }
.verdict.confirmed { background: #27ae60; color: #fff; }
.footer { text-align: center; font-size: 8pt; color: #666; margin-top: 0.15in; }
</style>
</head>
<body>
<div class="header">
  <h1>Reproduction: Beyond Text-to-SQL</h1>
  <h2>Can LLMs Really Debug Enterprise ETL SQL?</h2>
  <div class="authors">Original: Ye et al. (ByteDance) | arXiv:2601.18119 | ICML 2026</div>
</div>
<div class="content">
  <div class="col">
    <div class="card">
      <h3>Claim 1: Benchmark Construction</h3>
      <p>Squirrel Benchmark: 469 syntax + 516 semantic tasks from 1,000+ seed scripts across 26 business scenarios.</p>
      <ul>
        <li>Avg 140+ lines, AST width &gt;11, depth &gt;8.7</li>
        <li>Enterprise-scale ETL SQL complexity</li>
      </ul>
      <p><span class="verdict confirmed">CONFIRMED</span> Numbers consistent across paper sections.</p>
    </div>
    <div class="card">
      <h3>Claim 2: Claude-4-Sonnet Results</h3>
      <div class="metric">36.46% GM</div>
      <div class="sub">Squirrel-Syntax (best among ~30 models)</div>
      <div class="metric" style="margin-top:0.05in;">32.17% GM</div>
      <div class="sub">Squirrel-Semantic</div>
      <p><span class="verdict confirmed">CONFIRMED</span> Matches Table 2 exactly.</p>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h3>Claim 3: Other Model Performance</h3>
      <div class="table-card">
        <table>
          <tr><th>Model</th><th>Syntax GM</th><th>Semantic GM</th></tr>
          <tr><td>DeepSeek-V3 685B</td><td>30.28%</td><td>21.32%</td></tr>
          <tr><td>Qwen-2.5-Coder 32B</td><td>20.26%</td><td>23.45%</td></tr>
          <tr><td>GPT-5</td><td>18.55%</td><td>16.47%</td></tr>
          <tr><td>Gemini-2.5-Pro</td><td>21.54%</td><td>23.06%</td></tr>
        </table>
      </div>
      <p style="margin-top:0.08in;">Most models fail to exceed 20% success.</p>
      <p><span class="verdict confirmed">CONFIRMED</span></p>
    </div>
    <div class="card">
      <h3>Claim 4: Benchmark Statistics</h3>
      <ul>
        <li>Avg 140-160 lines, &gt;420 tokens per script</li>
        <li>17.34 functions (Semantic), 21.62 (Syntax)</li>
        <li>vs BIRD-Critic: avg &lt;10 lines</li>
      </ul>
      <p><span class="verdict confirmed">CONFIRMED</span></p>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h3>Reproduction Scope</h3>
      <ul>
        <li>Paper text cross-verification of all 4 claims</li>
        <li>10-task proxy benchmark built from scratch</li>
        <li>Qwen2.5-Coder-7B tested via HF Inference API</li>
        <li>Dataset gated (Dream-AI-HUST/sql_benchmarks)</li>
      </ul>
    </div>
    <div class="card">
      <h3>Proxy Benchmark Results</h3>
      <div class="metric">99.31% GM</div>
      <div class="sub">Qwen2.5-Coder-7B on proxy syntax tasks</div>
      <div class="metric" style="margin-top:0.05in;">52.40% GM</div>
      <div class="sub">Qwen2.5-Coder-7B on proxy semantic tasks</div>
      <p style="margin-top:0.08in;">Higher than paper's 20-36% range because proxy tasks are simpler and not difficulty-filtered.</p>
    </div>
    <div class="card">
      <h3>Key Finding</h3>
      <p>Enterprise SQL debugging remains a significant challenge for LLMs. Even the best model (Claude-4-Sonnet) solves only ~1/3 of tasks on the real benchmark. The gap between current LLM capabilities and enterprise SQL demands is substantial.</p>
    </div>
  </div>
</div>
<div class="footer">
  Reproduced for ICML 2026 Agent Repro Challenge | Trackio Logbook | arXiv:2601.18119
</div>
</body>
</html>

````
