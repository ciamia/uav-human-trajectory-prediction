"""Fair comparison of v6 vs v8 using scale-invariant metrics."""
import sys, os, json, torch, numpy as np, dill
import matplotlib.pyplot as plt

sys.path.append('../../trajectron')
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
from utils import prediction_output_to_trajectories

DEVICE = 'cpu'
PH = 12
MAX_HL = 7
NUM_SAMPLES = 20


def evaluate_model(model_dir, epoch, test_pkl):
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        hp = json.load(f)
    with open(test_pkl, 'rb') as f:
        eval_env = dill.load(f, encoding='latin1')

    mr = ModelRegistrar(model_dir, DEVICE)
    traj = Trajectron(mr, hp, None, DEVICE)
    traj.set_environment(eval_env)
    traj.set_annealing_params()

    cp = torch.load(os.path.join(model_dir, f'model_registrar-{epoch}.pt'),
                    map_location=DEVICE)
    mr.model_dict.load_state_dict(cp.state_dict())
    mr.eval()

    results = []

    with torch.no_grad():
        for scene in eval_env.scenes:
            valid_range = range(MAX_HL, scene.timesteps - PH)
            if len(valid_range) < 1:
                continue
            for ts in valid_range:
                try:
                    predictions = traj.predict(
                        scene, np.array([ts]), PH,
                        num_samples=NUM_SAMPLES, min_future_timesteps=PH,
                        min_history_timesteps=1, z_mode=False, gmm_mode=False,
                        full_dist=False)
                except Exception:
                    continue
                if not predictions:
                    continue
                pred_dict, hist_dict, fut_dict = prediction_output_to_trajectories(
                    predictions, scene.dt, MAX_HL, PH)
                if ts not in pred_dict:
                    continue
                for node in pred_dict[ts]:
                    samps = np.array(pred_dict[ts][node])
                    if samps.ndim == 4:
                        samps = samps[0]
                    gt = np.array(fut_dict[ts][node])
                    hist = np.array(hist_dict[ts][node])
                    if gt is None or len(gt) < PH:
                        continue
                    gt = gt[:PH]
                    results.append({
                        'scene': scene.name, 'ts': ts,
                        'samples': samps, 'gt': gt, 'hist': hist
                    })

    metrics = {
        'n': len(results),
        'norm_ade_b20': [], 'norm_ade_mean': [],
        'norm_fde_b20': [], 'norm_fde_mean': [],
        'direction_acc': [],
        'relative_spread': [],
        'raw_ade_b20': [], 'raw_ade_mean': [],
        'raw_spread': [],
    }

    for r in results:
        samps = r['samples']
        gt = r['gt']
        hist = r['hist']

        gt_displacements = np.diff(gt, axis=0)
        gt_total_length = np.sum(np.linalg.norm(gt_displacements, axis=1))
        gt_net_displacement = np.linalg.norm(gt[-1] - gt[0])

        if gt_total_length < 1e-6:
            continue

        errs = np.sqrt(((samps - gt[None]) ** 2).sum(axis=-1))
        ade_per = errs.mean(axis=1)
        fde_per = errs[:, -1]

        mean_pred = samps.mean(axis=0)
        mean_errs = np.sqrt(((mean_pred - gt) ** 2).sum(axis=-1))

        # Normalized ADE/FDE (relative to GT trajectory length)
        metrics['norm_ade_b20'].append(ade_per.min() / gt_total_length)
        metrics['norm_ade_mean'].append(mean_errs.mean() / gt_total_length)
        metrics['norm_fde_b20'].append(fde_per.min() / gt_total_length)
        metrics['norm_fde_mean'].append(mean_errs[-1] / gt_total_length)

        # Direction accuracy: cosine similarity between predicted and GT displacement
        pred_disp = mean_pred[-1] - mean_pred[0]
        gt_disp = gt[-1] - gt[0]
        pred_norm = np.linalg.norm(pred_disp)
        gt_norm = np.linalg.norm(gt_disp)
        if pred_norm > 1e-6 and gt_norm > 1e-6:
            cos_sim = np.dot(pred_disp, gt_disp) / (pred_norm * gt_norm)
            metrics['direction_acc'].append(cos_sim)

        # Relative spread (spread / GT net displacement)
        endpoint_spread = np.sqrt((samps[:, -1, :].std(axis=0) ** 2).sum())
        if gt_net_displacement > 1e-6:
            metrics['relative_spread'].append(endpoint_spread / gt_net_displacement)

        metrics['raw_ade_b20'].append(ade_per.min())
        metrics['raw_ade_mean'].append(mean_errs.mean())
        metrics['raw_spread'].append(endpoint_spread)

    return metrics


models = [
    ('v6\n(no perspective\nscale=100)',
     'models/models_09_Mar_2026_16_10_23_uav_ft_v6_aug', 30,
     'processed/uav_finetune_test.pkl'),
    ('v8\n(perspective corr.\nscale=1)',
     'models/models_10_Mar_2026_00_42_50_uav_ft_v8_persp', 50,
     'processed/uav_persp_v2_test.pkl'),
]

all_metrics = {}
for name, mdir, ep, tpkl in models:
    print(f'Evaluating {name.replace(chr(10), " ")}...')
    all_metrics[name] = evaluate_model(mdir, ep, tpkl)
    m = all_metrics[name]
    print(f'  {m["n"]} predictions, {len(m["norm_ade_b20"])} with valid GT length')

# ─── Print comparison ───
print('\n' + '=' * 75)
print('FAIR COMPARISON: Scale-Invariant Metrics')
print('=' * 75)

header = f'{"Metric":<35s}'
for name in all_metrics:
    short = name.split('\n')[0]
    header += f' | {short:>12s}'
print(header)
print('-' * 75)

metric_labels = [
    ('Normalized ADE (B20) ↓', 'norm_ade_b20'),
    ('Normalized ADE (Mean) ↓', 'norm_ade_mean'),
    ('Normalized FDE (B20) ↓', 'norm_fde_b20'),
    ('Normalized FDE (Mean) ↓', 'norm_fde_mean'),
    ('Direction Accuracy ↑', 'direction_acc'),
    ('Relative Spread ↓', 'relative_spread'),
]

for label, key in metric_labels:
    row = f'{label:<35s}'
    for name in all_metrics:
        vals = all_metrics[name][key]
        if vals:
            row += f' | {np.mean(vals):12.4f}'
        else:
            row += f' | {"N/A":>12s}'
    print(row)

print('-' * 75)
print('↓ = lower is better, ↑ = higher is better')
print('Normalized = metric / GT trajectory length (scale-invariant)')
print('Direction Accuracy = cosine similarity (-1 to 1, 1 = perfect)')

# ─── Visualization ───
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('v6 vs v8: Scale-Invariant Comparison\n(Which model predicts better?)',
             fontsize=14, fontweight='bold')

plot_configs = [
    ('Normalized ADE (Best-of-20)', 'norm_ade_b20', 'lower = better', '#3498db'),
    ('Normalized ADE (Mean Pred)', 'norm_ade_mean', 'lower = better', '#e74c3c'),
    ('Normalized FDE (Best-of-20)', 'norm_fde_b20', 'lower = better', '#9b59b6'),
    ('Normalized FDE (Mean Pred)', 'norm_fde_mean', 'lower = better', '#e67e22'),
    ('Direction Accuracy', 'direction_acc', 'higher = better', '#2ecc71'),
    ('Relative Spread', 'relative_spread', 'lower = better', '#f39c12'),
]

model_names_short = ['v6', 'v8']
model_keys = list(all_metrics.keys())

for ax, (title, key, note, color) in zip(axes.flat, plot_configs):
    vals = []
    for name in model_keys:
        v = all_metrics[name][key]
        vals.append(np.mean(v) if v else 0)

    bars = ax.bar(model_names_short, vals, color=[color, color], edgecolor='white')
    bars[0].set_alpha(0.5)
    bars[1].set_alpha(1.0)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')

    ax.set_title(f'{title}\n({note})', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)

    # Winner indicator
    if key == 'direction_acc':
        winner = 'v8' if vals[1] > vals[0] else 'v6'
        improvement = abs(vals[1] - vals[0]) / max(abs(vals[0]), 1e-6) * 100
    else:
        winner = 'v8' if vals[1] < vals[0] else 'v6'
        improvement = abs(vals[0] - vals[1]) / max(abs(vals[0]), 1e-6) * 100

    ax.set_xlabel(f'Winner: {winner} ({improvement:.1f}% better)', fontsize=10,
                  color='#27ae60' if winner == 'v8' else '#e74c3c', fontweight='bold')

plt.tight_layout()
plt.savefig('results_v8/v6_vs_v8_fair_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('\nSaved: results_v8/v6_vs_v8_fair_comparison.png')

# ─── Distribution comparison ───
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle('Distribution Comparison: v6 vs v8', fontsize=14, fontweight='bold')

dist_configs = [
    ('Normalized ADE (Mean)', 'norm_ade_mean'),
    ('Direction Accuracy', 'direction_acc'),
    ('Relative Spread', 'relative_spread'),
]

for ax, (title, key) in zip(axes2, dist_configs):
    for i, name in enumerate(model_keys):
        vals = all_metrics[name][key]
        short = model_names_short[i]
        if vals:
            ax.hist(vals, bins=25, alpha=0.5, label=f'{short} (mean={np.mean(vals):.3f})',
                    edgecolor='white')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results_v8/v6_vs_v8_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: results_v8/v6_vs_v8_distributions.png')
