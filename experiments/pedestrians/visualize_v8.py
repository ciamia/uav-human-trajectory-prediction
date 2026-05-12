"""Visualize v8 predictions and compare with v6."""
import sys, os, json, torch, numpy as np, dill
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.append('../../trajectron')
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
from utils import prediction_output_to_trajectories

DEVICE = 'cpu'
PH = 12
MAX_HL = 7
NUM_SAMPLES = 20


def load_and_predict(model_dir, epoch, test_pkl):
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
            n_ts = min(3, len(valid_range))
            timesteps = np.linspace(valid_range.start, valid_range.stop - 1, n_ts, dtype=int)

            for ts in timesteps:
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
                    gt = np.array(fut_dict[ts][node])[:PH]
                    hist = np.array(hist_dict[ts][node])
                    results.append({
                        'scene': scene.name, 'ts': ts,
                        'samples': samps, 'gt': gt, 'hist': hist
                    })
    return results


configs = {
    'v6 (no persp)': ('models/models_09_Mar_2026_16_10_23_uav_ft_v6_aug', 30,
                       'processed/uav_finetune_test.pkl'),
    'v8 (persp)': ('models/models_10_Mar_2026_00_42_50_uav_ft_v8_persp', 50,
                    'processed/uav_persp_v2_test.pkl'),
}

all_results = {}
for name, (mdir, ep, tpkl) in configs.items():
    print(f"Loading {name}...")
    all_results[name] = load_and_predict(mdir, ep, tpkl)
    print(f"  {len(all_results[name])} predictions")

n_show = min(6, min(len(r) for r in all_results.values()))
fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
fig.suptitle('v6 (pixel coords) vs v8 (perspective corrected)', fontsize=14, fontweight='bold')

for row_idx, (name, results) in enumerate(all_results.items()):
    for col_idx in range(n_show):
        ax = axes[row_idx, col_idx] if n_show > 1 else axes[row_idx]
        r = results[col_idx]

        ax.plot(r['hist'][:, 0], r['hist'][:, 1], 'b-o', ms=3, lw=1.5,
                label='History', zorder=5)
        ax.plot(r['gt'][:, 0], r['gt'][:, 1], 'g-s', ms=3, lw=1.5,
                label='Ground Truth', zorder=5)

        for s in range(min(NUM_SAMPLES, r['samples'].shape[0])):
            ax.plot(r['samples'][s, :, 0], r['samples'][s, :, 1],
                    'r-', alpha=0.15, lw=0.8)

        mean_pred = r['samples'].mean(axis=0)
        ax.plot(mean_pred[:, 0], mean_pred[:, 1], 'r-o', ms=3, lw=2,
                label='Mean Pred', zorder=4,
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])

        ade = np.mean(np.linalg.norm(mean_pred - r['gt'], axis=1))
        spread = np.mean([np.std(r['samples'][:, s, :], axis=0).mean()
                         for s in range(r['samples'].shape[1])])
        ax.set_title(f"{r['scene']}\nADE={ade:.2f} Spr={spread:.2f}", fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.set_ylabel(name, fontsize=11, fontweight='bold')
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('v6_vs_v8_comparison.png', dpi=150, bbox_inches='tight')
print("Saved v6_vs_v8_comparison.png")

# Also make a detailed v8 plot
results_v8 = all_results['v8 (persp)']
n_v8 = min(9, len(results_v8))
fig2, axes2 = plt.subplots(3, 3, figsize=(15, 15))
fig2.suptitle('v8 (Perspective Corrected, scale=1) - Detailed Predictions',
              fontsize=14, fontweight='bold')

for idx in range(n_v8):
    ax = axes2[idx // 3, idx % 3]
    r = results_v8[idx]

    ax.plot(r['hist'][:, 0], r['hist'][:, 1], 'b-o', ms=4, lw=2,
            label='History', zorder=5)
    ax.plot(r['gt'][:, 0], r['gt'][:, 1], 'g-s', ms=4, lw=2,
            label='Ground Truth', zorder=5)

    for s in range(min(NUM_SAMPLES, r['samples'].shape[0])):
        ax.plot(r['samples'][s, :, 0], r['samples'][s, :, 1],
                'r-', alpha=0.2, lw=1)

    mean_pred = r['samples'].mean(axis=0)
    ax.plot(mean_pred[:, 0], mean_pred[:, 1], 'r-o', ms=4, lw=2.5,
            label='Mean Pred', zorder=4,
            path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    ade = np.mean(np.linalg.norm(mean_pred - r['gt'], axis=1))
    fde = np.linalg.norm(mean_pred[-1] - r['gt'][-1])
    spread = np.mean([np.std(r['samples'][:, s, :], axis=0).mean()
                     for s in range(r['samples'].shape[1])])

    ax.set_title(f"{r['scene']} t={r['ts']}\nADE={ade:.3f}  FDE={fde:.3f}  Spread={spread:.3f}",
                 fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=8)

for idx in range(n_v8, 9):
    axes2[idx // 3, idx % 3].set_visible(False)

plt.tight_layout()
plt.savefig('v8_detailed_predictions.png', dpi=150, bbox_inches='tight')
print("Saved v8_detailed_predictions.png")
