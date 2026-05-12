"""Quick evaluation of v7 (perspective-corrected) model vs v6."""
import sys, os, json, torch, numpy as np, dill
sys.path.append('../../trajectron')
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
from utils import prediction_output_to_trajectories

DEVICE = 'cpu'
PH = 12
MAX_HL = 7
NUM_SAMPLES = 20

def load_model_and_evaluate(model_dir, epoch, test_pkl):
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

    all_ade, all_fde, all_spread = [], [], []
    n_points = 0

    with torch.no_grad():
        for scene in eval_env.scenes:
            valid_range = range(MAX_HL, scene.timesteps - PH)
            if len(valid_range) < 1:
                continue

            n_ts = min(5, len(valid_range))
            timesteps = np.linspace(valid_range.start, valid_range.stop - 1, n_ts, dtype=int)

            for ts in timesteps:
                try:
                    predictions = traj.predict(
                        scene, np.array([ts]), PH,
                        num_samples=NUM_SAMPLES,
                        min_future_timesteps=PH,
                        min_history_timesteps=1,
                        z_mode=False, gmm_mode=False, full_dist=False
                    )
                except Exception:
                    continue

                if not predictions:
                    continue

                pred_dict, hist_dict, fut_dict = prediction_output_to_trajectories(
                    predictions, scene.dt, MAX_HL, PH
                )
                if ts not in pred_dict:
                    continue

                for node in pred_dict[ts]:
                    samples = pred_dict[ts][node]   # (1, num_samples, PH, 2) or list
                    future = fut_dict[ts][node]      # (PH, 2)

                    samps = np.array(samples)
                    if samps.ndim == 4:
                        samps = samps[0]  # (num_samples, PH, 2)

                    gt = np.array(future)
                    if gt.ndim != 2 or gt.shape[0] < PH:
                        continue

                    gt = gt[:PH]
                    mean_pred = samps.mean(axis=0)

                    ade = np.mean(np.linalg.norm(mean_pred - gt, axis=1))
                    fde = np.linalg.norm(mean_pred[-1] - gt[-1])
                    spread = np.mean([np.std(samps[:, s, :], axis=0).mean()
                                      for s in range(min(PH, samps.shape[1]))])

                    all_ade.append(ade)
                    all_fde.append(fde)
                    all_spread.append(spread)
                    n_points += 1

    return {
        'n_points': n_points,
        'ade_mean': np.mean(all_ade) if all_ade else float('nan'),
        'fde_mean': np.mean(all_fde) if all_fde else float('nan'),
        'spread_mean': np.mean(all_spread) if all_spread else float('nan'),
        'ade_std': np.std(all_ade) if all_ade else float('nan'),
        'fde_std': np.std(all_fde) if all_fde else float('nan'),
        'spread_std': np.std(all_spread) if all_spread else float('nan'),
    }


models = [
    ('v6 (aug, no persp, scale=100) ep30',
     'models/models_09_Mar_2026_16_10_23_uav_ft_v6_aug', 30,
     'processed/uav_finetune_test.pkl'),
    ('v8 (aug + persp, scale=1) ep30',
     'models/models_10_Mar_2026_00_42_50_uav_ft_v8_persp', 30,
     'processed/uav_persp_v2_test.pkl'),
    ('v8 (aug + persp, scale=1) ep50',
     'models/models_10_Mar_2026_00_42_50_uav_ft_v8_persp', 50,
     'processed/uav_persp_v2_test.pkl'),
]

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

for name, mdir, ep, tpkl in models:
    print(f"\n>>> {name}")
    r = load_model_and_evaluate(mdir, ep, tpkl)
    print(f"    Points evaluated: {r['n_points']}")
    print(f"    ADE  (mean): {r['ade_mean']:.4f} +/- {r['ade_std']:.4f}")
    print(f"    FDE  (mean): {r['fde_mean']:.4f} +/- {r['fde_std']:.4f}")
    print(f"    Spread (mean): {r['spread_mean']:.4f} +/- {r['spread_std']:.4f}")

print("\n" + "=" * 70)
print("NOTE: v6 and v7 use DIFFERENT coordinate systems, so ADE/FDE are")
print("NOT directly comparable. SPREAD shows prediction confidence.")
print("=" * 70)
