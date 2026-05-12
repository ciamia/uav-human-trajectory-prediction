"""Debug: check what predict returns for v7."""
import sys, os, json, torch, numpy as np, dill
sys.path.append('../../trajectron')
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
from utils import prediction_output_to_trajectories
from environment import NodeTypeEnum

DEVICE = 'cpu'
PH = 12
NUM_SAMPLES = 20

model_dir = 'models/models_10_Mar_2026_00_10_49_uav_ft_v7_persp'
epoch = 30
test_pkl = 'processed/uav_persp_aug_test.pkl'

with open(os.path.join(model_dir, 'config.json'), 'r') as f:
    hp = json.load(f)
with open(test_pkl, 'rb') as f:
    eval_env = dill.load(f, encoding='latin1')

mr = ModelRegistrar(model_dir, DEVICE)
traj = Trajectron(mr, hp, None, DEVICE)
traj.set_environment(eval_env)
traj.set_annealing_params()

cp = torch.load(os.path.join(model_dir, f'model_registrar-{epoch}.pt'), map_location=DEVICE)
mr.model_dict.load_state_dict(cp.state_dict())
mr.eval()

scene = eval_env.scenes[0]
print(f"Scene: {scene.name}, timesteps: {scene.timesteps}, nodes: {len(scene.nodes)}")
print(f"Node types: {[str(n.type) for n in scene.nodes]}")
print(f"Node IDs: {[n.id for n in scene.nodes]}")

node = scene.nodes[0]
print(f"\nNode data type: {type(node.data)}")
if hasattr(node.data, 'data'):
    d = node.data.data
    print(f"  inner data type: {type(d)}, shape: {d.shape if hasattr(d,'shape') else 'N/A'}")
    if hasattr(d, 'shape') and len(d.shape) == 2:
        print(f"  first 3 rows: {d[:3]}")
print(f"  first_timestep: {node.first_timestep}")

for t in [7, 10, 15, 20]:
    if t >= scene.timesteps - PH:
        print(f"\nt={t}: skip (scene too short)")
        continue
    print(f"\nt={t}:")
    try:
        preds = traj.predict(scene, np.array([t]), np.array([PH]),
                             num_samples=NUM_SAMPLES, min_future_timesteps=PH)
        print(f"  predict returned type: {type(preds)}")
        if preds is None:
            print("  preds is None!")
            continue
        if isinstance(preds, dict):
            print(f"  preds keys: {list(preds.keys())}")
            for k, v in preds.items():
                print(f"    {k}: type={type(v)}")
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        print(f"      {k2}: type={type(v2)}")
                        if isinstance(v2, torch.Tensor):
                            print(f"        shape={v2.shape}")
                        elif isinstance(v2, np.ndarray):
                            print(f"        shape={v2.shape}")
        elif isinstance(preds, (list, tuple)):
            print(f"  preds len: {len(preds)}")
            for i, p in enumerate(preds[:3]):
                print(f"  [{i}]: type={type(p)}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
    break
