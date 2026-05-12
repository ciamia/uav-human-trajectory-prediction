#!/usr/bin/env python
"""
Object detection / video CSV'yi Trajectron++'ın kullandığı .pkl formatına çevirir.

Kullanım:
  cd experiments/pedestrians
  python csv_to_trajectron_data.py --csv path/to/detections.csv --output ../processed/my_video_test.pkl

CSV formatı: Her satır bir frame'de bir kişinin koordinatı.
  - Varsayılan kolonlar: frame_id, track_id, pos_x, pos_y
  - Kendi kolon isimleriniz varsa: --frame_col, --track_col, --x_col, --y_col ile belirtin.
  - Eksik frame'ler (bir track'ta atlama varsa) lineer interpolasyonla doldurulur.
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
import dill

sys.path.append("../../trajectron")
from environment import Environment, Scene, Node
from environment import derivative_of
from utils import maybe_makedirs

# Trajectron pedestrian env ayarları (process_data.py ile uyumlu)
DT = 0.4
DATA_COLUMNS = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
STANDARDIZATION = {
    'PEDESTRIAN': {
        'position': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1}},
        'velocity': {'x': {'mean': 0, 'std': 2}, 'y': {'mean': 0, 'std': 2}},
        'acceleration': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1}}
    }
}


def fill_gaps(df, frame_col, x_col, y_col):
    """Bir track için eksik frame'leri lineer interpolasyonla doldurur."""
    df = df.sort_values(frame_col)
    frames = df[frame_col].values
    x = df[x_col].values
    y = df[y_col].values
    f_min, f_max = int(frames.min()), int(frames.max())
    full_frames = np.arange(f_min, f_max + 1, dtype=int)
    if len(full_frames) == len(frames) and np.all(frames == full_frames):
        return df
    # Interpolate
    x_full = np.interp(full_frames.astype(float), frames, x)
    y_full = np.interp(full_frames.astype(float), frames, y)
    return pd.DataFrame({frame_col: full_frames, x_col: x_full, y_col: y_full})


def main():
    parser = argparse.ArgumentParser(description='CSV detection verisini Trajectron++ .pkl formatına çevirir.')
    parser.add_argument('--csv', required=True, help='Detection CSV dosya yolu')
    parser.add_argument('--output', required=True, help='Çıktı .pkl dosya yolu (örn: ../processed/my_video_test.pkl)')
    parser.add_argument('--sep', default=',', help='CSV ayırıcı (varsayılan: virgül)')
    parser.add_argument('--frame_col', default='frame_id', help='Frame numarası kolon adı')
    parser.add_argument('--track_col', default='track_id', help='Track / kişi ID kolon adı')
    parser.add_argument('--x_col', default='pos_x', help='X koordinat kolon adı')
    parser.add_argument('--y_col', default='pos_y', help='Y koordinat kolon adı')
    parser.add_argument('--center', action='store_true', help='Pozisyonları ortala (tüm verinin ortalamasını çıkar)')
    parser.add_argument('--scale', type=float, default=1.0, help='x,y koordinatlarını bu çarpanla ölçekle (örn. 100)')
    parser.add_argument('--dt', type=float, default=DT, help='Zaman adımı (saniye), varsayılan %.2f' % DT)
    parser.add_argument('--min_track_length', type=int, default=2, help='Bu uzunluktan kısa track\'leri atla')
    parser.add_argument('--scene_name', default='custom', help='Sahne adı (log için)')
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print('Hata: CSV dosyası bulunamadı:', args.csv)
        sys.exit(1)

    # CSV oku
    data = pd.read_csv(args.csv, sep=args.sep)
    for col in [args.frame_col, args.track_col, args.x_col, args.y_col]:
        if col not in data.columns:
            print('Hata: CSV\'de "%s" kolonu yok. Mevcut kolonlar:' % col, list(data.columns))
            sys.exit(1)

    data = data.rename(columns={
        args.frame_col: 'frame_id',
        args.track_col: 'track_id',
        args.x_col: 'pos_x',
        args.y_col: 'pos_y'
    })
    data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
    data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')
    data['node_id'] = data['track_id'].astype(str)
    data.sort_values('frame_id', inplace=True)

    if args.center:
        data['pos_x'] = data['pos_x'] - data['pos_x'].mean()
        data['pos_y'] = data['pos_y'] - data['pos_y'].mean()

    if args.scale != 1.0:
        data['pos_x'] = data['pos_x'] * args.scale
        data['pos_y'] = data['pos_y'] * args.scale

    # Frame id'leri 0'dan başlat
    data['frame_id'] = data['frame_id'] - data['frame_id'].min()
    max_timesteps = data['frame_id'].max()

    env = Environment(node_type_list=['PEDESTRIAN'], standardization=STANDARDIZATION)
    env.attention_radius = {(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): 3.0}

    scene = Scene(timesteps=int(max_timesteps) + 1, dt=args.dt, name=args.scene_name)

    for node_id in data['node_id'].unique():
        node_df = data[data['node_id'] == node_id].copy()
        # Eksik frame'leri doldur
        node_df = fill_gaps(node_df, 'frame_id', 'pos_x', 'pos_y')
        node_df = node_df.sort_values('frame_id')

        if len(node_df) < args.min_track_length:
            continue

        node_values = node_df[['pos_x', 'pos_y']].values
        new_first_idx = int(node_df['frame_id'].iloc[0])

        x = node_values[:, 0]
        y = node_values[:, 1]
        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        data_dict = {
            ('position', 'x'): x, ('position', 'y'): y,
            ('velocity', 'x'): vx, ('velocity', 'y'): vy,
            ('acceleration', 'x'): ax, ('acceleration', 'y'): ay
        }
        node_data = pd.DataFrame(data_dict, columns=DATA_COLUMNS)
        node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
        node.first_timestep = new_first_idx
        scene.nodes.append(node)

    env.scenes = [scene]
    maybe_makedirs(os.path.dirname(os.path.abspath(args.output)))
    with open(args.output, 'wb') as f:
        dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)

    print('Yazıldı: %s (sahne: %s, %d node, %d timestep)' % (args.output, args.scene_name, len(scene.nodes), scene.timesteps))


if __name__ == '__main__':
    main()
