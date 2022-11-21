import os
import pandas as pd
import numpy as np

root_path = r'/root/datasets/normal'
log_path = os.path.join(root_path, 'log.csv')
pred_fps = [350, 450]


class PredDataLoader(object):
    def __init__(self):
        self.df = pd.read_csv(log_path)
        self.index = None

    def __len__(self):
        return len(pred_fps) * len(self.df)

    def __getitem__(self, item):
        # self.index = item
        folder_num = item // len(pred_fps)
        fps_num = item % len(pred_fps)
        folder = str(self.df.iloc[folder_num, 0])
        fps = str(pred_fps[fps_num])
        path = os.path.join(root_path, folder)
        path = os.path.join(path, r'all_particles_' + fps + '.csv')
        self.current_folder_index = folder_num
        self.current_fps = fps
        data = get_data_from_file(path)
        point_set = data[:, :3]
        normal = data[:, 3:7]
        seg = data[:, 7:10]
        return point_set, normal, seg

    def save_time(self, time):
        self.df.loc[self.current_folder_index, self.current_fps] = time
        self.df.to_csv(r'log.csv', index=False)

def build_data(pos, vel, s_f, out, voxel_size):
    if voxel_size:
        voxel = np.floor(pos * (1 / voxel_size)).astype(int)
    else:
        voxel = np.zeros_like(pos)
    data = np.concatenate([pos, vel, s_f, out, voxel], axis=1)
    return data


def get_data_from_file(file_path, voxel_size=None):
    df = pd.read_csv(file_path, dtype=float)
    df['isFluidSolid'] = df['isFluidSolid'].astype(int)
    return build_data(df.iloc[:, :3].values, df.iloc[:, 3:6].values, df.iloc[:, 7:8].values,
                      df.iloc[:, 12:15].values - df.iloc[:, 3:6].values, voxel_size)
