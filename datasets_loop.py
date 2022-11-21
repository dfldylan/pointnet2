import os
import pandas as pd
import numpy as np
from hash import hash
import shutil

root_path = r'/root/datasets/normal'
# log_path = os.path.join(root_path, 'log.csv')
# pred_fps = [350, 450]
output_folder_root = r'/root/datasets/pred/pointnet2'


class PredDataLoader(object):
    def __init__(self):
        # self.df = pd.read_csv(log_path)
        self.index = None
        self.start_csv = r'/root/datasets/large/all_particles_390.csv'
        self.pred_loop_num = 30

    def __len__(self):
        return 1

    def __getitem__(self, item):
        start_csv = self.start_csv
        self.fps = int(start_csv.split(r'/')[-1].split(r'.')[0].split(r'_')[-1])
        id = hash(start_csv)
        print(id, start_csv)
        self.pred_folder = r'./pred/' + id
        self.output_folder = os.path.join(output_folder_root, id)
        if not os.path.exists(self.pred_folder):
            os.makedirs(self.pred_folder)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # # self.index = item
        # folder_num = item // len(pred_fps)
        # fps_num = item % len(pred_fps)
        # folder = str(self.df.iloc[folder_num, 0])
        # fps = str(pred_fps[fps_num])
        # path = os.path.join(root_path, folder)
        # path = os.path.join(path, r'all_particles_' + fps + '.csv')
        # self.current_folder_index = folder_num
        # self.current_fps = fps
        data = get_data_from_file(start_csv)
        point_set = data[:, :3]
        normal = data[:, 3:7]
        seg = data[:, 7:10]
        return point_set, normal, seg

    #
    # def save_time(self, time):
    #     self.df.loc[self.current_folder_index, self.current_fps] = time
    #     self.df.to_csv(r'log.csv', index=False)

    def write_csv(self, batch_data):
        # concat csv data with solid particles
        df = pd.read_csv(self.start_csv)
        # df = df[df.isFluidSolid == 1]
        # data_fluid = np.concatenate((pos, vel), axis=1)  # [-1, 6]
        df0 = pd.DataFrame(
            {df.columns[0]: batch_data[:, 0], df.columns[1]: batch_data[:, 1], df.columns[2]: batch_data[:, 2],
             df.columns[3]: batch_data[:, 3], df.columns[4]: batch_data[:, 4], df.columns[5]: batch_data[:, 5],
             df.columns[7]: batch_data[:, 6]}, columns=df.columns[:18], index=range(batch_data.shape[0]))
        # df0[df.columns[6:18]] = 0
        df = df0

        print(str(self.fps) + ' ok!')
        # write csv -- fast mode start
        df.to_csv(os.path.join(self.pred_folder, 'all_particles_' + str(self.fps) + '.csv'), index=False)
        if self.output_folder is not None:
            shutil.copy(os.path.join(self.pred_folder, 'all_particles_' + str(self.fps) + '.csv'), self.output_folder)


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
