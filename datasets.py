import os
import numpy as np
import pandas as pd

class NormalDataset:
    def __init__(self, root):
        self.root = root
        self.datapath = find_files(self.root, 401, 400,scene_num=[0])

    def __getitem__(self, item):
        data = get_data_from_file(self.datapath[item])
        point_set = data[:, :3]
        normal = data[:, 3:7]
        seg = data[:, 7:10]
        return point_set, normal, seg

    def __len__(self):
        return len(self.datapath)

def find_files(root_path, range_up=None, range_down=None, scene_num=None):
    folders = []
    exist_folders = [item for item in os.listdir(root_path) if len(item.split(r'.')) < 2]
    if scene_num is not None:
        for each in scene_num:
            each = str(each)
            if each in exist_folders:
                folders.append(each)
            else:
                print("folder "+each+" doesn't exist!")
    else:
        folders = exist_folders

    files_path = []
    for item in folders:
        path = os.path.join(root_path, item)
        files = [file for file in os.listdir(path) if file.split(r'.')[-1] == 'csv']
        if range_up is None and range_down is None:
            files_path+=(os.path.join(path, file) for file in files)
        else:
            for file in files:
                fps = int(file.split(r'.')[0].split(r'_')[-1])
                if range_up is not None and fps >= range_up:
                    continue
                if range_down is not None and fps < range_down:
                    continue
                files_path.append(os.path.join(path, file))
    return files_path

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
