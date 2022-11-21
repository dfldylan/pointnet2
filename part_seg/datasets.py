import os
import numpy as np
import pandas as pd

class NormalDataset:
    def __init__(self, root):
        self.root = root
        self.datapath = find_files(self.root, 500, 300)

    def __getitem__(self, item):
        data = get_data_from_file(self.datapath[item])
        point_set = data[:, :3]
        normal = data[:, 3:7]
        seg = data[:, 3:7]
        return point_set, normal, seg

    def __len__(self):
        return len(self.datapath)

class RNNDataset:
    def __init__(self, root, seq=5):
        self.root = root
        # self.datapath = find_files(self.root, 500, 300)
        self.seq_length = seq
        self.folder_path, self.down,  self.up = self.find_files(root, seq, range_down=300, range_up=500)


    def __getitem__(self, item):
        folder = np.random.choice(self.folder_path)
        start_fps = np.random.randint(self.down, self.up)
        print(folder, str(start_fps))
        log_data_list = []
        for i in range(self.seq_length):
            fps = start_fps+ i
            path = os.path.join(folder, "all_particles_"+str(fps)+".csv")
            log_data = pd.read_csv(path, dtype=float).iloc[:, :7].values
            print(str(log_data.shape[0]))

            point_set = log_data[:, :3]
            normal = log_data[:, 3:7]
            seg = log_data[:, 3:7]
            log_data_list.append((point_set, normal, seg))

        return log_data_list

    def __len__(self):
        return len(self.folder_path) * (self.up-self.down)

    def find_files(self, root_path, seq_length, range_up=None, range_down=None, scene_num=None):
        folders = []
        exist_folders = [item for item in os.listdir(root_path) if len(item.split(r'.')) < 2]
        if scene_num is not None:
            for each in scene_num:
                each = str(each)
                if each in exist_folders:
                    folders.append(each)
                else:
                    print("folder " + each + " doesn't exist!")
        else:
            folders = exist_folders

        folder_path = [os.path.join(root_path, item) for item in folders]
        down = 1 if range_down is None else range_down
        up = 1500 if range_up is None else range_up
        up -= seq_length
        if up <= down:
            print('failed')
            exit(-1)
        return folder_path, down, up

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
