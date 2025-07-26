"""
读取LAS格式文件
建立kd_tree
保存kdtree及ply文件
"""
from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import os, glob, pickle
import sys
from help_ply import write_ply
# from help_tool import DataProcessing as DP
import pandas as pd
from data_prepare.step0_read_las import read
from help_tool import DataProcessing as DP

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

def load_pc_h3d(filename):
    pc_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.float32)
    pc = pc_pd.values
    return pc

def main(dataset_path):

    original_pc_folder = join(dirname(dataset_path),
                              'points-1')
    kdtree_folder = join(dirname(dataset_path),
                         'kdtree-1')
    os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None
    os.mkdir(kdtree_folder) if not exists(kdtree_folder) else None

    grid_size = 0.04
    for pc_path in glob.glob(join(dataset_path, '*.txt')):
        print(pc_path)

        file_name = pc_path.split('/')[-1][:-4]  # 点云文件名，例如：'marketplacefeldkirch_station4_intensity_rgb'
        full_ply_path = join(original_pc_folder, file_name + '.ply')  # ply文件，存放于original_ply文件中
        # check if it has already calculated
        if exists(join(kdtree_folder, file_name + '_KDTree.pkl')):  # kdtree文件
            continue

        # points_xyz, points_labels, points_min, points_color = read(pc_path)
        pc = load_pc_h3d(pc_path)
        points_xyz = pc[:, :3].astype(np.float32)
        points_color = pc[:, 3:6].astype(np.float32)
        if file_name == 'test':
            # points_f1 = pc[:, -1].astype(np.int8)
            points_f2 = pc[:, -2].astype(np.int8)
            # f1, _ = np.histogram(points_f1, range=5)
            f2, _ = np.histogram(points_f2, range(5))

            points_labels = np.ones([points_xyz.shape[0]]).astype(np.int8)
        else:
            # points_f1 = pc[:, -2].astype(np.int8)
            points_f2 = pc[:, -3].astype(np.int8)
            # f1, _ = np.histogram(points_f1, range=5)
            f2, _ = np.histogram(points_f2, range(10))
            # points_ins = pc[:, -2].astype(np.int8)
            points_labels = pc[:, -1].astype(np.int8)

        points_ins = points_f2.reshape(-1, 1)
        # points_color = points_color / 255.0
        points_feature = np.hstack((points_color, points_ins))
        sub_xyz, sub_feature, sub_labels = DP.grid_sub_sampling(points_xyz, points_feature, points_labels, grid_size)

        write_ply(full_ply_path, (sub_xyz, sub_feature, sub_labels),
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'ins', 'label'])
        # 对下采样后的点建立kd-tree并保存
        search_tree = KDTree(sub_xyz, leaf_size=50)
        kd_tree_file = join(kdtree_folder, file_name + '_KDTree.pkl')
        with open(kd_tree_file, 'wb') as f:
            pickle.dump(search_tree, f)

        proj_idx = np.squeeze(
            search_tree.query(points_xyz, return_distance=False))  # sub_points（采样前）中的点在sub_xyz（采样后）中的最近点的索引
        proj_idx = proj_idx.astype(np.int32)
        proj_save = join(kdtree_folder, file_name + '_proj.pkl')
        with open(proj_save, 'wb') as f:
            pickle.dump([proj_idx, points_labels], f)



if __name__ == '__main__':
    main('/home/yunl/study/CODE/paper_new/H3D/H3D')


