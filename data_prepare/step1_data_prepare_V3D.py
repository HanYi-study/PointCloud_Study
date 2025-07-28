"""
读取txt格式文件
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

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

def load_pc_v3d(filename):
    pc_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.float32)
    pc = pc_pd.values
    return pc

def main(dataset_path):

    original_pc_folder = join(dirname(dataset_path),'original_test')
    kdtree_folder = join(dirname(dataset_path),'kdtree')
    os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None
    os.mkdir(kdtree_folder) if not exists(kdtree_folder) else None

    for pc_path in glob.glob(join(dataset_path, '*.txt')):
        print(pc_path)

        file_name = pc_path.split('/')[-1][:-4]  # 点云文件名，例如：'marketplacefeldkirch_station4_intensity_rgb'
        full_ply_path = join(original_pc_folder, file_name + '.ply')  # ply文件，存放于original_ply文件中
        # check if it has already calculated
        if exists(join(kdtree_folder, file_name + '_KDTree.pkl')):  # kdtree文件
            continue
        
        pc = load_pc_v3d(pc_path)
        xyz = pc[:, :3].astype(np.float32)
        label = np.array(pc[:, -1].astype(np.int8))
        his, _ = np.histogram(label, range(11))
        print(his)
        write_ply(full_ply_path, (xyz, label), ['x', 'y', 'z', 'class'])

        search_tree = KDTree(xyz, leaf_size=50)
        kd_tree_file = join(kdtree_folder, file_name + '_KDTree.pkl')
        with open(kd_tree_file, 'wb') as f:
            pickle.dump(search_tree, f)



if __name__ == '__main__':
    main('/home/yunl/study/CODE/paper_new/V3D/DATA')


