"""
读取LAS格式文件
建立kd_tree
保存kdtree及ply文件
"""
from sklearn.neighbors import KDTree  #KD-Tree构建与空间查询
from os.path import join, exists, dirname, abspath
import numpy as np
import os, glob, pickle
import sys
from help_ply import write_ply  # write_ply函数用于将点云数据写入PLY文件
# from help_tool import DataProcessing as DP
import pandas as pd
from step0_read_las import read  # read函数用于读取点云数据
from help_tool import DataProcessing as DP  #数据预处理工具

#新补充下面这行代码
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BASE_DIR = dirname(abspath(__file__))  # 获取当前文件的绝对路径
ROOT_DIR = dirname(BASE_DIR)  #获取根目录路径
sys.path.append(BASE_DIR)  # 将当前文件夹路径添加到系统路径中
sys.path.append(ROOT_DIR)  # 将根目录路径添加到系统路径中   添加路径从而确保可以正常导入本地模块（如 help_tool、step0_read_las）。

def load_pc_h3d(filename): #加载H3D格式的点云数据
    pc_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.float32)  #使用pandas读取点云数据，读入为numpy array，delim_whitespace=True表示以空格为分隔符
    #读取.txt点云文件，每行格式为：x y z R G B instance_id semantic_label
    pc = pc_pd.values
    return pc

def main(dataset_path, output_root=None):  #主函数，dataset_path为.txt文件的输入目录

    #构建输出目录
    #original_pc_folder = join(dirname(dataset_path),'points-1')  # points-1 用于保存 .ply 格式点云文件。
    #kdtree_folder = join(dirname(dataset_path),'kdtree-1')  # kdtree-1 用于保存 KDTree 和投影索引文件。
    #为了实现将输出结果存放在指定目录，现在将这两行代码替换为接下来这四行

    if output_root is None:
        output_root = dirname(dataset_path)
    original_pc_folder = join(output_root, 'points-1')
    kdtree_folder = join(output_root, 'kdtree-1')


    os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None  # 如果 original_pc_folder 不存在，则创建它
    os.mkdir(kdtree_folder) if not exists(kdtree_folder) else None  # 如果 kdtree_folder 不存在，则创建它

    grid_size = 0.04  #下采样步长设置为0.04米
    for pc_path in glob.glob(join(dataset_path, '*.txt')):  #遍历dataset_path目录下的所有.txt文件
        print(pc_path)

        file_name = pc_path.split('/')[-1][:-4]  # 点云文件名，例如：'marketplacefeldkirch_station4_intensity_rgb'   #文件名提取，并去掉后缀.txt
        full_ply_path = join(original_pc_folder, file_name + '.ply')  # ply文件，存放于original_ply文件中  #更改原.txt文件为.ply文件
        #例如 marketplacefeldkirch_station4.txt → marketplacefeldkirch_station4.ply
        # check if it has already calculated
        #若 ply 文件已存在，则跳过该文件的处理
        if exists(join(kdtree_folder, file_name + '_KDTree.pkl')):  # kdtree文件
            continue

        # points_xyz, points_labels, points_min, points_color = read(pc_path)
        pc = load_pc_h3d(pc_path)  #读取点云数据，返回numpy array格式的点云数据
        points_xyz = pc[:, :3].astype(np.float32)  #分离点云的坐标信息，前三列为x, y, z坐标，并转换为float32类型
        points_color = pc[:, 3:6].astype(np.float32)  #分离点云的颜色信息，后三列为R, G, B颜色值，并转换为float32类型

        #标签处理
        # H3D数据集的标签处理：根据文件名判断是训练集还是测试集
        # 如果是测试集，则使用倒数第二列作为标签，否则使用倒数第三列作为标签
        if file_name == 'test':  #如果文件名为'test'，则表示是测试集
            # points_f1 = pc[:, -1].astype(np.int8)
            points_f2 = pc[:, -2].astype(np.int8)  #点云数据中倒数第二列作为标签，并转换为int8类型
            # f1, _ = np.histogram(points_f1, range=5)
            f2, _ = np.histogram(points_f2, range(5))  #对标签进行直方图统计，范围为0-5

            points_labels = np.ones([points_xyz.shape[0]]).astype(np.int8)  #测试集的标签全部设置为1，表示未知类别
        else:  #如果文件名不是'test'，则表示是训练集
            # points_f1 = pc[:, -2].astype(np.int8)
            points_f2 = pc[:, -3].astype(np.int8)  #点云数据中倒数第三列作为标签，并转换为int8类型
            # f1, _ = np.histogram(points_f1, range=5)
            f2, _ = np.histogram(points_f2, range(10))  #对标签进行直方图统计，范围为0-10
            # points_ins = pc[:, -2].astype(np.int8)
            points_labels = pc[:, -1].astype(np.int8)  #点云数据中最后一列作为标签，并转换为int8类型

        points_ins = points_f2.reshape(-1, 1)  #将标签转换为列向量，方便后续处理
        # points_color = points_color / 255.0
        points_feature = np.hstack((points_color, points_ins))  #将颜色信息和标签信息合并为特征向量，features = [R, G, B, instance_id]
        sub_xyz, sub_feature, sub_labels = DP.grid_sub_sampling(points_xyz, points_feature, points_labels, grid_size)  
        # 将点云划分为网格，网格大小为grid_size，每格保留一个代表点（如质心），同时下采样特征与标签
        # 返回下采样后的点云坐标、特征向量和标签

        write_ply(full_ply_path, (sub_xyz, sub_feature, sub_labels),
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'ins', 'label'])
        # 对下采样后的点建立kd-tree并保存
        #写入标准格式的.ply文件

        #构建KDTree
        # sub_xyz为下采样后的点云坐标，sub_feature为下采样后的特征向量，sub_labels为下采样后的标签
        search_tree = KDTree(sub_xyz, leaf_size=50)  #对下采样后的点云坐标建立KDTree，leaf_size设置为50，lead_size是KDTree的叶子节点大小，影响查询速度和内存使用
        # KDTree是一个高效的空间索引结构，用于快速查询点云
        kd_tree_file = join(kdtree_folder, file_name + '_KDTree.pkl')  # kdtree文件保存路径 #例如 marketplacefeldkirch_station4_KDTree.pkl
        with open(kd_tree_file, 'wb') as f:  #以二进制写入模式打开文件
            #将KDTree对象序列化并保存到文件中
            pickle.dump(search_tree, f)  

        #kd_tree_file文件中保存了KDTree对象，可以用于快速查询点云数据    

        #保存原始点与下采样点的投影索引
        # proj_idx用于记录原始点在下采样点中的最近点索引
        # points_xyz为原始点云坐标，sub_xyz为下采样后的点云坐标
        # points_labels为原始点云标签，sub_labels为下采样后的点云标签
        proj_idx = np.squeeze(search_tree.query(points_xyz, return_distance=False))  # sub_points（采样前）中的点在sub_xyz（采样后）中的最近点的索引
        proj_idx = proj_idx.astype(np.int32)
        proj_save = join(kdtree_folder, file_name + '_proj.pkl')  #保存投影索引的文件路径 #例如 marketplacefeldkirch_station4_proj.pkl
        with open(proj_save, 'wb') as f:  #以二进制写入模式打开文件
            #将投影索引和标签保存到文件中
            pickle.dump([proj_idx, points_labels], f)

        #proj_save文件中保存了原始点云到下采样点云的投影索引和标签信息



if __name__ == '__main__':
    input_path = '/home/hy/projects/PointCloud_Code/Results/data_prepare_result/step0_result'
    output_path = '/home/hy/projects/PointCloud_Code/Results/data_prepare_result/step1_H3D_result'
    main(input_path, output_path)

#该数据集通常来源于三维激光雷达（LiDAR）或摄影测量设备，传入数据集的格式是.txt文件，每个.txt是一个点云帧，包含了点的空间坐标、颜色和标签。
# 该脚本的主要功能是读取H3D格式的点云数据，进行下采样处理，并保存为PLY格式，同时建立KDTree索引和投影索引。
# 该脚本的输入是H3D格式的点云数据，输出是PLY格式的点云文件、KDTree索引文件和投影索引文件。
# 该脚本的主要步骤包括：
# 1. 读取H3D格式的点云数据
# 2. 对点云数据进行下采样处理
# 3. 保存下采样后的点云数据为PLY格式
# 4. 建立KDTree索引
# 5. 保存KDTree索引和投影索引
# 6. 输出处理结果

