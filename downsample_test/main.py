"""
主要功能是对点云进行**两步法下采样（Two-step FPS, Farthest Point Sampling）**处理。

这段代码实现了以下两部分功能：
    提取原始点云中邻近采样中心的子区域（可选，get_ori_data 函数）
    使用 TensorFlow 实现 Farthest Point Sampling（FPS,最远点采样）对点云进行均匀下采样（fps 函数）

最终输出一个 .ply 文件，包含了下采样后的：
    空间坐标 (x, y, z)
    颜色信息 (r, g, b)
    分类标签 label

输入
    .ply 格式的点云文件，包含字段：x, y, z, red, green, blue, label
    示例路径：/home/yunl/Data/paper_data/0611_sample/ts_512_.ply-----例子
    可能包含 512 个点

输出：
    下采样后的 .ply 文件
    路径：/home/yunl/Data/paper_data/0611_sample/ts_256_.ply-----例子
    含 256 个点
    字段保持一致：x, y, z, red, green, blue, label



"""
import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(BASE_DIR)  # model
# sys.path.append(ROOT_DIR)  # provider
sys.path.append(os.path.join(BASE_DIR, 'sampling'))  #---------------------------------------------------------------------------------------导入sampling文件夹下的模块
#设置当前文件路径为基准路径，并将 sampling 子目录加入 Python 模块搜索路径，方便导入自定义模块（如 tf_sampling.py）。

import numpy as np
from help_ply import read_ply, write_ply  # 读取/写入.ply文件 （自定义函数） ------------------------------------------------------------------导入las文件夹下的两个自定义函数
from sklearn.neighbors import KDTree  #KDTree：快速查找最近邻点
from tf_sampling import gather_point, farthest_point_sample  #farthest_point_sample：Farthest Point Sampling 算法（TF实现）
import tensorflow as tf
def get_ori_data(filename):  #提取原始点云邻域
    ori_data = read_ply(filename)
    xyz = np.vstack((ori_data['x'], ori_data['y'], ori_data['z'])).T # （n,3）
    rgb = np.vstack((ori_data['red'], ori_data['green'], ori_data['blue'])).T
    label = ori_data['class']  # （n,）
    pick_point = np.array([57.93, 412.72, 27.77]).reshape(1,3)
    #从 .ply 文件读取坐标、颜色和标签
    #指定一个采样中心点（pick_point）

    search_tree = KDTree(xyz, leaf_size=50)
    idx = np.squeeze(search_tree.query(pick_point, k=65536, return_distance=False))  # sub_points（采样前）中的点在sub_xyz（采样后）中的最近点的索引
    #利用 KD 树，寻找距离采样点最近的 65536 个点（构成局部区域）

    sub_xyz = xyz[idx]
    sub_rgb = rgb[idx]
    sub_lable = label[idx]
    #提取子区域的 xyz、rgb、label
    print("success!")

    write_ply('/home/yunl/Data/paper_data/fps/ori.ply', (sub_xyz, sub_rgb, sub_lable),
              ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    #写入新的 .ply 文件（提取的原始子区域）

def fps(filename, num_points,out_path):   # 使用 FPS 执行均匀下采样
    ori_data = read_ply(filename)
    xyz = np.vstack((ori_data['x'], ori_data['y'], ori_data['z'])).T  # （n,3）
    rgb = np.vstack((ori_data['red'], ori_data['green'], ori_data['blue'])).T
    label = ori_data['label']  # （n,）
    #读取下采样前的点云：坐标、颜色、标签

    xyz_ = np.expand_dims(xyz, 0).astype(np.float32)
    sub_idx = farthest_point_sample(num_points, xyz_)
    # 使用 farthest_point_sample 选取最远点索引
    # num_points 是目标点数（如 256）

    with tf.Session() as sess:
        idx = sess.run(sub_idx)  # 在 TensorFlow 会话中运行采样索引

    sub_xyz = np.squeeze(xyz[idx], axis=0)
    sub_rgb = np.squeeze(rgb[idx], axis=0)
    sub_label = np.squeeze(label[idx], axis=0)
    # 提取采样后的子点云数据

    write_ply(out_path, (sub_xyz, sub_rgb, sub_label),
              ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])  #写入新的 .ply 文件：256 个均匀采样点
    # sub_xyz = gather_point(xyz, sub_idx)

if __name__ == '__main__':
    # get_ori_data('/home/yunl/Data/paper_data/fps/fps_0.ply')
    input_file = '/home/yunl/Data/paper_data/0611_sample/ts_512_.ply'
    output_file = '/your/custom/path/your_filename.ply'  # 你想要的输出路径
    fps(input_file, 256,output_file)
    #将一个含 512 点的 .ply 文件，下采样为 256 个点





