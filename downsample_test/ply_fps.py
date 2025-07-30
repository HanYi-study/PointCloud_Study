import open3d as o3d  #用于处理 .ply 点云文件。
import os
import numpy as np

'''
ply_fps.py 是一个用于批量处理点云 .ply 文件的脚本，核心功能是对点云进行 最远点下采样（Farthest Point Sampling, FPS），从而将点的数量减少一半并保存为新的 .ply 文件。

输入输出文件格式都是 .ply（Polygon File Format 或 Stanford Triangle Format），它是一种用于存储三维点云数据的常见文件格式。
输入的.ply点云格式：（以 ASCII 格式为例）
ply
format ascii 1.0
element vertex 1000
property float x
property float y
property float z
end_header
0.123 0.456 0.789
1.123 0.856 0.389
...

有 1000 个点，每行一个点的 x y z 坐标。
如果原始文件中还有颜色信息（如 red green blue），也可以被 open3d 读取。

'''


print(o3d.__version__)
def farthest_point_sampling(points, k):
    """
    最远点下采样
    points: 是一个形状为 (N, 3) 的 numpy 数组，表示原始点云，包含 N 个点，每个点由三维坐标（x, y, z）表示。
    k: 你想采样出的点的数量，通常 k < N。
    从输入点云中采样 k 个点。

    从点云中选择代表性较强、分布均匀的子集。
    """
    farthest_pts = np.zeros((k, 3))  # 初始化一个空数组 farthest_pts，形状为 (k, 3)，用于保存最终采样出的点。
    #每一行是一个点的三维坐标。初始值为 0。
    farthest_pts[0] = points[np.random.randint(len(points))]  # 随机从 points 中选一个点，作为采样的第一个点。把这个点存入 farthest_pts[0]。
    #np.random.randint(len(points)) 生成一个随机索引。
    distances = np.linalg.norm(points - farthest_pts[0], axis=1)  # 计算所有点到第一个采样点的 欧几里得距离（L2 距离）。
    #points - farthest_pts[0] 结果是一个 (N, 3) 的差值矩阵。
    #np.linalg.norm(..., axis=1) 是沿着每一行求 L2 范数，即每个点到当前采样点的距离。
    #distances 是一个长度为 N 的一维数组，表示每个点到采样点的最近距离。

#逐个选择最远的点
    for i in range(1, k):
        farthest_pts[i] = points[np.argmax(distances)]  # np.argmax(distances) 找出当前距离最大的点的索引。
        #也就是距离前面所有采样点中最近点仍然很远的那个点。
        #这个点被选为下一个采样点，存入 farthest_pts[i]。
        distances = np.minimum(distances, np.linalg.norm(points - farthest_pts[i], axis=1))  # 更新所有点到“当前所有采样点”的最小距离。
        #新的采样点是 farthest_pts[i]，所以：np.linalg.norm(points - farthest_pts[i], axis=1) 是每个点到这个新采样点的距离。
        #np.minimum(...) 会对比旧的距离和新的距离，保留“更小”的那个。
        #最终 distances 始终保存着每个点到“当前所有已采样点”中最近一个的距离。

    return farthest_pts  # 该函数返回一个 k × 3 的数组，包含了采样出来的 k 个代表性点。

    '''
    fps举例理解：

    假设输入点云：
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0.5, 0.5, 0]
    ])
    k = 2

    第一步：
    随机选择一个点，比如选择了 [0, 0, 0]。然后计算所有点到 [0, 0, 0] 的距离：
    [0, 0, 0] -> 距离 0
    [1, 0, 0] -> 距离 1
    [0, 1, 0] -> 距离 1
    [1, 1, 0] -> 距离 √2 ≈ 1.41
    [0.5, 0.5, 0] -> 距离 √0.5 ≈ 0.707
    距离最大的是 [1, 1, 0]，所以它是下一个采样点。

    第二步：
    更新每个点到最近一个采样点的距离（比较 [0, 0, 0] 和 [1, 1, 0] 距离，取更小的）。
    如此循环直到采出 k 个点。

    '''


source_folder = "/home/zxb/Poppy/Data/fenlei3_ply"  # 原始 .ply 点云文件所在目录。
target_folder = "/home/zxb/Poppy/Data/fenlei3_ply_fps"  # 保存下采样后 .ply 的目录。

# 如果输出目录不存在，则创建它。
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 获取所有以 .ply 结尾的文件名，用于批处理。
ply_files = [f for f in os.listdir(source_folder) if f.endswith('.ply')]

for ply_file in ply_files:  #遍历所有点云文件，逐一处理。
    # 加载 .ply 文件，并提取点坐标数据，转为 numpy 数组。
    point_cloud = o3d.io.read_point_cloud(os.path.join(source_folder, ply_file))
    points = np.asarray(point_cloud.points)

    # 调用前面定义的 farthest_point_sampling 函数，对点云进行 50% 下采样。
    sampled_points = farthest_point_sampling(points, int(0.5 * len(points)))

    # 创建新的点云对象，将采样后的点写入其中。
    new_point_cloud = o3d.geometry.PointCloud()
    new_point_cloud.points = o3d.utility.Vector3dVector(sampled_points)

    #将新的 .ply 文件写入目标文件夹中，文件名与原始一致。
    o3d.io.write_point_cloud(os.path.join(target_folder, ply_file), new_point_cloud)

