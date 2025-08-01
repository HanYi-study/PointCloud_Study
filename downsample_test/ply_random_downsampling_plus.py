import open3d as o3d  # 用于读取、处理、保存点云数据
import numpy as np    # 用于随机采样、数组处理
import os             # 用于文件路径操作
import argparse       # 用于命令行参数解析
import sys            # 用于退出程序
from sklearn.neighbors import KDTree
import tensorflow as tf
from tf_sampling import farthest_point_sample
from utils.ply import read_ply  # 你自己的 ply 读取工具

#版本2.0

def get_sampling_choice(): #------------------------------------------------------------------------------- 🔧 获取用户选择的采样方式（比例采样或固定数量采样）
    print("请选择采样方式：")
    print("1 - 固定比例采样（如保留 50% 点）")
    print("2 - 固定数量采样（如随机采样 16384 个点）")
    print("3 - 使用 PCL 的 RandomSample 随机采样（固定点数）")
    print("4 - 多轮自定义采样")
    print("5 - 中心点邻域提取 + FPS采样（先选定邻域，再做FPS）")
    choice = input("> ").strip()

    if choice == "1":
        ratio = float(input("请输入采样比例（0~1之间）：").strip())
        return "ratio", ratio
    elif choice == "2":
        count = int(input("请输入采样点数量（如 16384）：").strip())
        return "count", count
    elif choice == "3":
        count = int(input("请输入采样点数量（如 16384）：").strip())
        return "pcl", count
    elif choice == "4":
        return "custom_multi", get_custom_multi_sampling_config()
    elif choice == "5":
        center = input("请输入中心点坐标（x,y,z，例如 57.93,412.72,27.77）：\n> ").strip()
        center_xyz = np.array([float(c) for c in center.split(',')]).reshape(1, 3)
        k1 = int(input("邻域保留多少点（如65536）：\n> ").strip())
        k2 = int(input("FPS保留多少点（如256）：\n> ").strip())
        return "center_fps", (center_xyz, k1, k2)
    else:
        print("输入无效，默认使用 0.5 比例采样。")
        return "ratio", 0.5


def get_custom_multi_sampling_config():  #----------------------------------------------------------------🎛️ 获取用户配置的双阶段采样流程，包括采样方法、轮数与每轮比例参数。
    print("参数解释：")
    print("ethod（采样方法）：选择的采样算法类型，支持 random（随机）或 fps（最远点采样），如第一种为随机，第二种为 FPS")
    print("rounds（轮数）：执行采样的轮次数量，表示你要重复几次该种采样方式")
    print("ratio（比例）：每一轮采样中保留点的比例（0 ~ 1 之间的小数），例如每轮保留 25% 点，即 ratio=0.25")
    print("-----------------------------------------------------------------------------------------------------")
    print("请选择第一种采样方法：")
    print("1 - 随机采样")
    print("2 - FPS采样")
    first_type = input("> ").strip()
    first_rounds = int(input("请输入采样轮数：\n> ").strip())
    first_ratio = float(input("请输入每轮采样比例（0~1 之间）：\n> ").strip())

    print("\n请选择第二种采样方法：")
    print("1 - 随机采样")
    print("2 - FPS采样")
    second_type = input("> ").strip()
    second_rounds = int(input("请输入采样轮数：\n> ").strip())
    second_ratio = float(input("请输入每轮采样比例（0~1 之间）：\n> ").strip())

    return {
        "first_type": "random" if first_type == "1" else "fps",
        "first_rounds": first_rounds,
        "first_ratio": first_ratio,
        "second_type": "random" if second_type == "1" else "fps",
        "second_rounds": second_rounds,
        "second_ratio": second_ratio
    }


def sample_indices(total_points, method, value):  #----------------------------------------------------------------📍 根据用户选择的采样方式，返回采样点的索引列表
    if method == "ratio":
        k = int(total_points * value)
    elif method == "count":
        k = min(value, total_points)
    else:
        raise ValueError("不支持的采样方式")
    return np.random.choice(total_points, k, replace=False)

def pcl_random_sample(input_path, sample_num, output_path):    #-------------------------🧪 使用 PCL 的 RandomSample 滤波器对 .pcd 文件进行固定数量的随机采样，并输出采样结果
    """
    使用 PCL 的 RandomSample 方法对点云进行随机采样。

    参数：
    - input_path: 输入 .ply 文件路径
    - sample_num: 采样点数
    - output_path: 输出 .ply 文件路径

    输出格式说明：
    - 输出为 .ply 文件（不带标签）
    - 若需要标签或 .txt 输出，可在主流程中再处理

    """


    """
    📌 PCL 采样方法：RandomSample（随机抽样滤波器）

    方法名称： pcl.filters.RandomSample

    作用：从点云中随机抽取固定数量的点，适用于预处理、数据压缩、快速可视化等。

    特点：
    ✅ 快速高效，适合大规模点云处理
    ✅ 保留全局分布的点，但不能保证空间均匀性
    ✅ 采样结果受随机种子控制，可复现
    ❌ 不自动保留颜色、标签等属性（需要单独处理）

    使用前后效果：
    原始点云保留整体轮廓，但细节随机丢失。
    如原始有 65000 个点，设定采样数为 16384，则输出将为 16384 个稀疏但代表性的点。

    与比例采样/固定采样区别：
    - 比例采样：保留百分比，如 50%
    - 固定采样（RandomSample）：保留精确数量，如 16384 个

    """

    import pclpy
    from pclpy import pcl

    # Step 1: 转换 .ply -> .pcd（因为 PCL 处理 PCD 更方便）
    temp_pcd = os.path.splitext(output_path)[0] + "_temp.pcd"
    pcd = o3d.io.read_point_cloud(input_path)
    o3d.io.write_point_cloud(temp_pcd, pcd)

    # Step 2: PCL 采样处理
    cloud_in = pcl.PointCloud.PointXYZ()
    cloud_out = pcl.PointCloud.PointXYZ()
    reader = pcl.io.PCDReader()
    reader.read(temp_pcd, cloud_in)

    rs = pcl.filters.RandomSample.PointXYZ()
    rs.setInputCloud(cloud_in)
    rs.setSample(sample_num)
    rs.setSeed(42)
    rs.filter(cloud_out)

    # Step 3: 保存结果
    writer = pcl.io.PCDWriter()
    temp_pcd_out = os.path.splitext(output_path)[0] + "_sampled.pcd"
    writer.write(temp_pcd_out, cloud_out)

    # Step 4: 转换为 .ply 输出
    sampled = o3d.io.read_point_cloud(temp_pcd_out)
    o3d.io.write_point_cloud(output_path, sampled)
    print(f"✔ PCL 采样完成，保存至：{output_path}")

def multi_stage_sampling(points, method, rounds, ratio):  #----------------------------------------------------------🔁 对输入点集进行多轮采样，每轮按指定方法（随机 / FPS）和比例递减保留点。
    sampled = points
    for i in range(rounds):
        num = int(len(sampled) * ratio)
        if method == "random":
            idx = np.random.choice(len(sampled), num, replace=False)
        elif method == "fps":
            sampled_ = np.expand_dims(sampled[:, :3], axis=0).astype(np.float32)
            idx_tf = farthest_point_sample(num, sampled_)
            with tf.Session() as sess:
                idx = sess.run(idx_tf)[0]
        sampled = sampled[idx]
    return sampled


def process_point_cloud_file(input_path, output_path, method, value, out_format):  #---------------------------------------🎯 处理单个 .ply 点云文件：读取、采样、输出结果（含 .ply 和可选 .txt）
    print(f"读取点云文件: {input_path}")

    if method == "pcl":
        pcl_random_sample(input_path, value, output_path)
        return

    pcd = o3d.io.read_point_cloud(input_path)

    has_color = pcd.has_colors()
    has_label = False
    labels = None

    ply = PlyData.read(input_path)
    if 'label' in ply.elements[0].data.dtype.names:
        has_label = True
        labels = np.array([p['label'] for p in ply.elements[0].data])
    else:
        labels = None

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if has_color else None

    if method == "custom_multi":
        cfg = value
        full_data = np.hstack((points, colors, labels.reshape(-1, 1) if labels is not None else np.zeros((len(points), 1))))
        sampled = multi_stage_sampling(full_data, cfg["first_type"], cfg["first_rounds"], cfg["first_ratio"])
        sampled = multi_stage_sampling(sampled, cfg["second_type"], cfg["second_rounds"], cfg["second_ratio"])
        sampled_points = sampled[:, 0:3]
        sampled_colors = sampled[:, 3:6] if colors is not None else None
        sampled_labels = sampled[:, 6] if labels is not None else None

    elif method == "center_fps":
        center_point, first_k, fps_k = value
        ply_data = read_ply(input_path)
        xyz = np.vstack((ply_data['x'], ply_data['y'], ply_data['z'])).T
        rgb = np.vstack((ply_data['red'], ply_data['green'], ply_data['blue'])).T
        label = ply_data['label'] if 'label' in ply_data.dtype.names else (ply_data['class'] if 'class' in ply_data.dtype.names else np.zeros((len(xyz),)))

        # KDTree 邻域提取
        tree = KDTree(xyz, leaf_size=50)
        idx = np.squeeze(tree.query(center_point.reshape(1, -1), k=first_k, return_distance=False))
        sub_xyz = xyz[idx]
        sub_rgb = rgb[idx]
        sub_label = label[idx]

        # Step 1: 保存邻域提取后的 .ply（加后缀 _center_65536）
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        ori_filename = f"{base_name}_center_{first_k}.ply"
        ori_path = os.path.join(os.path.dirname(output_path), ori_filename)
        write_ply(ori_path, (sub_xyz, sub_rgb, sub_label), ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        print(f"✔ 局部邻域点（{first_k}）保存至：{ori_path}")


        # Step 2: FPS采样
        fps_input = np.expand_dims(sub_xyz, 0).astype(np.float32)
        sub_idx = farthest_point_sample(fps_k, fps_input)
        with tf.Session() as sess:
            fps_idx = sess.run(sub_idx)[0]

        final_xyz = sub_xyz[fps_idx]
        final_rgb = sub_rgb[fps_idx]
        final_label = sub_label[fps_idx]

        # Step 3: 保存 FPS 后的 .ply（加后缀 _center_fps_256）
        fps_filename = f"{base_name}_center_fps_{fps_k}.ply"
        fps_path = os.path.join(os.path.dirname(output_path), fps_filename)
        write_ply(ts_path, (final_xyz, final_rgb, final_label), ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        print(f"✔ FPS采样结果（{fps_k}）保存至：{fps_path}")


        # 同时构建 sampled_pcd 用于主流程 .ply / .txt 输出控制
        sampled_points = final_xyz
        sampled_colors = final_rgb / 255.0
        sampled_labels = final_label
        has_color = True
        has_label = True
        '''
        功能简述：
          在原始 .ply 点云中，选取距离指定中心点最近的 65536 个点；
          对这 65536 个点再执行一次 FPS（最远点采样）；
          最终输出 .ply 文件，保留选定子集的 xyz、RGB、label 信息。
        输入：
          /home/yunl/Data/paper_data/fps/fps_0.ply（带 x/y/z、r/g/b、class 字段的 .ply 文件）
        输出：
          /home/yunl/Data/paper_data/fps/ori.ply：65536 个点的局部区域点云（step 1）
          /home/yunl/Data/paper_data/0611_sample/ts_256_.ply：对上述点云进行 256 点 FPS 后的采样结果
        格式：
          输入输出均为 .ply 文件，字段包括：x y z red green blue label

        '''

    else:
        indices = sample_indices(len(points), method, value)
        sampled_points = points[indices]
        sampled_colors = colors[indices] if has_color else None
        sampled_labels = labels[indices] if has_label else None

    # 构建输出点云
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    if sampled_colors is not None:
        sampled_pcd.colors = o3d.utility.Vector3dVector(sampled_colors)

    o3d.io.write_point_cloud(output_path, sampled_pcd)
    print(f"✔ 采样完成，保存至：{output_path}")

    # === 输出 .txt 控制逻辑 ===
    need_txt = (
        (out_format == '2' and has_label) or
        (out_format == '3')
    )

    if need_txt:
        out_txt = os.path.splitext(output_path)[0] + ".txt"
        xyz = sampled_points
        rgb = (sampled_colors * 255).astype(np.uint8) if sampled_colors is not None else np.zeros_like(xyz, dtype=np.uint8)
        label = sampled_labels.reshape(-1, 1) if sampled_labels is not None else np.zeros((len(xyz), 1), dtype=np.int32)
        full = np.hstack((xyz, rgb, label))
        np.savetxt(out_txt, full, fmt='%f %f %f %d %d %d %d')
        print(f"✔ 点数据同时保存至：{out_txt}")

def main():   #-------------------------------------------------------------------------------------------------------🚀 脚本主流程：处理输入参数、控制批处理或单文件采样、调用核心处理函数
    input_path = input("请输入要处理的文件路径或文件夹路径：").strip()
    is_dir = os.path.isdir(input_path)

    output_dir = input("请输入保存输出的文件夹路径：").strip()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 采样方式
    method, value = get_sampling_choice()

    # 输出格式
    print("\n请选择输出格式：")
    print("1 - 只输出 .ply")
    print("2 - 输出 .ply，如果有标签则同时输出 .txt")
    print("3 - 始终输出 .ply 和 .txt（即使没有标签）")
    out_format = input("请输入选项 1 / 2 / 3：").strip()
    if out_format not in ['1', '2', '3']:
        print("输入无效，默认使用 2：有标签时输出 .txt")
        out_format = '2'

    # 批处理或单文件
    if is_dir:
        ply_files = [f for f in os.listdir(input_path) if f.endswith(".ply")]
        for f in ply_files:
            src = os.path.join(input_path, f)
            dst = os.path.join(output_dir, f)
            process_point_cloud_file(src, dst, method, value, out_format)
    else:
        filename = os.path.basename(input_path)
        dst = os.path.join(output_dir, filename)
        process_point_cloud_file(input_path, dst, method, value, out_format)


if __name__ == "__main__":
    main()

'''    2.0版本执行解说：

$ python ply_random_sampling_plus.py

请输入要处理的文件路径或文件夹路径：
> /home/yunl/Data/paper_data/fps/

请输入保存输出的文件夹路径：
> /home/yunl/Data/paper_data/fps_sampled/

请选择采样方式：
1 - 固定比例采样（如保留 50% 点）
2 - 固定数量采样（如随机采样 16384 个点）
3 - 使用 PCL 的 RandomSample 随机采样（固定点数）
4 - 多轮自定义采样
> 2

请输入采样点数量（如 16384）：
> 16384

请选择输出格式：
1 - 只输出 .ply
2 - 输出 .ply，如果有标签则同时输出 .txt
3 - 始终输出 .ply 和 .txt（即使没有标签）
> 2

读取点云文件: /home/yunl/Data/paper_data/fps/chair.ply
✔ 采样完成，保存至：/home/yunl/Data/paper_data/fps_sampled/chair.ply
✔ 点数据同时保存至：/home/yunl/Data/paper_data/fps_sampled/chair.txt

读取点云文件: /home/yunl/Data/paper_data/fps/table.ply
✔ 采样完成，保存至：/home/yunl/Data/paper_data/fps_sampled/table.ply

'''
