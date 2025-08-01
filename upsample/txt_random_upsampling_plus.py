
'''  -----------------------------------------------------------------------------------------------------------------------------------------版本1.0
import os
import random  #random 用于随机选择和扰动点。
def get_user_upsample_config():   #------------------------------------------------------------------------------------------------------
    print("欢迎使用点云上采样工具（增强版）")

    # ✅ 采样策略选择
    print("\n请选择上采样策略：")
    print("1 - 标签点复制 + 加扰动")
    print("2 - 标签点直接复制（无扰动）")
    print("3 - 基于边界框生成新点（随机坐标 + 平均RGB）")
    strategy = input("> ").strip()

    # 默认设为策略1
    if strategy not in ['1', '2', '3']:
        print("⚠ 无效选择，默认使用策略1（复制+扰动）")
        strategy = "1"

    # ✅ 统一输入目录
    input_folder = input("请输入输入文件夹路径（如 ./input）:\n> ").strip()
    output_folder = input("请输入输出文件夹路径（如 ./output，请与输入文件夹不同）:\n> ").strip()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ✅ 标签号（策略都需要）
    label_input = input("请输入你希望上采样的标签号（如 2）:\n> ").strip()
    label_target = float(label_input) if label_input else 2.0

    # ✅ 扰动幅度（仅策略1）
    if strategy == "1":
        user_scale = input("请输入扰动幅度（默认 0.01，直接回车使用默认）:\n> ").strip()
        perturbation_scale = float(user_scale) if user_scale else 0.01
    elif strategy == "2":
        perturbation_scale = 0.0
    else:
        perturbation_scale = None  # 无扰动项

    # ✅ 输出格式
    print("\n请选择输出格式：")
    print("1 - 输出完整点信息（x y z R G B label）")
    print("2 - 只输出坐标信息（x y z）")
    format_choice = input("> ").strip()
    output_xyz_only = (format_choice == "2")

    return {
        "strategy": strategy,
        "input_folder": input_folder,
        "output_folder": output_folder,
        "label_target": label_target,
        "perturbation_scale": perturbation_scale,
        "output_xyz_only": output_xyz_only
    }

def upsample_by_bbox(input_folder, output_folder, label_target, output_xyz_only=False):   #-----------------------------------------------------基于边界框的标签上采样逻辑
    for file in os.listdir(input_folder):
        if file.endswith(".txt"):
            with open(os.path.join(input_folder, file), 'r') as f:
                lines = f.readlines()

            points = [line.strip().split() for line in lines]
            label_points = [p for p in points if float(p[6]) == label_target]

            if len(label_points) == 0:
                print(f"⚠ 文件 {file} 中无标签为 {label_target} 的点，跳过")
                continue

            # 边界框
            min_x = min(float(p[0]) for p in label_points)
            max_x = max(float(p[0]) for p in label_points)
            min_y = min(float(p[1]) for p in label_points)
            max_y = max(float(p[1]) for p in label_points)
            min_z = min(float(p[2]) for p in label_points)
            max_z = max(float(p[2]) for p in label_points)

            # 平均颜色
            avg_r = sum(float(p[3]) for p in label_points) / len(label_points)
            avg_g = sum(float(p[4]) for p in label_points) / len(label_points)
            avg_b = sum(float(p[5]) for p in label_points) / len(label_points)

            num_to_generate = int(2.5 * len(label_points))
            new_points = []
            for _ in range(num_to_generate):
                x = random.uniform(min_x, max_x)
                y = random.uniform(min_y, max_y)
                z = random.uniform(min_z, max_z)
                new_points.append([x, y, z, avg_r, avg_g, avg_b, label_target])

            points.extend(new_points)

            with open(os.path.join(output_folder, file), 'w') as f:
                for p in points:
                    if output_xyz_only:
                        f.write(' '.join(map(str, p[:3])) + '\n')
                    else:
                        f.write(' '.join(map(str, p)) + '\n')

            print(f"✔ {file} 基于边界框上采样完成，保存至：{output_folder}")


# 0.01->0.5
def upsample_points(input_folder, output_folder, label_target, perturbation_scale=0.01, output_xyz_only=False):    #----------------------------------------
# 从 input_folder 中读取 .txt 点云文件
# 对指定标签（默认是 label=2）的点进行上采样（复制 + 随机扰动）
# perturbation_scale 控制扰动强度，越大新增点分布越“发散”
    for file in os.listdir(input_folder):  # 遍历文件夹中的 .txt 文件
        if file.endswith(".txt"):  
            with open(os.path.join(input_folder, file), 'r') as f:  # 读取并解析点云数据
                lines = f.readlines()    # 每一行是一个点，格式为： x y z R G B label


            # 提取所有点，筛选标签为 2 的点
            points = [line.strip().split() for line in lines]  # 把所有点拆分成数值列表；
            label_target_points = [p for p in points if float(p[6]) == label_target]

            if len(label_target_points) == 0:
                print(f"⚠ 文件 {file} 中未找到标签为 {label_target} 的点，跳过该文件")
                continue

            # 对每个点进行上采样（复制 2 倍数量）
            num_to_generate = int(2 * len(label_target_points))  # 设置上采样倍率为 2×，生成同样标签的点；
            new_points = []
            for _ in range(num_to_generate):  
                chosen = random.choice(label_target_points)  # 每次随机从 label 的点中选一个做扰动。
                # 0.5->0.3  2.000000->1.000000
                perturbed_x = float(chosen[0]) + perturbation_scale * (random.random() - 0.3)
                perturbed_y = float(chosen[1]) + perturbation_scale * (random.random() - 0.3)
                perturbed_z = float(chosen[2]) + perturbation_scale * (random.random() - 0.3)
                # 对坐标加扰动，生成新点,在原坐标附近微扰 0.01 量级（即 1% 左右随机偏移）。
                # perturbation_scale 默认是 0.01；
                # random.random() 生成一个 [0, 1) 的随机浮点数；
                # 减去 0.3 让扰动值分布大致在 [-0.3, 0.7) 区间；
                # 所以乘积范围约为：-0.3 * 0.01 = -0.003  0.7 * 0.01 =  0.007  即：±0.01 范围内的小偏移。
                # 为什么加扰动？因为如果只复制点，所有新点和原点完全重合，毫无意义。微扰使新点稍微“扩散”，避免训练中被模型判为重复值。

                new_point = [
                    perturbed_x, perturbed_y, perturbed_z,
                    chosen[3], chosen[4], chosen[5], label_target
                ]
                new_points.append(new_point)

                
                # 组装新点、赋值 RGB 和标签
                # 保留 RGB；强制标签为 2.0（不变）。
                

            # 将新的点添加到原始数据中
            points.extend(new_points)

            # 保存数据到新文件,把原始点 + 上采样生成点 一起保存到输出目录。
            # 保存数据到新文件
            with open(os.path.join(output_folder, file), 'w') as f:
                 for p in points:
                     if output_xyz_only:
                         f.write(' '.join(map(str, p[0:3])) + '\n')  # 只输出 xyz
                     else:
                         f.write(' '.join(map(str, p)) + '\n')  # 输出全字段


            print(f"✔ {file} 上采样完成，保存至：{output_folder}")

# 使用方法
# 一种更好的方法来上采样点云数据，即使用随机采样的方法来复制标签为2的点，并对其位置进行微小的随机扰动。
# 这样做的好处是生成的点更加紧密地分布在原始点的附近，从而更好地保留了点云的结构。
# 在这个方法中，我们使用了一个名为 perturbation_scale 的参数来控制随机扰动的大小。您可以根据需要调整这个参数。
# 默认情况下，它设置为0.01，这意味着新生成的点的位置可能会在原始点的位置上下浮动1%。
if __name__ == "__main__":
    config = get_user_upsample_config()
    strategy = config.pop("strategy")

    if strategy == "1":
        upsample_points(**config)
    elif strategy == "2":
        upsample_points(**config)  # 已强制扰动为0.0
    elif strategy == "3":
        upsample_by_bbox(**config)
    else:
        print("❌ 无效策略，程序退出。")

    print("\n 全部处理完成。")
'''

'''
输入输出总结
✅ 输入文件要求：
格式：.txt 文件；
每行格式：
x y z R G B label（共 7 列）；
一般来自 .ply 转换结果，例如 ply_random_sampling_plus.py 生成的 .txt 文件。

✅ 输出文件：
格式：.txt 文件；
每行结构保持不变；
点数 比原文件更多，label==2 的点被随机复制并扰动；
文件名与输入一致，保存在新目录。

'''


'''
示例运行输出演示：

欢迎使用点云上采样工具（增强版）

请选择上采样策略：
1 - 标签点复制 + 加扰动
2 - 标签点直接复制（无扰动）
3 - 基于边界框生成新点（随机坐标 + 平均RGB）
> 3

请输入输入文件夹路径（如 ./input）:
> ./my_input
请输入输出文件夹路径（如 ./output，请与输入文件夹不同）:
> ./my_output

请输入你希望上采样的标签号（如 2）:
> 2

请选择输出格式：
1 - 输出完整点信息（x y z R G B label）
2 - 只输出坐标信息（x y z）
> 1

✔ pointcloud01.txt 基于边界框上采样完成，保存至：./my_output
✔ pointcloud02.txt 基于边界框上采样完成，保存至：./my_output

 全部处理完成。




'''


#版本2.0

import os
import random
import numpy as np


def bbox_random_upsample(file_path, output_path, label=2, upsample_ratio=2.5, use_avg_color=True, add_perturb=False):#--------------------边界框随机采样 / upsample_1.5.py
    with open(file_path, 'r') as f:
        lines = f.readlines()

    points = [line.strip().split() for line in lines]
    label_points = [p for p in points if int(float(p[6])) == label]

    if len(label_points) == 0:
        print(f"{os.path.basename(file_path)} 中未找到标签为 {label} 的点，跳过。")
        return

    coords = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in label_points])
    min_xyz = coords.min(axis=0)
    max_xyz = coords.max(axis=0)

    if use_avg_color:
        avg_color = np.mean([[float(p[3]), float(p[4]), float(p[5])] for p in label_points], axis=0)
    else:
        color_list = [[float(p[3]), float(p[4]), float(p[5])] for p in label_points]

    num_to_generate = int(upsample_ratio * len(label_points))
    new_points = []

    for _ in range(num_to_generate):
        new_xyz = np.random.uniform(min_xyz, max_xyz)
        if use_avg_color:
            rgb = avg_color
        else:
            rgb = random.choice(color_list)

        if add_perturb:
            new_xyz += np.random.normal(scale=0.01, size=3)

        new_point = list(new_xyz) + list(rgb) + [label]
        new_points.append(new_point)

    points.extend(new_points)

    with open(output_path, 'w') as f:
        for p in points:
            f.write(' '.join(map(str, p)) + '\n')


def copy_perturb_upsample(file_path, output_path, label=2, upsample_ratio=2.0):  #----------------------------------------------标签点复制 + 扰动 / random_upsample.py
    with open(file_path, 'r') as f:
        lines = f.readlines()

    points = [line.strip().split() for line in lines]
    label_points = [p for p in points if int(float(p[6])) == label]

    if len(label_points) == 0:
        print(f"{os.path.basename(file_path)} 中未找到标签为 {label} 的点，跳过。")
        return

    new_points = []
    for _ in range(int(len(label_points) * upsample_ratio)):
        p = random.choice(label_points)
        xyz = [float(v) + np.random.normal(scale=0.01) for v in p[:3]]
        rgb = [float(v) for v in p[3:6]]
        new_points.append(xyz + rgb + [label])

    points.extend(new_points)

    with open(output_path, 'w') as f:
        for p in points:
            f.write(' '.join(map(str, p)) + '\n')

'''
def gaussian_upsample(file_path, output_path, label=2, upsample_ratio=2.0):  #--------------------------------------------------高斯扰动采样  
    with open(file_path, 'r') as f:
        lines = f.readlines()

    points = [line.strip().split() for line in lines]
    label_points = [p for p in points if int(float(p[6])) == label]

    if len(label_points) == 0:
        print(f"{os.path.basename(file_path)} 中未找到标签为 {label} 的点，跳过。")
        return

    coords = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in label_points])
    colors = np.array([[float(p[3]), float(p[4]), float(p[5])] for p in label_points])

    mean_xyz = np.mean(coords, axis=0)
    cov_xyz = np.cov(coords.T)

    new_points = []
    for _ in range(int(len(label_points) * upsample_ratio)):
        xyz = np.random.multivariate_normal(mean_xyz, cov_xyz)
        color = colors[random.randint(0, len(colors) - 1)]
        new_points.append(list(xyz) + list(color) + [label])

    points.extend(new_points)

    with open(output_path, 'w') as f:
        for p in points:
            f.write(' '.join(map(str, p)) + '\n')
'''

def get_label_from_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    labels = [int(float(line.strip().split()[6])) for line in lines]
    return max(set(labels), key=labels.count)


def main():
    print("请选择上采样策略：")
    print("1 - 边界框随机采样：标签号 / 上采样倍数 / 平均颜色（可选） / 坐标扰动（可选）")
    print("2 - 标签点复制 + 扰动： 标签号 / 上采样倍数 / 坐标扰动（强制）")
    #print("3 - 高斯扰动上采样：标签号 / 上采样倍数 / 坐标扰动（强制）")
    strategy = input("请输入你的选择（1/2）：").strip()

    input_dir = input("请输入输入路径：").strip()
    output_dir = input("请输入输出路径：").strip()
    os.makedirs(output_dir, exist_ok=True)

    label_input = input("请输入要上采样的标签号（留空自动识别）：").strip()
    label = int(label_input) if label_input else None

    ratio = float(input("请输入上采样倍数（例如 2.5）：").strip())

    use_avg_color = False
    add_perturb = False

    if strategy == '1':
        color_opt = input("是否使用平均颜色生成新点？(y/n)：").strip().lower()
        use_avg_color = (color_opt == 'y')
        perturb_opt = input("是否为新点加入扰动？(y/n)：").strip().lower()
        add_perturb = (perturb_opt == 'y')
    elif strategy in ['2', '3']:
        print("该策略默认使用扰动")

    for file in os.listdir(input_dir):
        if file.endswith(".txt"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)

            label_used = label if label is not None else get_label_from_file(input_path)

            if strategy == '1':
                bbox_random_upsample(input_path, output_path, label_used, ratio, use_avg_color, add_perturb)
                method_name = "边界框随机采样"
            elif strategy == '2':
                copy_perturb_upsample(input_path, output_path, label_used, ratio)
                method_name = "复制+扰动上采样"
            elif strategy == '3':
                gaussian_upsample(input_path, output_path, label_used, ratio)
                method_name = "高斯扰动上采样"
            else:
                print("策略无效，退出。")
                return

            print(f"{file} {method_name}完成，保存至：{output_dir}")

    print("全部处理完成！")


if __name__ == "__main__":
    main()

'''
示例输出：

策略1：边界框随机采样
请选择上采样策略：
1 - 边界框随机采样：标签号 / 上采样倍数 / 平均颜色（可选） / 坐标扰动（可选）
2 - 标签点复制 + 扰动： 标签号 / 上采样倍数 / 坐标扰动（强制）
请输入你的选择（1/2）：1

请输入输入路径：./my_input
请输入输出路径：./my_output
请输入要上采样的标签号（留空自动识别）：2
请输入上采样倍数（例如 2.5）：2.5
是否使用平均颜色生成新点？(y/n)：y
是否为新点加入扰动？(y/n)：n

file1.txt 边界框随机采样完成，保存至：./my_output
file2.txt 边界框随机采样完成，保存至：./my_output

全部处理完成！
-----------------------------------------------------------------------
策略2：标签点复制 + 扰动
请选择上采样策略：
1 - 边界框随机采样：标签号 / 上采样倍数 / 平均颜色（可选） / 坐标扰动（可选）
2 - 标签点复制 + 扰动： 标签号 / 上采样倍数 / 坐标扰动（强制）
3 - 高斯扰动上采样：标签号 / 上采样倍数 / 坐标扰动（强制）
请输入你的选择（1/2/3）：2

请输入输入路径：./my_input
请输入输出路径：./my_output
请输入要上采样的标签号（留空自动识别）：2
请输入上采样倍数（例如 2.5）：2.0
该策略默认使用扰动

file1.txt 复制+扰动上采样完成，保存至：./my_output
file2.txt 复制+扰动上采样完成，保存至：./my_output

全部处理完成！
--------------------------------------------------------------------------
策略3：高斯扰动上采样     ----------------------------------------------------------注销了
请选择上采样策略：
1 - 边界框随机采样：标签号 / 上采样倍数 / 平均颜色（可选） / 坐标扰动（可选）
2 - 标签点复制 + 扰动： 标签号 / 上采样倍数 / 坐标扰动（强制）
3 - 高斯扰动上采样：标签号 / 上采样倍数 / 坐标扰动（强制）
请输入你的选择（1/2/3）：3

请输入输入路径：./my_input
请输入输出路径：./my_output
请输入要上采样的标签号（留空自动识别）：2
请输入上采样倍数（例如 2.5）：1.5
该策略默认使用扰动

file1.txt 高斯扰动上采样完成，保存至：./my_output
file2.txt 高斯扰动上采样完成，保存至：./my_output

全部处理完成！
---------------------------------------------------------------------------
1. 边界框随机采样（bbox_random_upsample）
核心思路：
  先找出指定标签的所有点，计算它们的三维坐标的最小值和最大值，构成一个包围这些点的边界框。
  在边界框内随机均匀采样新点（xyz坐标随机生成在边界框范围内）。
  新点颜色可以用指定标签点的平均颜色，也可以随机挑选原有点的颜色。
  可以选择是否给新点坐标加入小扰动。

待处理数据流程：
  读取所有点。
  筛选指定标签点，计算边界框。
  随机生成新点坐标在边界框内。
  根据颜色设置（平均或随机），赋颜色。
  加扰动（可选）。
  新点加到原数据里，写文件。

上采样结果特点：
  新点分布较为均匀，覆盖标签点所在空间范围。
  不完全依赖原始点的具体位置分布，可能生成边界框内的“空白”区域点。
  颜色较统一（如果用平均颜色），或保留原色彩多样性（随机颜色）。

2. 标签点复制 + 扰动（copy_perturb_upsample）
核心思路：
  从指定标签点中随机复制点。
  在复制的点坐标上加入小的高斯扰动，使得新点稍微偏移，但位置和原点非常接近。
  颜色保持原点颜色。

待处理数据流程：
  读取所有点。
  筛选指定标签点。
  随机选标签点，复制并加扰动坐标。
  颜色沿用原点颜色。
  新点加到原数据里，写文件。

上采样结果特点：
  新点严格“簇”在原始点附近，分布密集。
  保留了原点的空间结构和颜色。
  有利于增强已有点簇密度，适合细节增强。

3. 高斯扰动上采样（gaussian_upsample）
核心思路：
  先计算指定标签点的均值坐标和协方差矩阵。
  按该高斯分布（多变量正态分布）随机采样新点坐标。
  颜色随机挑选自原标签点颜色。
  新点更随机但依赖整体分布统计特性。

待处理数据流程：
  读取所有点。
  筛选指定标签点。
  计算点云均值和协方差矩阵。
  按高斯分布采样新点坐标。
  随机赋颜色。
  新点加到原数据里，写文件。

上采样结果特点：
  新点服从与原点云相似的整体统计分布。
  比边界框采样更贴合数据分布，但不像复制扰动那样严格局限于原点附近。
  适合数据整体形态保持的增强。



