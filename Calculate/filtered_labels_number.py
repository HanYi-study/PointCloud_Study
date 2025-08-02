''' #------------------------------------------------------------------------------------------------------------------版本1.0
import os

def count_values_in_files(folder_path):
    count_0, count_1, count_2 = 0, 0, 0

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as f:
                local_count_0, local_count_1, local_count_2 = 0, 0, 0
                for line in f:
                    values = line.strip().split()
                    if values:  # 确保values列表不为空
                        try:
                            last_value = float(values[-1])
                            if last_value == 0.000000:
                                local_count_0 += 1
                                count_0 += 1
                            elif last_value == 1.000000:
                                local_count_1 += 1
                                count_1 += 1
                            elif last_value == 2.000000:
                                local_count_2 += 1
                                count_2 += 1
                        except ValueError:
                            # 忽略无法转换为浮点数的行
                            pass

                print(f"In file '{file}', Count of 0.000000: {local_count_0}, Count of 1.000000: {local_count_1}, Count of 2.000000: {local_count_2}")

    print(f"\nTotal Count in folder - 0.000000: {count_0}, 1.000000: {count_1}, 2.000000: {count_2}")
    print(f"\nFolder point cloud percent - 0.000000: {count_0/(count_0 + count_1 + count_2)}, 1.000000: {count_1/(count_0 + count_1 + count_2)}, 2.000000: {count_2/(count_0 + count_1 + count_2)}")

folder_path = '/home/zxb/Poppy/Data/paper_deepleaning_lvbo/fenlei_3/version2_data/lvbo_0ds2.5_2us2.5_1us2_hy20_0.1_guass_10%/txt/test'  # Replace this with the path to your folder
count_values_in_files(folder_path)

'''
#---------------------------------------------------------------------------------------------计算滤波后数据所有标签点的个数
import os
import numpy as np
import csv

def count_labels_in_files(input_folder, output_path, save_as_csv=True):
    total_counts = {}
    all_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

    for file in all_files:
        file_path = os.path.join(input_folder, file)
        local_counts = {}

        with open(file_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                if values:
                    try:
                        label = int(float(values[-1]))
                        total_counts[label] = total_counts.get(label, 0) + 1
                        local_counts[label] = local_counts.get(label, 0) + 1
                    except ValueError:
                        continue  # 忽略无法转换的行

        print(f"文件 {file} 标签计数: {local_counts}")

    total_sum = sum(total_counts.values())
    print("\n 总体统计：")
    for label, count in sorted(total_counts.items()):
        print(f"标签 {label}: 数量 = {count}, 占比 = {count / total_sum:.4f}")

    # 保存文件
    if save_as_csv:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['标签', '数量', '占比'])
            for label in sorted(total_counts.keys()):
                writer.writerow([label, total_counts[label], total_counts[label] / total_sum])
    else:
        with open(output_path, 'w') as f:
            for label in sorted(total_counts.keys()):
                f.write(f"标签 {label}: 数量 = {total_counts[label]}, 占比 = {total_counts[label] / total_sum:.4f}\n")

    print(f"\n 统计结果已保存至：{output_path}")

# ========== 主函数入口 ==========
if __name__ == "__main__":
    print("=== 标签计数器 ===")
    print("请确保输入的txt文件符合以下要求：")
    print("1. 每行的最后一列是“标签”（例如 0.000000, 1.000000, 2.000000 等）")
    print("2. 每行可以有多个字段，但只取最后一个字段作为标签")
    print("3. 标签应为数字格式（float/int 可混用）")
    print("4. 如果某行无法被解析为数字，会自动跳过\n")

    input_folder = input("请输入待分析 .txt 文件所在目录路径：\n> ").strip()
    output_folder = input("请输入统计结果保存目录路径：\n> ").strip()

    if not os.path.isdir(input_folder) or not os.path.isdir(output_folder):
        print(" 输入路径有误，请确保目录存在。")
        exit()

    format_choice = input("请选择输出格式：1 - .txt，2 - .csv\n> ").strip()
    save_as_csv = (format_choice == '2')

    output_filename = "label_count_result.csv" if save_as_csv else "label_count_result.txt"
    output_path = os.path.join(output_folder, output_filename)

    count_labels_in_files(input_folder, output_path, save_as_csv)



'''
示例输入输出:

=== 标签计数器 ===
请输入待分析 .txt 文件所在目录路径：
> /home/xxx/input_txts
请输入统计结果保存目录路径：
> /home/xxx/output
请选择输出格式：1 - .txt，2 - .csv
> 2
文件 test1.txt 标签计数: {0: 132, 1: 421, 2: 137}
文件 test2.txt 标签计数: {0: 152, 1: 389, 2: 162}

 总体统计：
标签 0: 数量 = 284, 占比 = 0.2283
标签 1: 数量 = 810, 占比 = 0.6517
标签 2: 数量 = 299, 占比 = 0.1200

 统计结果已保存至：/home/xxx/output/label_count_result.csv


'''