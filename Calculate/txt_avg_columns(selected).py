''' #--------------------------------------------------------------------------------------------------------------------------------版本1.0
# 打开原始文件和新文件
with open('val_IoUs_yh_9yue7.txt', 'r') as f_in, open('new_val_IoUs_yh_9yue7.txt', 'w') as f_out:
    # 逐行处理原始文件
    for line in f_in:
        # 分割每一行的内容
        columns = line.strip().split()
        if len(columns) >= 6:
            # 将前三列转换为数字并相加
            sum_result = sum(map(float, columns[:3])) / 3.0
            # 将计算结果添加到原始行的末尾
            new_line = '\t'.join(columns + [str(sum_result)])
            # 将新行写入新文件
            f_out.write(new_line + '\n')

print("处理完成！")
'''

import os

def process_file(input_path, output_path, selected_cols):
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            columns = line.strip().split()
            try:
                selected_values = [float(columns[i]) for i in selected_cols if i < len(columns)]
                if len(selected_values) != len(selected_cols):
                    print(f"  第 {line_num} 行：列数不足，跳过。")
                    continue
                avg = sum(selected_values) / len(selected_values)
                new_line = '\t'.join(columns + [f"{avg:.6f}"])
                f_out.write(new_line + '\n')
            except (ValueError, IndexError) as e:
                print(f"  第 {line_num} 行发生错误，跳过：{e}")
    print(f"\n 处理完成！输出已保存至：{output_path}")

def main():
    print("=== 指定列均值计算器 ===")
    print("输入文件格式要求：")
    print("1. 文本文件，每一行至少包含你指定的列数量")
    print("2. 每列用空格或制表符分隔")
    print("例如某行： 0.500 0.600 0.550 0.90 0.80\n")
    input_path = input("请输入待处理文件的完整路径（如 /path/to/input.txt）：\n> ").strip()
    output_path = input("请输入处理后输出文件的完整路径（如 /path/to/output.txt）：\n> ").strip()

    cols_input = input("请输入要参与平均计算的列索引（从0开始，用逗号分隔，如 0,1,2）：\n> ").strip()
    try:
        selected_cols = [int(idx) for idx in cols_input.split(',')]
    except ValueError:
        print(" 输入的列索引格式不正确，请输入如 0,1,2")
        return

    if not os.path.isfile(input_path):
        print(" 输入文件不存在，请检查路径。")
        return

    process_file(input_path, output_path, selected_cols)

if __name__ == "__main__":
    main()


'''
示例输出：

=== 指定列均值计算器 ===
请输入待处理文件的完整路径（如 /path/to/input.txt）：
> /home/user/data/val_IoUs_yh_9yue7.txt
请输入处理后输出文件的完整路径（如 /path/to/output.txt）：
> /home/user/data/processed_val_IoUs.txt
请输入要参与平均计算的列索引（从0开始，用逗号分隔，如 0,1,2）：
> 0,1,2


'''