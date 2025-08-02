import matplotlib.pyplot as plt

# Assuming your data is in a file called "data.txt" and is structured as:
# Ef=0.25 eV
# freq1 value1
# freq2 value2
# ...
# Ef=0.5 eV
# freq1 value1
# ...
# and so on...
''' #-------------------------------------------------------------------------------------------------------------------版本1.0
with open("data.txt", "r") as f:
    lines = f.readlines()

fermi_energies = []
data = {}
current_energy = None

for line in lines:
    if "Ef=" in line:
        current_energy = line.strip()
        fermi_energies.append(current_energy)
        data[current_energy] = []
    else:
        freq, value = map(float, line.split())
        data[current_energy].append((freq, value))

plt.figure(figsize=(10, 6))
for energy in fermi_energies:
    x = [item[0] for item in data[energy]]
    y = [item[1] for item in data[energy]]
    plt.plot(x, y, label=energy)

plt.title("Real Part vs THz Frequency")
plt.xlabel("Frequency (THz)")
plt.ylabel("Real Part (uS)")
plt.legend()
plt.grid(True)
plt.show()
'''

import os
import matplotlib.pyplot as plt

#multi_state：输入文件中包含多个状态（如 Ef=0.25 eV, Ef=0.5 eV），每个状态都有一组频率-数值对；

#frequency_value：表示横轴是频率，纵轴是对应的数值，即“频率曲线”。

def input_with_default(prompt, default):
    s = input(f"{prompt}（默认：{default}，直接回车使用默认）:\n> ").strip()
    return s if s else default

def main():
    print("=== 数据绘图工具 ===")
    input_path = input("请输入数据文件路径（如 data.txt）：\n> ").strip()
    while not os.path.isfile(input_path):
        print("文件不存在，请重新输入。")
        input_path = input("请输入数据文件路径（如 data.txt）：\n> ").strip()

    output_dir = input("请输入输出图片保存文件夹路径（不存在则自动创建）：\n> ").strip()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"目录 {output_dir} 已创建。")

    # 默认绘图参数
    default_fig_width = 10
    default_fig_height = 6
    default_title = "Real Part vs THz Frequency"
    default_xlabel = "Frequency (THz)"
    default_ylabel = "Real Part (uS)"

    # 用户输入绘图参数，按回车用默认
    fig_width = float(input_with_default("请输入图形宽度", default_fig_width))
    fig_height = float(input_with_default("请输入图形高度", default_fig_height))
    title = input_with_default("请输入图形标题", default_title)
    xlabel = input_with_default("请输入X轴标签", default_xlabel)
    ylabel = input_with_default("请输入Y轴标签", default_ylabel)

    # 读取文件
    with open(input_path, "r") as f:
        lines = f.readlines()

    fermi_energies = []
    data = {}
    current_energy = None

    for line in lines:
        if "Ef=" in line:
            current_energy = line.strip()
            fermi_energies.append(current_energy)
            data[current_energy] = []
        else:
            freq, value = map(float, line.split())
            data[current_energy].append((freq, value))

    # 绘图
    plt.figure(figsize=(fig_width, fig_height))
    for energy in fermi_energies:
        x = [item[0] for item in data[energy]]
        y = [item[1] for item in data[energy]]
        plt.plot(x, y, label=energy)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    # 生成输出文件名，加后缀 "_processed.png"
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_processed.png")

    # 保存图像
    plt.savefig(output_file)
    plt.close()

    print(f"\n 绘图完成，图像已保存至：{output_file}")

if __name__ == "__main__":
    main()

'''
示例输出：

=== 数据绘图工具 ===
请输入数据文件路径（如 data.txt）：
> data.txt
请输入输出图片保存文件夹路径（不存在则自动创建）：
> ./output_images
目录 ./output_images 已创建。
请输入图形宽度（默认：10，直接回车使用默认）:
> 
请输入图形高度（默认：6，直接回车使用默认）:
> 
请输入图形标题（默认：Real Part vs THz Frequency，直接回车使用默认）:
> 频率 vs 电导率
请输入X轴标签（默认：Frequency (THz)，直接回车使用默认）:
> 频率(THz)
请输入Y轴标签（默认：Real Part (uS)，直接回车使用默认）:
> 电导率(uS)

 绘图完成，图像已保存至：./output_images/data_processed.png
'''