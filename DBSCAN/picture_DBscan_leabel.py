import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import preprocessing

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def perform_dbscan(input_file, output_dir, eps, min_samples):
    df = pd.read_csv(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    if 'VAR00003' not in df.columns or 'VAR00004' not in df.columns:
        print(f" 文件 {input_file} 中缺少必要字段 VAR00003 或 VAR00004")
        return

    X = preprocessing.scale(df[['VAR00003', 'VAR00004']])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    df['cluster'] = db.labels_

    # 保存聚类数据
    output_csv = os.path.join(output_dir, f"{base_name}_dbscan_eps{eps}_min{min_samples}.csv")
    df.to_csv(output_csv, index=False)

    # 可视化
    sns.set(style='whitegrid')
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='VAR00003', y='VAR00004', hue='cluster', data=df, palette='deep', legend='full')
    plt.title(f"{base_name} - DBSCAN 聚类 (eps={eps}, min_samples={min_samples})")
    plt.xlabel('出生率 VAR00003')
    plt.ylabel('死亡率 VAR00004')
    plt.legend(title='聚类簇')
    plot_path = os.path.join(output_dir, f"{base_name}_dbscan_plot.png")
    plt.savefig(plot_path)
    plt.close()

    print(f" 处理完成：{input_file}")
    print(f" 结果保存至：{output_csv}")
    print(f" 图像保存至：{plot_path}")

def main():
    print("===  DBSCAN 批量聚类分析工具 ===")
    input_path = input("请输入文件路径（支持单文件或文件夹）：\n> ").strip()
    output_dir = input("请输入输出文件夹路径：\n> ").strip()
    eps = float(input("请输入 eps 参数（如 0.5）：\n> ").strip())
    min_samples = int(input("请输入 min_samples 参数（如 5）：\n> ").strip())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.csv')]
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        print(" 输入路径无效。")
        return

    for file in files:
        perform_dbscan(file, output_dir, eps, min_samples)

    print("\n 全部聚类分析完成！")

if __name__ == "__main__":
    main()


'''
| 项目                      | `csv_DBscan_KMeans_gps.py` ✅          | `picture_DBscan_leabel.py` ✅               |
| --------------------      | ----------------------------           | ------------------------------------        |
| 📌 **聚类对象**           | **GPS 经纬度**（`lon`, `lat`）          | **普通数值指标**（如 `VAR00003`, `VAR00004`） |
| 📂 **输入文件格式**       | 无表头 CSV，只有两列坐标（lon, lat）     | 有表头 CSV，包含指标字段                       |
| 📊 **使用的距离度量**     | `haversine`（球面距离，适用于地理数据）   | 欧几里得距离（`metric='euclidean'`）          |
| 🔍 **目标任务**           | 获取每个 GPS 簇的**中心点坐标**         | 获取每个指标簇的**样本分组情况**，并可视化       |
| 🧠 **是否提取聚类中心**   | ✅ 使用 KMeans 取每个 DBSCAN 簇的中心点 | ❌ 不做中心点提取，只分簇                      |
| 📈 **是否生成图像可视化** | ❌ 无图像（纯控制台 + CSV 输出）         | ✅ 生成聚类图（scatter + hue=cluster）        |
| 📁 **是否批处理**         | ✅ 扫描输入目录所有 `.csv` 批量处理     | ✅ 单文件/目录皆可                            |
| 📄 **输出文件**           | `xxx_clustered.csv`：每簇中心坐标       | `xxx_dbscan_epsX_minY.csv`：所有样本带聚类标签|
| 🖼️ **输出图像文件**       | ❌ 无图像输出                          | ✅ `.png` 聚类图输出                         |
| ⚙️ **参数交互方式**       | 控制台输入 `eps_km`、`min_samples`      | 控制台输入 `eps`、`min_samples`（适用指标）   |
| 🧪 **标准化处理**         | ❌ 不标准化（经纬度不能随便缩放）        | ✅ 使用 `preprocessing.scale` 标准化指标     |
| 🧭 **核心应用场景**       | 地理坐标类聚类分析（如车、节点聚合）      | 指标评估型聚类分析（如人口统计、行为分类）      |


'''
