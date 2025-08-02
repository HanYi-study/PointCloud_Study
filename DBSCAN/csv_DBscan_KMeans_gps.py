import pandas as pd  # 用于读取 CSV 文件并进行表格数据处理。
import numpy as np  # 数组处理与数值计算库。
from sklearn.cluster import DBSCAN  # 一种基于密度的聚类算法。
from sklearn import metrics
from sklearn.cluster import KMeans  # 用于对每个 DBSCAN 聚类结果再次做中心点提取。
import os

'''----------------------------------------------------------------------------------------------------------------------------版本1.0
该脚本主要完成以下任务：
  从多个 .csv 文件中读取 GPS 坐标数据（经度和纬度）；
  使用 DBSCAN（基于球面距离的 Haversine 度量）进行聚类分析，找出具有空间密度的聚类簇；
  对每个聚类簇使用 KMeans 聚类算法找出中心点坐标（虽然指定了 n_clusters=1，本质是寻找均值中心）；
  打印输出每个聚类簇的中心点坐标。

输入数据：
  输入文件格式：CSV
  文件内容：每行包含两个值，分别是经度（lon）和纬度（lat）
  没有表头
输出数据：
  无输出文件，只有控制台打印：
     聚类簇数目；
     每个聚类簇的中心点（由 KMeans 计算）坐标。

分析结果/能获得的数据意义：
  DBSCAN 聚类结果：
     将密集的 GPS 点聚集成“簇”；
     可有效识别热点区域、建筑、路口或人群集中点；
     能自动剔除离群点（DBSCAN 的 label = -1 被忽略）；
  KMeans 寻找每簇中心：
     输出每个聚类簇的“代表点”或几何中心；
     可用于构建简化模型或地图摘要，如“兴趣点提取”。

'''

'''
all_data=pd.read_csv("结点数据1000.csv")
print(all_data.head())  # 打印前 5 行查看内容（用于初步理解数据格式）。

def dbscan(input_file):
    columns=['lon','lat']  # 设置字段名
    in_df = pd.read_csv(input_file, sep=',', header=None, names=columns)  # 读取经纬度数据
    # header=None：表示 CSV 文件没有标题行。   names=columns：人为指定列名为 lon 和 lat。

    # 转为 numpy 数组：
    coords = in_df[['lon', 'lat']].to_numpy()
    # 将 pandas 表格中经纬度列转为 NumPy 数组，用于后续聚类。 

    # 设置 DBSCAN 的参数：
    kms_per_radian = 6371.0086  # 地球半径约 6371 公里。
    # 这里的 0.5 指允许的聚类半径为 0.5 公里（转换为弧度单位，用于 haversine 距离计算）
    # http://www.movable-type.co.uk/scripts/latlong.html
    epsilon = 0.5 / kms_per_radian

    # 执行 DBSCAN 聚类：
    db = DBSCAN(eps=epsilon, min_samples=15, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    # eps=epsilon：聚类最大距离为 0.5 公里。 
    # min_samples=15：每个聚类至少包含 15 个点。
    # metric='haversine'：使用球面距离计算（适用于经纬度）。
    # np.radians(coords)：将角度转换为弧度。

    # 获取聚类标签并打印聚类数：
    cluster_labels = db.labels_
    # db.labels_：返回每个点所属的聚类编号（-1 表示噪声点）。
    num_clusters = len(set(cluster_labels) - set([-1]))
    print( 'Clustered ' + str(len(in_df)) + ' points to ' + str(num_clusters) + ' clusters')  # 打印聚类总数（不包括噪声点）。

    # 对每个聚类使用 KMeans 提取中心：
    kmeans = KMeans(n_clusters=1, n_init=1, max_iter=20, random_state=20)
    # 设置一个 1 类的 KMeans（即只求质心）。

    for n in range(num_clusters):
        one_cluster = coords[cluster_labels == n]
        kk = kmeans.fit(one_cluster)
        print(kk.cluster_centers_)
    # 遍历所有聚类簇。
    # 提取当前簇的所有坐标点，使用 KMeans 找到它们的中心点。
    # 打印该中心点坐标。

def main():
    path = '/media/yunl/晴晴'
    filelist = os.listdir(path)
    for f in filelist:
        datafile = os.path.join(path,f)
        print(datafile)
        dbscan(datafile)

if __name__ == '__main__':
    main()
'''

def run_dbscan(input_file, output_dir, eps_km, min_samples):
    # 读取输入文件（无表头，默认两列为经纬度）
    in_df = pd.read_csv(input_file, sep=',', header=None, names=['lon', 'lat'])
    coords = in_df[['lon', 'lat']].to_numpy()

    # 参数处理
    kms_per_radian = 6371.0086
    epsilon = eps_km / kms_per_radian

    # DBSCAN 聚类（使用 haversine 球面距离）
    db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    cluster_labels = db.fit_predict(np.radians(coords))

    num_clusters = len(set(cluster_labels) - {-1})
    print(f" 文件 {os.path.basename(input_file)} 聚类结果：{num_clusters} 个簇")

    # 对每个聚类提取中心点
    centers = []
    kmeans = KMeans(n_clusters=1, n_init=1, max_iter=20, random_state=42)

    for n in range(num_clusters):
        cluster_points = coords[cluster_labels == n]
        if len(cluster_points) == 0:
            continue
        center = kmeans.fit(cluster_points).cluster_centers_[0]
        centers.append(center)

    # 显示聚类中心
    for i, c in enumerate(centers):
        print(f" 聚类 {i} 中心点坐标: 经度 = {c[0]:.6f}, 纬度 = {c[1]:.6f}")

    # 保存聚类中心为 CSV 文件
    base = os.path.basename(input_file)
    name = os.path.splitext(base)[0]
    output_path = os.path.join(output_dir, name + "_clustered.csv")

    centers_df = pd.DataFrame(centers, columns=['lon_center', 'lat_center'])
    centers_df.to_csv(output_path, index=False)
    print(f" 聚类中心已保存至：{output_path}\n")

def main():
    print("=== GPS 点聚类分析器 ===")

    input_dir = input(" 请输入包含输入 .csv 文件的目录路径：\n> ").strip()
    output_dir = input(" 请输入处理后结果保存的目录路径：\n> ").strip()
    if not os.path.isdir(input_dir) or not os.path.isdir(output_dir):
        print(" 输入路径无效，请确认目录存在。")
        return

    eps_km = float(input(" 设置 DBSCAN 聚类半径 / 单位距离（单位：公里，如 0.5 表示 500 米 / 值越大越容易形成大簇）：\n> ").strip())
    min_samples = int(input(" 设置 DBSCAN 最小样本数 / 每个簇中最少点数量 （如15， 每个聚类至少包含 15 个点 / 值越大：更严格、更容易标记为噪声）：\n> ").strip())

    print("\n 开始处理所有 CSV 文件...\n")
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(input_dir, filename)
            run_dbscan(filepath, output_dir, eps_km, min_samples)

    print(" 所有文件处理完成！")

if __name__ == "__main__":
    main()

'''
输出结果示例：
=== GPS 点聚类分析器 ===
 请输入包含输入 .csv 文件的目录路径：
> /home/user/gps_input
 请输入处理后结果保存的目录路径：
> /home/user/gps_output
 设置 DBSCAN 聚类半径（单位：公里，如 0.5 表示 500 米）：
> 0.3
 设置 DBSCAN 最小样本数（如 10~20）：
> 15

 开始处理所有 CSV 文件...

 文件 gps1.csv 聚类结果：4 个簇
 聚类 0 中心点坐标: 经度 = 121.453210, 纬度 = 31.236850
 聚类 1 中心点坐标: 经度 = 121.455830, 纬度 = 31.238260
...
 聚类中心已保存至：/home/user/gps_output/gps1_clustered.csv
--------------------------------------------------------------
推荐文件结构：
DBSCAN/
├── gps_clustering_custom.py
├── input/
│   └── gps1.csv
├── output/
│   └── gps1_clustered.csv

'''