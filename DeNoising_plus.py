import numpy as np
import os

def read_point_cloud(path):
    if path.endswith('.npy'):
        return np.load(path)
    elif path.endswith('.txt'):
        return np.loadtxt(path)
    else:
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(path)
            return np.asarray(pcd.points)
        except ImportError:
            raise RuntimeError("请安装open3d以支持ply等格式")
        
def write_point_cloud(path, points):
    if path.endswith('.npy'):
        np.save(path, points)
    elif path.endswith('.txt'):
        np.savetxt(path, points)
    else:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(path, pcd)

def statistical_outlier_removal(points, k=16, std_ratio=2.0):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
    distances, _ = nbrs.kneighbors(points)
    avg_distances = np.mean(distances[:, 1:], axis=1)
    mean = np.mean(avg_distances)
    std = np.std(avg_distances)
    mask = avg_distances < mean + std_ratio * std
    return points[mask]

def radius_outlier_removal(points, radius=0.05, min_neighbors=5):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(radius=radius).fit(points)
    neighbors = nbrs.radius_neighbors(points, return_distance=False)
    mask = np.array([len(n) > min_neighbors for n in neighbors])
    return points[mask]

def knn_smooth(points, k=16):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
    _, indices = nbrs.kneighbors(points)
    smoothed = np.array([points[neighbors[1:]].mean(axis=0) for neighbors in indices])
    return smoothed

def dbscan_denoising(points, eps=0.05, min_samples=10):
    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    unique, counts = np.unique(labels, return_counts=True)
    main_label = unique[np.argmax(counts[unique != -1])]
    mask = labels == main_label
    return points[mask]

def gaussian_filter_denoising(points, sigma=1.0):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(points, sigma=sigma, axis=0)

def mls_denoising(input_path, output_path, search_radius=0.05):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(input_path)
    pcd = pcd.voxel_down_sample(voxel_size=search_radius/2)
    pcd = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]
    pcd = pcd.uniform_down_sample(every_k_points=1)
    pcd = pcd.estimate_normals()
    pcd = o3d.geometry.PointCloud.create_from_point_cloud_mls(pcd, search_radius=search_radius)
    o3d.io.write_point_cloud(output_path, pcd)

def main():
    print("==== 点云去噪工具 ====")
    input_path = input("请输入要处理的文件路径或文件夹路径：\n> ").strip()
    is_dir = os.path.isdir(input_path)

    output_dir = input("请输入保存输出的文件夹路径：\n> ").strip()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n请选择去噪方法：")
    print("1 - Statistical Outlier Removal（统计离群点去除）")
    print("2 - Radius Outlier Removal（半径离群点去除）")
    print("3 - KNN Smooth（KNN平滑）")
    print("4 - DBSCAN（聚类去噪）")
    print("5 - Gaussian Filter（高斯滤波）")
    print("6 - MLS（移动最小二乘）")
    method = input("> ").strip()

    # 参数输入
    params = {}
    if method == "1":
        params["k"] = int(input("请输入K值（默认16）：\n> ") or 16)
        params["std_ratio"] = float(input("请输入std_ratio（默认2.0）：\n> ") or 2.0)
    elif method == "2":
        params["radius"] = float(input("请输入半径radius（默认0.05）：\n> ") or 0.05)
        params["min_neighbors"] = int(input("请输入最小邻居数min_neighbors（默认5）：\n> ") or 5)
    elif method == "3":
        params["k"] = int(input("请输入K值（默认16）：\n> ") or 16)
    elif method == "4":
        params["eps"] = float(input("请输入eps（默认0.05）：\n> ") or 0.05)
        params["min_samples"] = int(input("请输入min_samples（默认10）：\n> ") or 10)
    elif method == "5":
        params["sigma"] = float(input("请输入sigma（默认1.0）：\n> ") or 1.0)
    elif method == "6":
        params["radius"] = float(input("请输入MLS搜索半径radius（默认0.05）：\n> ") or 0.05)
    else:
        print("无效选择，默认使用 Statistical Outlier Removal")
        method = "1"
        params["k"] = 16
        params["std_ratio"] = 2.0

    # 批处理或单文件
    def process_one(src, dst):
        print(f"读取点云文件: {src}")
        if method == "1":
            points = read_point_cloud(src)
            filtered = statistical_outlier_removal(points, params["k"], params["std_ratio"])
            write_point_cloud(dst, filtered)
        elif method == "2":
            points = read_point_cloud(src)
            filtered = radius_outlier_removal(points, params["radius"], params["min_neighbors"])
            write_point_cloud(dst, filtered)
        elif method == "3":
            points = read_point_cloud(src)
            filtered = knn_smooth(points, params["k"])
            write_point_cloud(dst, filtered)
        elif method == "4":
            points = read_point_cloud(src)
            filtered = dbscan_denoising(points, params["eps"], params["min_samples"])
            write_point_cloud(dst, filtered)
        elif method == "5":
            points = read_point_cloud(src)
            filtered = gaussian_filter_denoising(points, params["sigma"])
            write_point_cloud(dst, filtered)
        elif method == "6":
            mls_denoising(src, dst, params["radius"])
        print(f" 去噪完成，保存至：{dst}")

    if is_dir:
        files = [f for f in os.listdir(input_path) if f.endswith((".ply", ".txt", ".npy"))]
        for f in files:
            src = os.path.join(input_path, f)
            dst = os.path.join(output_dir, f)
            process_one(src, dst)
    else:
        filename = os.path.basename(input_path)
        dst = os.path.join(output_dir, filename)
        process_one(input_path, dst)

if __name__ == "__main__":
    main()