# 点云代码学习
此文档用于记录点云代码学习笔记

点云处理简要流程：采集 → 清洗预处理 → 添加噪声/增强 → 特征提取 → 输入模型训练 → 后处理 → 评估 → 可视化  
数据集Data和预处理后的结果Results，还有方法等日志Log（目前没有运行日志，但是可以后期运行存储运行日志）可以详见data_structure.txt

---

# 目录
 - 处理流程/逻辑展示
 - 文件结构说明
 - 文件功能说明

---

 # 处理流程/逻辑展示
 ## 1.创建虚拟环境
conda create -n PointCloud_02 python=3.9  
 ## 2.配置虚拟环境  
screen  
conda activate PointCloud_02  
#安装依赖  
pip install numpy==1.23.5 -i https://pypi.tuna.tsinghua.edu.cn/simple  
pip install pandas==1.5.3 -i https://pypi.tuna.tsinghua.edu.cn/simple  
pip install scikit-learn==1.2.2 -i https://pypi.tuna.tsinghua.edu.cn/simple  
pip install laspy==2.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple  
pip install plyfile==0.7.4 -i https://pypi.tuna.tsinghua.edu.cn/simple  
pip install open3d==0.17.0 -i https://pypi.tuna.tsinghua.edu.cn/simple  
pip install numpy cython  
pip install plyfile  (使用 plyfile 正确解析二进制 PLY)  

conda list  
  
    numpy                     1.23.5                   pypi_0    pypi
    open3d                    0.17.0                   pypi_0    pypi
    pandas                    1.5.3                    pypi_0    pypi
    pip                       25.1.1             pyh8b19718_0    conda-forge
    plyfile                   0.7.4                    pypi_0    pypi
    python                    3.10.18         hd6af730_0_cpython    conda-forge
    python-dateutil           2.9.0.post0              pypi_0    pypi
    rpds-py                   0.26.0                   pypi_0    pypi
    scikit-learn              1.2.2                    pypi_0    pypi
    tqdm                      4.67.1                   pypi_0    pypi
    laspy                     2.3.0                    pypi_0    pypi

## 3.编译
首先编译C++依赖模块  

    cd ~/projects/PointCloud_Code/nearest_neighbors  
    python setup.py build_ext --inplace  
  >此目录（nearest_neighbors）下生成`grid_subsampling.cpython-39-xxx.so` 文件   
   
再编译cpp\_subsampling 模块  

    cd ~/projects/PointCloud_Code/cpp_wrappers/cpp_subsampling  
    g++ -O3 -Wall -shared -std=c++14 -fPIC \
     -I$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])") \
     -I$(python3 -c "import numpy; print(numpy.get_include())") \
     cpp_wrappers/cpp_subsampling/grid_subsampling/grid_subsampling.cpp \
     cpp_wrappers/cpp_subsampling/wrapper.cpp \
     cpp_wrappers/cpp_utils/cloud/cloud.cpp \
     -o cpp_wrappers/cpp_subsampling/grid_subsampling.cpython-39-x86_64-linux-gnu.so  
    ls -lh cpp_wrappers/cpp_subsampling/*.so  #验证.so文件是否生成
  >验证结果输出：  
  rwxrwxr-x 1 hy hy 47K Jul 28 01:16  
  cpp_wrappers/cpp_subsampling/grid_subsampling.cpython-39-x86_64-linux-gnu. so  

    python
    >>from cpp_wrappers.cpp_subsampling import grid_subsampling
    >>dir(grid_subsampling)  
  >python代码中第一行对应的输出：grid_subsampling  
   python代码中第二行对应的输出：['__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'compute']  

## 4.预处理操作-->运行.py文件  
### 运行step0_read_las.py  
>输入：/home/hy/projects/PointCloud_Code/Data/Data_prepare/3-4(3_4).las  
输出：/home/hy/projects/PointCloud_Code/Results/data_prepare_result/step0_result/3-4(3_4).txt  
（三个.las文件都分别执行了一次该.py文件，每次执行都更改对应的输入输出路径）

### 运行step1_dara_prepare_H3D.py  
>输入：/home/hy/projects/PointCloud_Code/Results/data_prepare_result/step0_result  
输出：/home/hy/projects/PointCloud_Code/Results/data_prepare_result/step1_H3D_result  
（将step0的运行结果.txt文件作为这一步的输入数据）  
（最终在step1_H3D_result 中生成了两个文件夹：kdtree-1和points-1，并且这两个文件夹中，生成了三个数据集的对应格式文件，详情可见文件结构）

### 运行Data_conversion文件夹下的ply_to_txt.py  
>输入：/home/hy/projects/PointCloud_Code/Results/data_prepare_result/step1_H3D_result/points-1  
（step1_data_prepare_H3D.py的运行结果中point-1文件夹中所有.ply文件）  
输出：/home/hy/projects/PointCloud_Code/Results/data_conversion_result/H3D_points-1_ply_to_txt  （经过格式转换，转换为txt文件，并存放再输出路径下）   
（文件转换后的txt文件作为step2_data_process_randla.py的输入）

### 运行data_prepare文件夹下的step2_data_process_randla.py  
>输入：/home/hy/projects/PointCloud_Code/Results/data_conversion_result/H3D_points-1_ply_to_txt'  
（将上一步转换完生成的txt文件作为输入）  
输出：/home/hy/projects/PointCloud_Code/Results/data_prepare_result/step2_result  
（生成的结果有处理后的特征数据.txt文件、标签文件.label文件）  

---

# 文件结构说明  
  
目录结构详见data_structure.txt

---
 
# 文件功能说明  
# 一. data-prepare文件夹: 

 
## 1.step0_read_las.py：  
- **输入**：原始.las 格式点云数据（含空间坐标、颜色、分类标签等）   
 输出：内存中的numpy数组（x,y,z,RGB,label）、min_xyz起始坐标（后续可用于还原位置信息）  
数据状态：原始格式转换为标准数组  
 处理：  
    - a）解析.las文件结构      
    - b）提取xyz坐标（归一化到起点）、RGB颜色、标签（classification）
## 2. step1_data_prepare_H3D.py / step1_data_prepare_V3D.py：  
 两文件的区别：  
 H3D.py 处理的是 H3D 格式的点云数据（含 RGB、instance、label、可能还带 intensity）  
 V3D.py 针对的是 V3D 格式的点云数据（结构简洁，可能只有 XYZ 和 label）  
 输入：step0产出的数组数据  
 输出：.ply格式的点云文件、KD-Tree或其他中间结构缓存（如 .npy/.pkl 文件）  
 数据状态：已变为可视化或深度学习框架可读格式  
 操作：  
    - a）构建KD-Tree空间结构  
    - b）写出.ply文件：XYZ+RGB+Label  

## 3. step2_data_process_randla.py:  
 输入：.ply格式的点云文件  
 输出：是保存在output_dir目录下的两个文件，一个是处理后的点云特征数据文件，另一个是标签文件。  
 a）处理后的特征数据文件(同名.txt文件),把原数据中除了最后一列(label)以外的列都提取出来，在第4列插入值0，用于之后的模型输入、特征工程、数据增强等。  
 b）标签文件(同名.label文件),只保存每一个点的标签（最后一列0,通常用于分类监督学习模型的目标值。  
 数据状态：结构化、标准化、块化后的训练输入数据。  
 操作：  
    - a）对每个点云文件提取标签（最后一列）。  
    - b）修改点云特征（在每行的第四列插入0）。  
    - c）将处理后的数据和标签分别保存到输出目录中。


# 二. Noise_DataAugmentation**  
 （1）txts_guassian_noise.py  
 （2）txts_guassian_noise_limited.py
 （3）txt_guassian_noise_simple.py(已在上传文档中说明)  


## 1. txts_guassian_noise.py：  
 输入：.txt 格式的点云特征数据文件（通常由 step2_data_process_randla.py 输出产生）  
 输出：  
 在指定输出目录中生成新文件：文件名为原文件名加后缀 _gauss5%.txt  
 数据包含原始点云 + 5% 高斯噪声点  
 总结构为：x，y，z，r，g，b，label。原始点label保留，噪声点标签为1）    
 数据状态：  
 1）原始点云 + 结构相关的噪声点  
 2）噪声点按点云平均尺度（均值半径）和标准差添加  
 操作：  
   - a）读取原始 .txt 点云数据  
   - b）统计其三维空间中心及点云扩展尺度（均值和标准差）  
   - c）按 5% 数量，得出要添加的噪声点的数量，并为每个噪声点添加高斯噪声  
   - d）为每个噪声点复制原颜色信息，并将标签置为 1，表示添加了高斯噪声  
   - e）将原始数据和新噪声点合并，保存为新 .txt 文件（同目录、不同后缀）




## 2. txts_guassian_noise_limited.py：  
 输入：与 txt_guassian_noise.py的输入相同。   
 输出：  
 在指定输出目录中生成新文件：文件名为原文件名加后缀 _gauss5%_0.1m.txt  
 数据包含原始点云 + 加入的高斯噪声点  
 所有噪声点坐标被限制在原点云的XYZ范围内  
 数据状态：  
 原始点云 + 控制在空间边界内的噪声点（以 noise_limit=0.1m 为最大扰动）  
 模拟设备稳定性较高的干扰或环境中微小扰动  
 操作：  
   - a）读取原始 .txt 点云数据  
   - b）统计其XYZ轴的最小值与最大值  
   - c）按5%比例采样原始点，因为要加入原点数5%的噪声点，对固定数量采样出来的原始点加入均值为0、标准差为0.1的高斯扰动  
   - d）对扰动后坐标做边界裁剪，确保不超过原始点云边界  
   - e）为每个噪声点复制原颜色信息，并将标签置为 1  
   - f）将原始数据与新噪声点合并并保存为新 .txt 文件

 >txt_guassian_noise_limited.py与txt_guassian_noise.py的区别：  
   - txt_guassian_noise.py：  
   1）噪声幅度：根据点云整体尺度（平均半径）自动计算。  
   2）噪声位置限制：不限制坐标范围，可能超出原始点云边界。  
   3）噪声方向：随机单位向量方向 + 距离扰动（结构敏感）。  
   4）标签：添加的点统一为 1。  
   5）控制程度：更随机、更依赖点云形态。  
   - txt_guassian_noise_limited.py：  
   1）噪声幅度：固定噪声强度，使用 noise_limit 参数控制（如 0.1m）。  
   2）噪声位置限制：限制噪声点坐标在 XYZ 最大/最小值之间。  
   3）噪声方向：随机方向（各轴独立），无结构考虑。  
   4）标签：相同，统一为 1。  
   5）控制程度：更可控、更适用于封闭环境。  

# 三. Noise_Analysis  
## （1）KNN_Noise_points.py  
  
 - KNN_Noise_points:  
 作用：基于 K 近邻搜索（KNN）从原始点云中识别出与滤波处理后点云差异较大的点作为噪声点，并将这些噪声点提取保存为单独的 PLY 文件。  
 输入：  
 1）original_cloud：原始点云数据，格式为 .ply。  
 2）filtered_cloud：滤波后（即去噪处理过的）点云数据，格式也为 .ply。  
 输出：  
 noise_points.ply（保存识别出的噪声点的点云文件。每个点的结构通常包含 x, y, z 三个坐标值。）  
 数据状态：  
 1）原始点云：含有未处理噪声的完整点云数据  
 2）滤波后点云：通过滤波算法（如半径滤波）消除部分噪声点后的点云  
 3）输出的噪声点：被 KNN 判断为“异常”（与滤波点云差异大）的原始点  
 操作：  
   - a）使用 PlyData.read 加载原始和滤波后的 .ply 文件为点云对象（PCL）  
   - b）将读取的点云转换为 pcl.PointCloud 格式，便于后续操作  
   - c）创建 k-d tree（K 近邻搜索树），对原始点云进行结构化以便查找最近邻 
   - d）对滤波后的点云中的每个点，执行 K 近邻搜索，找出它在原始点云中的邻近点  
   - e）比较原始点和邻近点的均值差异，如果超过阈值 threshold=0.01，判定为噪声点  
   - f）将噪声点保存到 noise_points.ply 输出文件中  

  >判断点云中一个点是否为“异常点”/“噪声点”：  
  - 常见的点云去噪思想：真实点云中的点与其邻居的空间关系应该是一致或连续的；而噪声点通常 孤立、偏离其邻居。KNN（K-Nearest Neighbor）算法在这里用来度量“一个点与周围点之间的相似性”。  
  - 判断的具体操作：  
    a）对点云中每个原始点，取出该点的xyz坐标  
    b）在滤波后的点云 filtered_cloud 中找到与它最接近的 k 个点（用 KD-Tree 查找）  
    c）计算“均值差异”：原始点mean(x, y, z)，邻居点集mean(mean(x1), mean(y1), mean(z1), ..., mean(xk))；然后做均值之间的绝对差值比较np.abs(np.mean(original_point) - np.mean(neighbors)) > threshold；如果差值大于 threshold=0.01，就说明这个点“与周围不一致”，可能是噪声  
  - 噪声点通常的特征：  
    离散、稀疏，与周围结构差异大。比如一堆密集的点突然出现一个远离它们的孤立点，很可能是传感器误差、反射干扰等导致的异常采样。  
  - 均值是衡量“集中趋势”的指标：  
    真实点云中的局部区域的点通常是连贯的，其空间位置均值接近。若某点与其邻域均值偏差很大，它在几何结构上就是“突兀的”。  
  - 为什么用 0.01 作为阈值：  
    这个值是经验设置的，表示“点之间在空间上的平均允许偏移量”。如果你的点云单位是 米，那么 0.01 就是 1 厘米。超出这个就可能是“异常值”。  

# 四. Calculate     
## (1) txt_avg_columns(selected).py:
- **输入**：  
  - 任意文本文件，每行包含若干数值字段（如 `/path/to/input.txt`）
  - 用户指定要参与均值计算的列索引（如 0,1,2）

- **输出**：  
  - 新文本文件，每行末尾添加所选列的均值（如 `/path/to/output.txt`）

- **数据状态**：  
  - 原始数据 + 均值结果（每行新增一列）

- **操作**：  
  1. 读取输入文件
  2. 按用户指定列索引提取数值
  3. 计算均值并添加到每行末尾
  4. 写入新文件

## (2) mIoU_OA.py:
- **输入**：  
  - 一个文件夹，包含成对的 `_label.txt`（真实标签）和 `_pred.txt`（预测标签）文件

- **输出**：  
  - 评估结果文本文件（如 `result.txt`），内容包括 mIoU、OA、各类别 IoU

- **数据状态**：  
  - 每行一个标签，统计混淆矩阵，输出整体评估指标

- **操作**：  
  1. 遍历文件夹，读取所有标签/预测文件
  2. 计算混淆矩阵
  3. 统计各类 IoU、mIoU、OA
  4. 写入结果文件

## (3)filtered_labels_number.py:
- **输入**：  
  - 一个目录下的所有 `.txt` 点云数据文件，每行最后一列为标签

- **输出**：  
  - 标签统计结果，支持 `.txt` 或 `.csv` 格式（如 `label_count_result.csv`）

- **数据状态**：  
  - 原始点云标签分布统计

- **操作**：  
  1. 遍历所有文件，统计每个标签出现次数
  2. 输出每个文件及总体标签分布
  3. 保存统计结果为文本或 CSV

## (4)filter_evaluation_tool.py:
- **输入**：  
  - 用户手动输入四个点数：原始非噪声点 On、原始噪声点 Ot、滤波后保留的非噪声点 Rn、滤波后保留的噪声点 Rt

- **输出**：  
  - 评估结果（可选保存为文本文件）

- **数据状态**：  
  - 点数统计与各类滤波性能指标

- **操作**：  
  1. 输入参数
  2. 计算 IOU0、IOU1、Qn、Acc、Eb 等指标
  3. 显示并可保存结果

## (5)draw_MultiState_x(frequency)_y(value).py:
- **输入**：  
  - 数据文件（如 `data.txt`），包含多个状态（如 Ef=0.25 eV），每个状态下有频率-数值对

- **输出**：  
  - 绘制并保存频率曲线图（如 `output_images/data_processed.png`）

- **数据状态**：  
  - 多状态频率-数值曲线

- **操作**：  
  1. 读取数据文件，解析各状态数据
  2. 用户自定义绘图参数
  3. 绘制所有状态曲线并保存图片

## (6) Confusion_matrix_visualization.py:
- **输入**：  
  - 混淆矩阵数据（支持文件读取或手动输入），可为 `.txt` 或 `.csv` 文件

- **输出**：  
  - 混淆矩阵可视化图片（如 `*_matrix.png`），支持批量处理

- **数据状态**：  
  - 混淆矩阵数值，类别标签

- **操作**：  
  1. 读取或手动输入混淆矩阵
  2. 用户自定义类别标签、配色方案
  3. 绘制并保存混淆矩阵图像


# 五.cpp_wrappers  
## (1)cpp_subsampling:
- **作用**：  
  实现点云的网格化降采样（grid subsampling），并通过 C++/Python 封装为 Python 可调用模块。
- **主要内容**：  
  - `grid_subsampling/`：核心 C++ 降采样算法实现（.cpp/.h）
  - `wrapper.cpp`：C++ 封装接口，暴露 compute 方法给 Python
  - `setup.py`：编译配置脚本，生成 `.so` 动态库
  - `grid_subsampling.cpython-39-xxx.so`：编译生成的 Python 扩展模块
  - `__init__.py`：Python 包初始化文件
- **典型用途**：  
  在 Python 中通过 `from cpp_wrappers.cpp_subsampling import grid_subsampling` 调用高效的 C++ 点云降采样功能。

## (2)cpp_utils:
- **作用**：  
  提供点云处理的基础 C++ 工具函数和第三方库支持。
- **主要内容**：  
  - `cloud/`：点云数据结构与操作（如 cloud.cpp/cloud.h）
  - `nanoflann/`：KD-Tree 快速最近邻查找库（nanoflann.hpp，header-only）
  - `__init__.py`：Python 包初始化文件
- **典型用途**：  
  被 cpp_subsampling、nearest_neighbors 等模块引用，实现点云数据结构、空间索引等底层功能。

## (3)`__pycache__`
- **作用**：  
  存放 Python 解释器自动生成的字节码缓存文件（.pyc），加速模块加载。
- **主要内容**：  
  - `__init__.cpython-39.pyc` 等
- **典型用途**：  
  Python 包的运行时优化，无需手动管理。

总结：  
- `cpp_subsampling`：点云降采样算法及 Python 封装
- `cpp_utils`：点云基础工具与 KD-Tree 支持
- `__pycache__`：Python 字节码缓存  


# 六.Data_process 文件夹说明

## 1. txt_to_ply_with_rgb_label.py

- **输入**：  
  - .txt 点云数据文件，格式为 x, y, z, r, g, b, label

- **输出**：  
  - .ply 格式点云文件，包含 RGB 颜色和标签信息

- **数据状态**：  
  - 原始点云数据（文本） → 可视化友好的 PLY 格式（带颜色和标签）

- **操作**：  
  1. 读取 .txt 文件，拆分坐标、颜色、标签
  2. 合并为一行七列
  3. 写入 .ply 文件并插入头部信息

---

## 2. txt_to_npy.py

- **输入**：  
  - 一个文件夹下所有 .txt 点云数据文件

- **输出**：  
  - 对应的 .npy 文件，保存到指定目录

- **数据状态**：  
  - 文本格式点云数据 → NumPy 二进制格式（加速后续加载）

- **操作**：  
  1. 批量读取 .txt 文件
  2. 转换为 numpy 数组
  3. 保存为 .npy 文件

---

## 3. npy_to_txt.py

- **输入**：  
  - 一个文件夹下所有 .npy 文件（模型输出或特征）

- **输出**：  
  - 对应的 .txt 文件，保存到指定目录

- **数据状态**：  
  - 二进制 .npy → 文本 .txt（便于查看和后处理）

- **操作**：  
  1. 批量读取 .npy 文件
  2. 保存为 .txt 文件

---

## 4. npy_to_label.py

- **输入**：  
  - 一个文件夹下所有 .npy 文件（通常为预测标签）

- **输出**：  
  - 对应的 .label 文件（每行一个整数标签）

- **数据状态**：  
  - 浮点型标签 → 整数型标准标签文件

- **操作**：  
  1. 读取 .npy 文件
  2. 转换为整数
  3. 保存为 .label 文件

---

## 5. txt_rgb_(float_to_int).py

- **输入**：  
  - 一个文件夹下所有 .txt 文件（RGB为浮点数）

- **输出**：  
  - 处理后的 .txt 文件（RGB变为整数）

- **数据状态**：  
  - RGB颜色值从 float → int，标准化为 [0, 255] 整数

- **操作**：  
  1. 读取每行数据
  2. 将第4-6列（RGB）取整
  3. 保存到新文件

- **对比说明**：  
  - 转换前：RGB为浮点数  
  - 转换后：RGB为整数，更适合可视化和模型输入

---

## 6. txt_insert_column.py

- **输入**：  
  - 一个文件夹下所有 .txt 文件（格式为 x y z r g b）

- **输出**：  
  - 处理后的 .txt 文件（格式变为 x y z 0 r g b）

- **数据状态**：  
  - 插入一列全为0的伪强度值，便于模型输入格式统一

- **操作**：  
  1. 读取原始数据
  2. 在第3和第4列之间插入一列0
  3. 保存到新文件夹

---

## 7. txt_divide_to_xyzrgb_labels.py

- **输入**：  
  - 一个文件夹下所有 .txt 文件（格式为 x y z r g b label）

- **输出**：  
  - xyzrgb文件（前六列），.labels文件（最后一列）

- **数据状态**：  
  - 点云数据与标签分离，便于语义分割模型处理

- **操作**：  
  1. 读取数据
  2. 分离前六列和最后一列
  3. 分别保存为 .txt 和 .labels 文件

---

## 8. remove_txt_commas.py

- **输入**：  
  - 单个 .txt 文件（每行末尾有逗号）

- **输出**：  
  - 新的 .txt 文件（去掉每行末尾逗号）

- **数据状态**：  
  - 清理格式，便于后续处理

- **操作**：  
  1. 读取原文件
  2. 替换每行末尾的逗号为换行
  3. 保存新文件

- **对比说明**：  
  - 转换前：每行以逗号结尾  
  - 转换后：每行无逗号结尾

---

## 9. ply_to_txt.py

- **输入**：  
  - 一个文件夹下所有 .ply 文件（点云）

- **输出**：  
  - 对应的 .txt 文件（点云数据）

- **数据状态**：  
  - PLY格式 → TXT格式，便于文本处理和模型输入

- **操作**：  
  1. 读取 .ply 文件（支持二进制/ASCII）
  2. 提取点数据
  3. 保存为 .txt 文件

---

## 10. combine_txt_txt_line.py

- **输入**：  
  - 两个 .txt 文件（逐行对应）

- **输出**：  
  - 合并后的 .txt 文件（每行拼接）

- **数据状态**：  
  - 两文件内容逐行合并为一行

- **操作**：  
  1. 逐行读取两个文件
  2. 拼接每行内容
  3. 写入新文件

- **对比说明**：  
  - 合并方式：逐行 zip 拼接  
  - 与 combine_txt_txt_file.py 区别：后者是整文件拼接

---

## 11. combine_txt_txt_file.py

- **输入**：  
  - 两个文件夹下同名 .txt 文件

- **输出**：  
  - 合并后的 .txt 文件（内容顺序拼接，中间加换行）

- **数据状态**：  
  - 两文件内容整体合并

- **操作**：  
  1. 查找同名文件
  2. 拼接内容（文件A内容 + 换行 + 文件B内容）
  3. 保存到新目录

- **对比说明**：  
  - 合并方式：整文件拼接  
  - 与 combine_txt_txt_line.py 区别：前者是逐行，后者是整文件

---

## 12. combine_txt_labels.py

- **输入**：  
  - 一个文件夹下的 .txt 文件和对应 .labels 文件

- **输出**：  
  - 合并后的 _merged.txt 文件（每行拼接标签）

- **数据状态**：  
  - 点云数据与标签合并为一行

- **操作**：  
  1. 配对同名 .txt 和 .labels 文件
  2. 逐行拼接内容
  3. 保存到目标文件夹

- **对比说明**：  
  - 合并前：点云和标签分开  
  - 合并后：每行包含点云数据和标签

---

## 13. combine_txt_labels_single.py

- **输入**：  
  - 单个 .txt 文件和单个 .label 文件

- **输出**：  
  - 合并后的 _merged.txt 文件

- **数据状态**：  
  - 点云数据与标签合并为一行

- **操作**：  
  1. 校验文件名一致性
  2. 逐行拼接内容
  3. 保存新文件

- **对比说明**：  
  - 与 combine_txt_labels.py 区别：前者批量处理，后者单文件处理

---

## 14. combine_ply_ply(xyzlabel).py

- **输入**：  
  - 两个 .ply 文件（地面点和预测点，支持 x y z label 或 x y z r g b label）

- **输出**：  
  - 合并后的 .ply 文件（x y z label）

- **数据状态**：  
  - 两组点云数据合并为一个文件，统一结构

- **操作**：  
  1. 读取两个 .ply 文件，自动识别结构
  2. 提取 xyz 和 label
  3. 合并并保存为新 .ply 文件

- **对比说明**：  
  - 合并前：两个独立的点云文件  
  - 合并后：一个文件包含所有点及标签，便于可视化和后处理

---

# 七.DBSCAN 文件夹说明

## 1. csv_DBscan_KMeans_gps.py

- **输入**：  
  - 无表头的 .csv 文件，每行包含两个值：经度（lon）、纬度（lat）

- **输出**：  
  - 控制台打印每个聚类簇的中心点坐标（经度、纬度）
  - 聚类中心保存为 .csv 文件（如 `xxx_clustered.csv`）

- **数据状态**：  
  - 原始 GPS 坐标数据 → 聚类簇分组 → 每簇中心点坐标

- **操作**：  
  1. 读取所有 .csv 文件，提取经纬度
  2. 使用 DBSCAN（haversine 距离）进行聚类
  3. 对每个聚类簇用 KMeans 提取中心点
  4. 打印聚类数和中心点坐标
  5. 保存聚类中心为 .csv 文件

- **对比说明**：  
  - 适用于地理坐标聚类分析（如车辆、节点聚合）
  - 只输出中心点，不做图像可视化
  - 不做数据标准化（经纬度不能缩放）

---

## 2. picture_DBscan_leabel.py

- **输入**：  
  - 有表头的 .csv 文件，包含指标字段（如 VAR00003、VAR00004）

- **输出**：  
  - 聚类结果 .csv 文件（如 `xxx_dbscan_epsX_minY.csv`，每行带聚类标签）
  - 聚类可视化图像 .png 文件（如 `xxx_dbscan_plot.png`）

- **数据状态**：  
  - 原始指标数据 → 标准化 → 聚类分组 → 可视化

- **操作**：  
  1. 读取 .csv 文件，提取指定指标字段
  2. 标准化数据（preprocessing.scale）
  3. 使用 DBSCAN（欧氏距离）进行聚类
  4. 保存聚类结果为 .csv 文件
  5. 绘制聚类分布图并保存为 .png

- **对比说明**：  
  - 适用于普通数值指标聚类分析（如人口统计、行为分类）
  - 输出聚类分组和可视化图像
  - 支持批量处理和单文件处理
  - 不提取聚类中心，只分簇

---

## 3. 两脚本对比

| 项目                      | `csv_DBscan_KMeans_gps.py`           | `picture_DBscan_leabel.py`               |
| ------------------------- | ------------------------------------- | ---------------------------------------- |
| 📌 聚类对象               | GPS 经纬度（lon, lat）                | 普通数值指标（如 VAR00003, VAR00004）    |
| 📂 输入文件格式           | 无表头 CSV，两列坐标                  | 有表头 CSV，包含指标字段                 |
| 📊 距离度量               | haversine（球面距离）                 | 欧氏距离                                 |
| 🔍 目标任务               | 获取每簇中心点坐标                    | 获取每簇分组情况并可视化                 |
| 🧠 是否提取聚类中心       | ✅ 使用 KMeans                        | ❌ 不做中心点提取                        |
| 📈 是否生成图像可视化     | ❌ 无图像输出                         | ✅ 生成聚类分布图                        |
| 📁 是否批处理             | ✅ 批量处理所有 .csv 文件              | ✅ 支持批量和单文件处理                   |
| 📄 输出文件               | `xxx_clustered.csv`                   | `xxx_dbscan_epsX_minY.csv`、`.png` 图像  |
| ⚙️ 参数交互方式           | 控制台输入 eps_km、min_samples        | 控制台输入 eps、min_samples              |
| 🧪 标准化处理             | ❌ 不标准化                           | ✅ 使用标准化                            |
| 🧭 应用场景               | 地理坐标聚类分析                      | 指标评估型聚类分析                       |

---

# 八.downsample_test 文件夹说明

## 1. ply_random_downsampling_plus.py

- **输入**：  
  - 单个或批量 .ply 点云文件（支持 x, y, z, r, g, b, label 字段）

- **输出**：  
  - 下采样后的 .ply 文件（可选 .txt 文件，包含采样点的坐标、颜色、标签）

- **数据状态**：  
  - 原始点云 → 下采样点云（可选多轮采样、中心邻域采样、FPS采样、PCL采样）

- **操作**：  
  1. 读取 .ply 文件，提取点、颜色、标签
  2. 根据用户选择执行比例采样、固定数量采样、PCL采样、FPS采样或多轮采样
  3. 可选中心点邻域提取 + FPS采样
  4. 输出采样结果为 .ply 和/或 .txt 文件

- **对比说明**：  
  - 支持多种采样方式（随机、FPS、PCL、组合）
  - 可输出带标签的 .txt 文件，便于后续训练或分析

---

## 2. ply_fps.py

- **输入**：  
  - 批量 .ply 文件（仅 x, y, z 或带颜色）

- **输出**：  
  - 下采样后的 .ply 文件（点数减半，均匀分布）

- **数据状态**：  
  - 原始点云 → FPS采样点云

- **操作**：  
  1. 读取每个 .ply 文件，提取点坐标
  2. 使用 Farthest Point Sampling (FPS) 算法采样一半点
  3. 保存采样结果为新的 .ply 文件

- **对比说明**：  
  - 采样前：原始点云点数较多  
  - 采样后：点数减半，分布更均匀

---

## 3. ply(xyz)_to_pcd(xyz_intensity).py

- **输入**：  
  - .ply 文件（只读 x, y, z 坐标）

- **输出**：  
  - .pcd 文件（x, y, z, intensity 字段，intensity 固定为 0）

- **数据状态**：  
  - .ply 点云 → .pcd 点云（适配 PCL 工具）

- **操作**：  
  1. 使用 Open3D 读取 .ply 文件
  2. 提取点坐标
  3. 构造 .pcd 文件头，写入点坐标和 intensity 字段

- **对比说明**：  
  - 转换前：仅有坐标信息  
  - 转换后：增加 intensity 字段，便于兼容 PCL

---

## 4. ply_pcd(xyz).py

- **输入**：  
  - .ply 文件（只读 x, y, z 坐标）

- **输出**：  
  - .pcd 文件（x, y, z 字段）

- **数据状态**：  
  - .ply 点云 → .pcd 点云（无 intensity）

- **操作**：  
  1. 使用 plyfile 读取 .ply 文件
  2. 提取 x, y, z 坐标
  3. 构造标准 .pcd 文件头，写入点数据

- **对比说明**：  
  - 与 ply(xyz)_to_pcd(xyz_intensity).py 区别：无 intensity 字段

---

## 5. main.py

- **输入**：  
  - .ply 文件（包含 x, y, z, red, green, blue, label 字段）

- **输出**：  
  - 下采样后的 .ply 文件（如 256 点）

- **数据状态**：  
  - 原始点云 → 局部区域提取 → FPS采样点云

- **操作**：  
  1. 选取中心点，提取邻域点（如 65536 个）
  2. 对邻域点执行 FPS 下采样（如 256 个点）
  3. 保存采样结果为新的 .ply 文件

- **对比说明**：  
  - 两步采样：先邻域提取，再均匀采样
  - 输出点云包含坐标、颜色、标签

---

## 6. help_ply.py

- **输入**：  
  - .ply 文件（支持点云或三角网格）

- **输出**：  
  - 读取：返回点云数据（numpy 数组，字段齐全）
  - 写入：保存为 .ply 文件（支持自定义字段）

- **数据状态**：  
  - 点云数据结构化、标准化

- **操作**：  
  1. 解析 .ply 文件头，读取点数据
  2. 支持二进制和 ASCII 格式
  3. 写入 .ply 文件，支持自定义字段和三角面

- **对比说明**：  
  - 与 open3d/pandas等库相比，支持更灵活字段和格式

---

## 7. las/read_las.py & las/write_las.py

- **输入**：  
  - read_las.py：.las 文件（激光点云，含坐标、颜色等）
  - write_las.py：numpy 数组（x, y, z, r, g, b）

- **输出**：  
  - read_las.py：打印点云属性、坐标、颜色
  - write_las.py：保存为 .las 文件

- **数据状态**：  
  - .las 格式 ↔ numpy 数组

- **操作**：  
  1. 读取 .las 文件，提取属性字段
  2. 写入 .las 文件，指定坐标和颜色

- **对比说明**：  
  - 适用于激光点云格式与 numpy 数据互转

---

## 8. sampling/tf_sampling.cpp, tf_sampling.py, tf_sampling_so.so

- **输入**：  
  - 点云数据（numpy 数组，batch、点数、维度）

- **输出**：  
  - 采样点索引（如 FPS、GatherPoint、ProbSample）

- **数据状态**：  
  - 原始点云 → 采样点索引

- **操作**：  
  1. 编译 CUDA/C++ 采样算子
  2. 在 Python/TensorFlow 中调用高效采样（如 FPS）

- **对比说明**：  
  - 与纯 Python 实现相比，速度更快，支持大规模点云

---

## 9. nearest_neighbors/knn.cpp, nanoflann.hpp, KDTreeTableAdaptor.h, knn.pyx

- **输入**：  
  - 点云数据（numpy 数组）

- **输出**：  
  - 最近邻索引、距离等

- **数据状态**：  
  - 原始点云 → 最近邻结构（KD-Tree）

- **操作**：  
  1. 构建 KD-Tree，支持高维空间最近邻搜索
  2. 批量查询 KNN，返回索引和距离
  3. 支持 C++/Python/Cython 混合调用

- **对比说明**：  
  - 与 sklearn.neighbors.KDTree 区别：更高效，支持自定义距离和批量处理

---

## 10. 采样方法对比说明

| 方法                | 采样对象         | 采样方式         | 输出格式         | 适用场景           |
|---------------------|------------------|------------------|------------------|--------------------|
| 随机采样            | .ply/.pcd 点云   | 比例/数量随机抽取 | .ply/.txt/.pcd   | 快速预处理         |
| FPS采样             | .ply/.pcd 点云   | 最远点均匀采样   | .ply/.txt/.pcd   | 保证空间均匀性     |
| PCL RandomSample    | .pcd 点云        | 固定数量随机抽取 | .ply/.pcd        | PCL工具链兼容      |
| 中心邻域+FPS采样    | .ply 点云        | 局部区域+均匀采样| .ply/.txt        | 局部分析/降采样    |
| 格式转换            | .ply ↔ .pcd/.txt | 字段兼容转换     | .pcd/.txt/.ply   | 跨工具/格


# 九.nearest_neighbors 文件夹说明

## 1. knn.pyx / knn_.cxx / knn_.h

- **功能**：  
  实现高效的 K 近邻（KNN）搜索，支持单批和多批点云数据的最近邻查找，底层调用 C++/nanoflann KD-Tree。

- **主要接口**：  
  - `knn(pts, queries, K, omp=False)`：单批点云 KNN 查询，支持并行
  - `knn_batch(pts, queries, K, omp=False)`：多批点云 KNN 查询
  - `knn_batch_distance_pick(...)`：批量采样与距离筛选

- **数据状态**：  
  - 输入：numpy 数组格式点云（shape: [N, 3] 或 [B, N, 3]）
  - 输出：每个查询点的 K 个最近邻索引

- **操作流程**：  
  1. 构建 KD-Tree（KDTreeTableAdaptor）
  2. 查询最近邻（findNeighbors）
  3. 返回索引结果，支持 OpenMP 并行加速

---

## 2. nanoflann.hpp / KDTreeTableAdaptor.h

- **功能**：  
  nanoflann.hpp：C++ header-only KD-Tree 最近邻库  
  KDTreeTableAdaptor.h：适配 numpy/float 数组为 nanoflann KD-Tree输入，支持高效点云索引和查询

- **数据状态**：  
  - 输入：C++ float 数组或 numpy 数组
  - 输出：KD-Tree 索引结构，支持 KNN/半径搜索

- **操作流程**：  
  1. 构建 KD-Tree 索引
  2. 提供 kdtree_get_pt、kdtree_get_point_count 等接口
  3. 支持多维点云数据的高效最近邻查找

---

## 3. setup.py

- **功能**：  
  编译 Cython/C++ KNN 模块，生成 Python 可调用的最近邻扩展（nearest_neighbors.so）

- **操作流程**：  
  1. 配置编译参数（C++14，O3优化）
  2. cythonize 源文件 knn.pyx、knn_.cxx
  3. 生成 Python 扩展模块

---

## 4. test.py

- **功能**：  
  测试 nearest_neighbors 模块的 KNN 查询性能和接口

- **操作流程**：  
  1. 随机生成点云数据
  2. 调用 knn_batch 查询最近邻
  3. 打印运行时间

---

## 5. lib/ 及 build/ 目录

- **功能**：  
  存放编译生成的 Python 包、动态库和缓存文件

- **主要内容**：  
  - `nearest_neighbors.cpython-39-x86_64-linux-gnu.so`：编译后的 Python 扩展模块
  - `__pycache__`：Python 字节码缓存
  - `KNN_NanoFLANN-0.0.0-py3.7.egg-info`：包元数据

---

## 6. __init__.py

- **功能**：  
  Python 包初始化文件，便于模块导入

---

## 7. 主要功能总结

- 提供高效的点云 K 近邻（KNN）查询能力，支持批量、并行、距离筛选等多种模式
- 底层采用 nanoflann KD-Tree，C++/Cython 混合实现，兼容 numpy 数据
- 适用于大规模点云的空间索引、特征提取、邻域分析等任务  

# 十.upsample 文件夹说明

## 1. txt_random_upsampling_plus.py

- **输入**：  
  - .txt 文件，每行格式为 x y z R G B label（共 7 列），通常来自下采样结果

- **输出**：  
  - 上采样后的 .txt 文件（结构不变，点数增多，指定标签点被复制或生成新点）

- **数据状态**：  
  - 原始点云 → 上采样点云（指定标签点数量增加）

- **操作**：  
  1. 策略1：边界框随机采样  
     - 在标签点的空间边界框内随机生成新点，颜色可选平均或随机，坐标可加扰动
  2. 策略2：标签点复制+扰动  
     - 随机复制标签点并对坐标加小扰动，颜色保持原值
  3. 策略3（已注释）：高斯扰动采样  
     - 按标签点的均值和协方差采样新点，颜色随机选自原标签点
  4. 支持批量处理，自动识别标签号，参数可交互输入

- **对比说明**：  
  - 边界框采样：新点分布均匀，覆盖空间范围，颜色可统一或多样
  - 复制+扰动：新点密集分布于原点附近，结构保持
  - 高斯扰动：新点服从整体分布，适合形态增强

---

## 2. make_ply(xyz_rgb_label).py

- **输入**：  
  - xyzs（点坐标数组），labels（标签数组），或含 label 字段的 .ply 文件

- **输出**：  
  - 带 RGB 和 label 的 .ply 文件，可视化点云

- **数据状态**：  
  - 点云坐标+标签 → .ply 文件（带颜色和标签）

- **操作**：  
  1. 根据标签映射表为每个点赋 RGB 颜色
  2. 写入标准 .ply 文件头和点数据
  3. 支持从已有 .ply 文件或随机生成点云
  4. 可选可视化输出

---

## 3. color_to_gray.py

- **输入**：  
  - 图像文件（.png/.jpg），每个像素为 RGB 颜色标签图

- **输出**：  
  - 灰度标签图（每个像素变为 (n, n, n)，n为类别编号），文件名加 _labeled

- **数据状态**：  
  - 原始彩色标签图 → 灰度标签图（语义类别编码）

- **操作**：  
  1. 遍历输入目录所有图像
  2. 按预设颜色表将指定颜色替换为类别编号
  3. 保存处理后图像到输出目录

- **对比说明**：  
  - 转换前：彩色标签图，颜色区分类别
  - 转换后：灰度标签图，像素值直接表示类别

---

## 4. pytorch_gpu_test.py

- **输入**：  
  - 无需外部输入，自动检测 GPU 环境

- **输出**：  
  - 控制台打印 GPU 信息、张量计算结果、性能测试循环次数

- **数据状态**：  
  - 检查 PyTorch GPU 可用性，执行张量运算

- **操作**：  
  1. 打印 PyTorch 版本、CUDA 可用性、GPU数量与名称
  2. 创建张量并在 GPU 上计算
  3. 循环扩大张量规模，测试 GPU 运算性能

---

## 5. utils/ply.py

- **输入**：  
  - .ply 文件（ASCII格式）

- **输出**：  
  - numpy 结构化数组（字段名与 .ply 文件一致）

- **数据状态**：  
  - .ply 点云 → numpy 数组（便于后续处理）

- **操作**：  
  1. 解析 .ply 文件头，提取字段名
  2. 读取点数据，按字段分组
  3. 返回结构化数组，支持 x, y, z, r, g, b, label 等字段

---

## 6. __init__.py

- **功能**：  
  Python 包初始化文件，便于模块导入

---

## 7. 主要功能对比

| 方法/脚本                  | 输入类型           | 输出类型           | 主要操作/特点                  | 适用场景           |
|----------------------------|--------------------|--------------------|-------------------------------|--------------------|
| txt_random_upsampling_plus | .txt 点云          | .txt 点云          | 多策略上采样，标签点增强       | 数据增强           |
| make_ply(xyz_rgb_label)    | xyzs+labels/.ply   | .ply 点云          | 按标签赋色，生成可视化点云     | 可视化/格式转换    |
| color_to_gray              | 彩色标签图像       | 灰度标签图像       | 颜色转类别编码，批量处理       | 语义分割/标注      |
| pytorch_gpu_test           | 无                 | 控制台输出         | GPU环境检测与性能测试          | 环境验证           |
| utils/ply.py               | .ply 文件          | numpy 数组         | 结构化读取，字段自动识别       | 点云数据


---


 
