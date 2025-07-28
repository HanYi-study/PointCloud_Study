# 点云代码学习
此文档用于记录点云代码学习笔记

点云处理简要流程：采集 → 清洗预处理 → 添加噪声/增强 → 特征提取 → 输入模型训练 → 后处理 → 评估 → 可视化


## 目录
 - 文件功能说明
 - 文件结构说明
 - 处理流程/逻辑展示



 
## 文件功能说明  
 **1. data-prepare文件夹:**  
 （1）step0_read_las.py  
 （2）step1_data_prepare_H3D.py  
 （3）step1_data_prepare_V3D.py  
 （4）step2_data_process_randla.py  
 （5）help_tool.py  
 （6）help_ply.py
 
 - step0_read_las.py：  
 输入：原始.las 格式点云数据（含空间坐标、颜色、分类标签等）   
 输出：内存中的numpy数组（x,y,z,RGB,label）、min_xyz起始坐标（后续可用于还原位置信息）  
数据状态：原始格式转换为标准数组  
 处理：  
    - a）解析.las文件结构      
    - b）提取xyz坐标（归一化到起点）、RGB颜色、标签（classification）

 - step1_data_prepare_H3D.py / step1_data_prepare_V3D.py：  
 两文件的区别：  
 H3D.py 处理的是 H3D 格式的点云数据（含 RGB、instance、label、可能还带 intensity）  
 V3D.py 针对的是 V3D 格式的点云数据（结构简洁，可能只有 XYZ 和 label）  
 输入：step0产出的数组数据  
 输出：.ply格式的点云文件、KD-Tree或其他中间结构缓存（如 .npy/.pkl 文件）  
 数据状态：已变为可视化或深度学习框架可读格式  
 操作：  
    - a）构建KD-Tree空间结构  
    - b）写出.ply文件：XYZ+RGB+Label  

 - step2_data_process_randla.py:  
 输入：.ply格式的点云文件  
 输出：是保存在output_dir目录下的两个文件，一个是处理后的点云特征数据文件，另一个是标签文件。  
 a）处理后的特征数据文件(同名.txt文件),把原数据中除了最后一列(label)以外的列都提取出来，在第4列插入值0，用于之后的模型输入、特征工程、数据增强等。  
 b）标签文件(同名.label文件),只保存每一个点的标签（最后一列0,通常用于分类监督学习模型的目标值。  
 数据状态：结构化、标准化、块化后的训练输入数据。  
 操作：  
    - a）对每个点云文件提取标签（最后一列）。  
    - b）修改点云特征（在每行的第四列插入0）。  
    - c）将处理后的数据和标签分别保存到输出目录中。


 **2. Noise_DataAugmentation**  
 （1）txt_guassian_noise.py  
 （2）txt_guassian_noise_limited.py


 - txt_guassian_noise.py：  
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




 - txt_guassian_noise_limited.py：  
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

**3. Noise_Analysis**  
（1）KNN_Noise_points.py  
  
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

 ## 文件结构说明  

目录结构详见data_structure.txt
 ## 处理流程/逻辑展示
 ### 1.创建虚拟环境
conda create -n PointCloud_02 python=3.9  
 ### 2.配置虚拟环境  
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

### 3.编译
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

### 4.预处理操作-->运行.py文件  
#### 运行step0_read_las.py  
>输入：/home/hy/projects/PointCloud_Code/Data/Data_prepare/3-4(3_4).las  
输出：/home/hy/projects/PointCloud_Code/Results/data_prepare_result/step0_result/3-4(3_4).txt  
（三个.las文件都分别执行了一次该.py文件，每次执行都更改对应的输入输出路径）

#### 运行step1_dara_prepare_H3D.py  
>输入：/home/hy/projects/PointCloud_Code/Results/data_prepare_result/step0_result  
输出：/home/hy/projects/PointCloud_Code/Results/data_prepare_result/step1_H3D_result  
（将step0的运行结果.txt文件作为这一步的输入数据）  
（最终在step1_H3D_result 中生成了两个文件夹：kdtree-1和points-1，并且这两个文件夹中，生成了三个数据集的对应格式文件，详情可见文件结构）

#### 运行Data_conversion文件夹下的ply_to_txt.py  
>输入：/home/hy/projects/PointCloud_Code/Results/data_prepare_result/step1_H3D_result/points-1  
（step1_data_prepare_H3D.py的运行结果中point-1文件夹中所有.ply文件）  
输出：/home/hy/projects/PointCloud_Code/Results/data_conversion_result/H3D_points-1_ply_to_txt  （经过格式转换，转换为txt文件，并存放再输出路径下）   
（文件转换后的txt文件作为step2_data_process_randla.py的输入）

#### 运行data_prepare文件夹下的step2_data_process_randla.py  
>输入：/home/hy/projects/PointCloud_Code/Results/data_conversion_result/H3D_points-1_ply_to_txt'  
（将上一步转换完生成的txt文件作为输入）  
输出：/home/hy/projects/PointCloud_Code/Results/data_prepare_result/step2_result  
（生成的结果有处理后的特征数据.txt文件、标签文件.label文件）  




