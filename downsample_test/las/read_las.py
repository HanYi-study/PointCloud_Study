import laspy
import numpy as np

las_file = r"/home/yunl/Data/paper_data/140-141.las"
las = laspy.read(las_file)
# 获取文件头
header = las.header
# 点类型
point_format = las.point_format
print(point_format.id)
# 属性字段名
dimension_names = point_format.dimension_names
print(list(dimension_names))

# 点集的外边框
print(header.mins)
print(header.maxs)

# 点个数
point_num = header.point_count

# 获取坐标和颜色
las_x = np.array(las.x)
las_y = np.array(las.y)
las_z = np.array(las.z)
las_r = np.array(las.red)
las_g = np.array(las.green)
las_b = np.array(las.blue)

# 组合
pt = np.stack([las_x,las_y,las_z],axis=1)
colors = np.stack([las_r,las_g,las_b],axis=1)

