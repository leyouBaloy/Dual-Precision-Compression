import numpy as np
from netCDF4 import Dataset
import libpysal as lps
from esda.moran import Moran

# 使用netCDF4打开数据文件和提取底层水温数据
dataset = Dataset('E:\MarineDatasets\HYCOM\hycom_GLBv0.08_530_1994010112_t006.nc', 'r')
water_temp_bottom_data = dataset.variables['water_temp_bottom'][0, ...]  # 假设时间维度只有一个值，去掉时间维度
dataset.close()

# 转换数据为numpy数组，以便处理
data_array = np.array(water_temp_bottom_data)

# 构建空间权重矩阵，这里需要根据实际情况选择合适的方法
# 假设我们使用一个简单的网格邻接矩阵
# 注意：这里的W需要根据实际的空间数据结构来定制
w = lps.weights.lat2W(3251, 4500)

# 计算莫兰指数
moran = Moran(data_array.flatten(), w)
print(f"莫兰指数: {moran.I}, 显著性水平p-value: {moran.p_sim}") # 莫兰指数: 0.9905585358686495, 显著性水平p-value: 0.001
