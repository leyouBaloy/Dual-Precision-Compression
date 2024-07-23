from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np

# 打开NetCDF文件
dataset = Dataset('E:\MarineDatasets\HYCOM\hycom_GLBv0.08_530_1994010112_t000.nc', 'r')

# 访问底层水温数据
water_temp_bottom_data = dataset.variables['water_temp_bottom'][0,...] # (3251, 4500)=>(3120, 4320)
# 为了适应训练，进行裁切
water_temp_bottom_data = water_temp_bottom_data[65:-66, 90:-90]

# 获取经度和纬度数据
lons = dataset.variables['lon'][...] # (4500,)=>(4320,)
lons = lons[90:-90]
lats = dataset.variables['lat'][...] # (3251,)=>(3120,)
lats = lats[65:-66]

# 创建一个新的figure
plt.figure(figsize=(15, 8))

# 绘制底层水温数据
plt.pcolormesh(lons, lats, water_temp_bottom_data, cmap='coolwarm', shading='auto')

# 添加颜色条
plt.colorbar(label='Temperature (°C)')

# 设置坐标轴标签
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# 设置图表标题
plt.title('Bottom Water Temperature')

# 显示图表
plt.show()
print("数据尺寸:",water_temp_bottom_data.shape) # (3120, 4320)
print("数据类型:",water_temp_bottom_data.dtype) # float32

# 关闭文件
dataset.close()