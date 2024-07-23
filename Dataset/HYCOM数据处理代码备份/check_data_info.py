from netCDF4 import Dataset

# 打开NetCDF文件
dataset = Dataset('E:\MarineDatasets\HYCOM\hycom_GLBv0.08_530_1994010112_t000.nc', 'r')

# 打印文件的维度信息
print(dataset.dimensions)

# 打印文件的变量信息
print(dataset.variables)

# 访问底层水温数据
water_temp_bottom_data = dataset.variables['water_temp_bottom'][...]

# 打印底层水温数据
print("打印底层水温数据",water_temp_bottom_data.shape) # 打印底层水温数据 (1, 3251, 4500)
print("底层水温数据max:",water_temp_bottom_data.max()) # 32.699
print("底层水温数据min:",water_temp_bottom_data.min()) # -2.7640018
# 获取时间变量
time_var = dataset.variables['time']

# 打印时间变量的信息
print(time_var)

# 获取时间维度的长度（即时间点的数量）
time_length = len(time_var)

# 打印时间维度的长度
print(f'The length of the time dimension is: {time_length}')

# 打印时间维度的所有值
print(time_var[...])

# 关闭文件
dataset.close()