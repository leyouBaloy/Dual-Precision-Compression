import netCDF4 as nc
import os
import datetime
import numpy as np

base_path = "E:\MarineDatasets\iHESP_SST"
# 定义文件路径
file_path = os.path.join(base_path, 'B.E.13.B1850C5.ne120_t12.sehires38.003.sunway_02.pop.h.nday1.SST.046001-046912.nc')

# 使用netCDF4读取文件
with nc.Dataset(file_path, 'r') as dataset:
    # 获取时间变量
    time_var = dataset.variables['time'][0:40]  # 提取前40个时间点
    time_units = dataset.variables['time'].units  # 获取时间单位

    # 转换时间单位，假设时间单位是'days since 1850-01-01'
    # 这里我们需要知道起始日期和时间单位的具体含义
    # 由于我们没有具体的起始日期，我们将使用一个假设的起始日期
    # 例如，如果时间单位是从1850年1月1日开始的天数，则可以使用以下代码
    start_date = datetime.datetime(1, 1, 1)  # 假设的起始日期
    time_values = np.array([start_date + datetime.timedelta(days=t) for t in time_var])

    # 打印前40个时间点的日期
    print("Dates of the first 40 time points:")
    for date in time_values:
        print(date)
