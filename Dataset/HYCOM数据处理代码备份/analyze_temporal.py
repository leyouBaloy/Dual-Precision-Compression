from netCDF4 import Dataset
import numpy as np
import os

# 读取原始数据
base_path = "E:\MarineDatasets\HYCOM"
# 文件路径列表
file_paths = ["hycom_GLBv0.08_530_1994010112_t000.nc",
    "hycom_GLBv0.08_530_1994010112_t003.nc",
    "hycom_GLBv0.08_530_1994010112_t006.nc",
    "hycom_GLBv0.08_530_1994010112_t009.nc",
    "hycom_GLBv0.08_530_1994010112_t012.nc",
    "hycom_GLBv0.08_530_1994010112_t015.nc",
    "hycom_GLBv0.08_530_1994010112_t018.nc",
    "hycom_GLBv0.08_530_1994010112_t021.nc",

    "hycom_GLBv0.08_530_1994010212_t000.nc",
    "hycom_GLBv0.08_530_1994010212_t003.nc",
    "hycom_GLBv0.08_530_1994010212_t006.nc",
    "hycom_GLBv0.08_530_1994010212_t009.nc",
    "hycom_GLBv0.08_530_1994010212_t012.nc",
    "hycom_GLBv0.08_530_1994010212_t015.nc",
    "hycom_GLBv0.08_530_1994010212_t018.nc",
    "hycom_GLBv0.08_530_1994010212_t021.nc",
    
    "hycom_GLBv0.08_530_1994010312_t000.nc",
    "hycom_GLBv0.08_530_1994010312_t003.nc",
    "hycom_GLBv0.08_530_1994010312_t006.nc",
    "hycom_GLBv0.08_530_1994010312_t009.nc",
    "hycom_GLBv0.08_530_1994010312_t012.nc",
    "hycom_GLBv0.08_530_1994010312_t015.nc",
    "hycom_GLBv0.08_530_1994010312_t018.nc",
    "hycom_GLBv0.08_530_1994010312_t021.nc",

    "hycom_GLBv0.08_530_1994010412_t000.nc",
    "hycom_GLBv0.08_530_1994010412_t003.nc",
    # "hycom_GLBv0.08_530_1994010412_t006.nc", # 这个官方没给
    "hycom_GLBv0.08_530_1994010412_t009.nc",
    "hycom_GLBv0.08_530_1994010412_t012.nc",
    "hycom_GLBv0.08_530_1994010412_t015.nc",
    "hycom_GLBv0.08_530_1994010412_t018.nc",
    "hycom_GLBv0.08_530_1994010412_t021.nc",

    "hycom_GLBv0.08_530_1994010512_t000.nc",
    "hycom_GLBv0.08_530_1994010512_t003.nc",
    "hycom_GLBv0.08_530_1994010512_t006.nc", 
    "hycom_GLBv0.08_530_1994010512_t009.nc",
    "hycom_GLBv0.08_530_1994010512_t012.nc",
    "hycom_GLBv0.08_530_1994010512_t015.nc",
    "hycom_GLBv0.08_530_1994010512_t018.nc",
    "hycom_GLBv0.08_530_1994010512_t021.nc",

    "hycom_GLBv0.08_530_1994010612_t000.nc",
]
water_temp_bottom_mean_list = []
# 遍历文件路径列表，读取每个文件的数据并添加到列表中
for file_path in file_paths:
    # 打开NetCDF文件
    dataset = Dataset(os.path.join(base_path, file_path), 'r')
    
    # 访问底层水温数据 (3251, 4500)=>(3120, 4320)
    water_temp_bottom_data = dataset.variables['water_temp_bottom'][...]  # 读取数据并移除掩码，掩码的值为-30000
    
    # 将数据添加到列表中
    water_temp_bottom_mean_list.append(water_temp_bottom_data.mean())
    
    # 关闭文件
    dataset.close()

print("water_temp_bottom_mean_list:",water_temp_bottom_mean_list)

import numpy as np
import matplotlib.pyplot as plt

# 假设 original_data 已经按照您提供的代码加载

# 绘制时间序列图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(water_temp_bottom_mean_list)+1), water_temp_bottom_mean_list, marker='o', linestyle='-', color='b')
plt.title('Average Bottom Water Temperature Over 1994.01.01-1994.01.06')
plt.xlabel('Time Point')
plt.ylabel('Average Temperature (°C)')
plt.xticks(range(1, len(water_temp_bottom_mean_list)+1))  # 假设时间点是连续的且等距的
plt.grid(True)
plt.savefig("E:\MarineDatasets\HYCOM\Average Bottom Water Temperature Over 1994.01.01-1994.01.06.jpg")
plt.show()

