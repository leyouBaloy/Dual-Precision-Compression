from netCDF4 import Dataset
import numpy as np
import os

# 定义一个空列表来存储每个文件的water_temp_bottom数据
water_temp_bottom_list = []

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
]

# 遍历文件路径列表，读取每个文件的数据并添加到列表中
for file_path in file_paths:
    # 打开NetCDF文件
    dataset = Dataset(os.path.join(base_path, file_path), 'r')
    
    # 访问底层水温数据 (3251, 4500)=>(3120, 4320)
    water_temp_bottom_data = dataset.variables['water_temp_bottom'][0, 65:-66, 90:-90].filled()  # 读取数据并移除掩码，掩码的值为-30000
    
    # 将数据添加到列表中
    water_temp_bottom_list.append(water_temp_bottom_data)
    
    # 关闭文件
    dataset.close()

# 将列表中的所有二维数组堆叠成一个新的三维数组
water_temp_bottom_3d = np.stack(water_temp_bottom_list, axis=0)

# 打印最终数据的维度
print("最终数据尺寸:", water_temp_bottom_3d.shape)  # 应为 (16, 3120, 4320)


# 如果需要保存为二进制文件（.bin），可以使用 tofile 方法
water_temp_bottom_3d.tofile('E:\MarineDatasets\HYCOM\water_temp_bottom_aggregate.bin')