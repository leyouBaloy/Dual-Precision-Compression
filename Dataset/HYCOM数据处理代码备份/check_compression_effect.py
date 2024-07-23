import numpy as np
import os

# 读取原始数据
original_data_path = 'E:/MarineDatasets/HYCOM/water_temp_bottom_aggregate.bin'
original_data = np.fromfile(original_data_path, dtype=np.float32).reshape((16, 3251, 4500))

# 压缩文件
compressed_file_path = "E:/MarineDatasets/HYCOM/compressed_bitstream.bin"

# 读取解压缩后的数据
decompressed_data_path = 'E:/MarineDatasets/HYCOM/decompressed_data.bin'
decompressed_data = np.fromfile(decompressed_data_path, dtype=np.float32).reshape((16, 3251, 4500))

data_range = 32.699 - (-2.7640018)

# 计算平均绝对误差 (MAE)
mae_error = np.mean(np.abs(original_data - decompressed_data))
print(f"平均绝对误差 (MAE): {mae_error:.6f}")

# 计算最大误差
max_error = np.max(np.abs(original_data - decompressed_data))
print(f"最大误差 (Max Error): {max_error:.6f}")

# 计算平均相对误差 (MRE)
mre_error = np.mean(np.abs((original_data - decompressed_data) / data_range))
print(f"平均相对误差 (REL_mean): {mre_error:.6f}")

# 计算最大相对误差 (MRE_max)
mre_max_error = np.max(np.abs((original_data - decompressed_data) / data_range))
print(f"最大相对误差 (REL_max): {mre_max_error:.6f}")

# 计算峰值信噪比 (PSNR)
max_value = np.max(original_data)
min_value = np.min(original_data)
signal_power = np.square(max_value - min_value)
noise_power = np.mean(np.square(original_data - decompressed_data))
psnr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
print(f"峰值信噪比 (PSNR): {psnr:.6f} dB")

# 计算压缩率 (CR)
original_data_size = os.path.getsize(original_data_path)
compressed_data_size = os.path.getsize(compressed_file_path)
compression_ratio = original_data_size / compressed_data_size
print(f"压缩率 (CR): {compression_ratio:.6f}")

# 计算比特率 (BR)
number_of_elements = original_data.size
bit_rate = compressed_data_size * 8 / number_of_elements  # 注意这里修改为乘以8，因为是bits per element
print(f"比特率 (BR): {bit_rate:.6f} bits per element")
