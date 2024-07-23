import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import sys
import torch.nn.functional as F
import torchac.torchac
import matplotlib.patches as patches
from Dataset.HYCOM import HYCOM, HYCOM_Train
from module import Model, LPModel, HPModel
import utils
import os
from matplotlib import pyplot as plt
import time

# 先测试重建误差
# load model
print("create model")
device = "cuda:0"
lpmodel = LPModel(num_hiddens=64, num_residual_layers=5, num_residual_hiddens=32,
                  num_embeddings=32, embedding_dim=32,
                  commitment_cost=0.25, decay=0.99).to(device)
hpmodel = HPModel(num_hiddens=64, num_residual_layers=5, num_residual_hiddens=32,
                  num_embeddings=32, embedding_dim=32,
                  commitment_cost=0.25, decay=0.99, n_component=2).to(device)
# lp_save_dict = torch.load("saves/comparison/HYCOM_SEL_lp.pth")
lp_save_dict = torch.load("saves/ablation/HYCOM_SEL_lt_partition/HYCOM_SEL_lt_lp_ga_part3.pth") # 后面把comparison改一下
hp_save_dict = torch.load("saves/ablation/HYCOM_SEL_lt_partition/HYCOM_SEL_lt_hp_ga_part3_ep120.pth")

print("lp model train epoch: ", lp_save_dict['epoch'])
print("hp model train epoch: ", hp_save_dict['epoch'])

lpmodel.load_state_dict(lp_save_dict['model'])
hpmodel.load_state_dict(hp_save_dict['model'])

print("create model completed")

# load data
print("loading data...")
dataset = HYCOM("/home/bailey/dataset/HYCOM/SEL_lt")

# Initialize lists for storing results
all_data_recon = []
all_real_res = []
all_test = []
all_mask = []
all_means = []
all_stds = []
all_weights = []

start_time = 196
time_len = 105
# time_len = 2

for index in range(start_time, start_time+time_len):
    # print(f"Processing index {index}...")
    data, mask = dataset.getitem(index=index, is_norm=True)
    test, mask = dataset.getitem(index=index, is_norm=False)
    
    data = data.unsqueeze(1).to(device)
    test = test.unsqueeze(1)
    mask = mask.unsqueeze(1)
    
    # eval
    lpmodel.eval()
    hpmodel.eval()
    with torch.no_grad():
        vq_loss, _data_recon, perplexity, quantized = lpmodel(data)
        means, stds, weights = hpmodel(quantized)
        data_recon = dataset.inverse_normalize(_data_recon) # (150, 1,)
        data_recon[data_recon > dataset.max_val] = dataset.max_val
        data_recon[data_recon < dataset.min_val] = dataset.min_val
        # data_recon_flat = dataset.reconstruct_from_blocks(data_recon.squeeze().cpu())
        
        real_res = data_recon.cpu() - test

        all_data_recon.append(data_recon)
        all_real_res.append(real_res)
        all_test.append(test)
        all_mask.append(mask)
        all_means.append(means)
        all_stds.append(stds)
        all_weights.append(weights)

print("eval finished.")

# Concatenate all results
all_data_recon = torch.cat(all_data_recon, dim=0)
all_real_res = torch.cat(all_real_res, dim=0)
all_test = torch.cat(all_test, dim=0)
all_mask = torch.cat(all_mask, dim=0)
all_means = torch.cat(all_means, dim=0)
all_stds = torch.cat(all_stds, dim=0)
all_weights = torch.cat(all_weights, dim=0)
print("all_data_recon.shape",all_data_recon.shape)
print("all_real_res.shape",all_real_res.shape)
print("all_test.shape",all_test.shape)
print("all_mask.shape",all_mask.shape)
print("all_means.shape",all_means.shape)


# quanti error
calc_res = all_real_res.clone()
calc_res[all_real_res > 1.0] = 1.0 # 只对[-1,1]区间的值进行概率建模和熵编码
calc_res[all_real_res < -1.0] = -1.0
res_quantized = utils.uniform_quantization(calc_res.cpu(), quan_num=1000, min_val=-1.0, max_val=1.0)
res_dequantized = utils.uniform_dequantization(res_quantized, quan_num=1000, min_val=-1.0, max_val=1.0)
res_abs_quantized_diff = torch.abs(res_dequantized.cpu() - calc_res)
quantized_mae = torch.mean(res_abs_quantized_diff).item()
quantized_max = torch.max(res_abs_quantized_diff).item()

print(f"量化平均绝对误差MAE: {quantized_mae:.6f}")
print(f"量化最大误差MAX: {quantized_max:.6f}")

print("======================")
all_data_recon_depad = all_data_recon.cpu()[...,8:-8,8:-8]
all_real_res_depad = all_real_res.cpu()[...,8:-8,8:-8]
all_test_depad = all_test[...,8:-8,8:-8]
all_mask_depad = all_mask[...,8:-8,8:-8]
all_test_depad = all_test[...,8:-8,8:-8]


mae = torch.mean(torch.abs(all_real_res_depad[all_mask_depad]))
max_error = torch.abs(all_real_res_depad[all_mask_depad]).max()

print(f"LP平均绝对误差(MAE): {mae:.6f}")
print(f"LP最大误差(MAX error): {max_error:.6f}")

# 计算相对误差
data_range = dataset.max_val - dataset.min_val
relative_error = torch.abs(all_real_res_depad[all_mask_depad] / data_range)
rel_mean = torch.mean(relative_error).item()
rel_max = torch.abs(all_real_res_depad[all_mask_depad]).max() / data_range

print(f"LP平均相对误差(REL): {rel_mean:.6f}")
print(f"LP最大相对误差(REL max): {rel_max:.6f}")

# 计算峰值信噪比 (PSNR)
mse = torch.mean(torch.square(all_real_res_depad[all_mask_depad]))
psnr = 10 * torch.log10(torch.square(data_range) / mse) if mse > 0 else float('inf')

print(f"LP峰值信噪比 (PSNR): {psnr:.6f} dB")
print("======================")
# print("all_means.shape",all_means.shape) # [3650, 256, 256, 2]
# all_means_depad = all_means[:, 8:-8, 8:-8, :]
# all_stds_depad = all_stds[:, 8:-8, 8:-8, :]
# all_weights_depad = all_weights[:, 8:-8, 8:-8, :]

mask_out1 = torch.zeros_like(all_mask, dtype=torch.bool)
mask_out1[all_mask & ((all_real_res > 1.0)|(all_real_res < -1.0))] = True # 有效数据，且在[-1,1]之外
# print(f"[-1,1]之外数据比例: {(torch.sum(mask_out1)/len(diff_np)).item():.6f}")

compressed_bits_lst = []
hp_datas = []
start_time = time.time()
for idx in range(all_means.shape[0]):
    compressed_bits,hp_data_one = utils.get_compressed_bits(res_quantized[idx,:,8:-8,8:-8,...].unsqueeze(0).cpu(),
                                                all_mask[idx,:,8:-8,8:-8, ...].unsqueeze(0).cpu(),
                                                mask_out1[idx,:,8:-8,8:-8, ...].unsqueeze(0).cpu(),
                                                all_means[idx,8:-8,8:-8, ...].unsqueeze(0).cpu(),
                                                all_stds[idx,8:-8,8:-8, ...].unsqueeze(0).cpu(),
                                                all_weights[idx,8:-8,8:-8, ...].unsqueeze(0).cpu(),
                                                quan_num=1000, HW_size=240)
    compressed_bits_lst.append(compressed_bits)
    hp_datas.append(hp_data_one.squeeze(0))
end_time = time.time()

print(f"耗时 {end_time-start_time}")

body_mask = (all_mask^mask_out1) # 异或操作
body_mask_depad = body_mask[:,:,8:-8,8:-8].squeeze().unsqueeze(1)
res_dequantized_depad = res_dequantized[:,:,8:-8,8:-8].squeeze().unsqueeze(1)

# 模拟出重建数据
hp_res_data = torch.stack(hp_datas).squeeze().unsqueeze(1) # (234, 1, 240, 240) 不包含mask_out1
print("hp_res_data.shape",hp_res_data.shape)
# 把[-1,1]之外的数据替换为real_res
mask_out1_depad = mask_out1[:,:,8:-8,8:-8].squeeze().unsqueeze(1)
all_real_res_depad = all_real_res[:,:,8:-8,8:-8].squeeze().unsqueeze(1)
hp_res_data[mask_out1_depad] = torch.round(all_real_res_depad[mask_out1_depad], decimals=3) # 模拟精度为 0.001

# 残差加上重建数据
print("all_data_recon_depad.shape",all_data_recon_depad.shape)

hp_data = all_data_recon_depad - hp_res_data # 把残差加上

# 计算平均绝对误差 (MAE)
mae_error = torch.mean(torch.abs(hp_data[all_mask_depad] - all_test_depad[all_mask_depad]))
print(f"HP平均绝对误差 (MAE): {mae_error:.6f}")

# 计算最大误差
max_error = torch.max(torch.abs(hp_data[all_mask_depad] - all_test_depad[all_mask_depad]))
print(f"HP最大误差 (Max Error): {max_error:.6f}")

# 计算平均相对误差 (MRE)
mre_error = torch.mean(np.abs((hp_data[all_mask_depad] - all_test_depad[all_mask_depad]) / data_range))
print(f"HP平均相对误差 (REL_mean): {mre_error:.6f}")

# 计算最大相对误差 (MRE_max)
mre_max_error = torch.max(torch.abs((hp_data[all_mask_depad] - all_test_depad[all_mask_depad]) / data_range))
print(f"HP最大相对误差 (REL_max): {mre_max_error:.6f}")

# 计算峰值信噪比 (PSNR)
signal_power = torch.square(data_range)
noise_power = torch.mean(np.square(hp_data[all_mask_depad] - all_test_depad[all_mask_depad]))
psnr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
print(f"HP峰值信噪比 (PSNR): {psnr:.6f} dB")

# 计算压缩率 (CR)
print(f"**HP压缩率**: {(time_len*1200*480*32)/(np.sum(compressed_bits_lst)+(time_len*10*64*64*np.log2(32)+32*32*32)):.6f}")

# 计算比特率 (BR)
print(f"比特率 (BR): {(np.sum(compressed_bits_lst)+(time_len*10*64*64*np.log2(32)+32*32*32))/(time_len*1200*480):.6f} bits per element")
