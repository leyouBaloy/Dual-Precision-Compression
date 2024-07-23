import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math
import skimage
from skimage import metrics
import numpy as np
import sys
import torch.nn.functional as F
import torchac.torchac
import matplotlib.patches as patches

from Dataset.HYCOM import HYCOM,HYCOM_Train
from module import Model,LPModel,HPModel
import utils
import os
from matplotlib import pyplot as plt
import time

# 先测试重建误差
# load model
print("create model")
device = "cuda:0"
# device = "cpu"
lpmodel = LPModel(num_hiddens=64, num_residual_layers=5, num_residual_hiddens=32,
              num_embeddings=32, embedding_dim=32,
              commitment_cost=0.25, decay=0.99).to(device)
hpmodel = HPModel(num_hiddens=64, num_residual_layers=5, num_residual_hiddens=32,
                      num_embeddings=32, embedding_dim=32,
                      commitment_cost=0.25, decay=0.99, n_component=2).to(device)
lp_save_dict = torch.load("saves/ablation/WTB_lt_train16/lp_train16.pth") # 后面把comparison改一下
hp_save_dict = torch.load("saves/ablation/WTB_lt_train16/hp_tain16_ep260.pth")

print("lp model train epoch: ", lp_save_dict['epoch'])
print("hp model train epoch: ", hp_save_dict['epoch'])

lpmodel.load_state_dict(lp_save_dict['model'])
hpmodel.load_state_dict(hp_save_dict['model'])

print("create model completed")

# load data
print("loading data...")
dataset = HYCOM("/home/bailey/dataset/HYCOM/WTB_lt")
# dataset.partition()
data,mask = dataset.getitem(index=352, is_norm=True)
test,mask = dataset.getitem(index=352, is_norm=False)

data = data.unsqueeze(1).to(device)
mask = mask.unsqueeze(1)
test = test.unsqueeze(1)

# data = data[:,8:-8,8:-8].unsqueeze(1).to(device)
# mask = mask[:,8:-8,8:-8].unsqueeze(1)
# test = test[:,8:-8,8:-8].unsqueeze(1)
print(data.shape)
print("load data finished.")

# eval
print("start eval...")
lpmodel.eval()
hpmodel.eval()
with torch.no_grad():
    vq_loss, _data_recon, perplexity, quantized = lpmodel(data)
    means, stds, weights = hpmodel(quantized)
    print("perplexity: ", perplexity.item())
    data_recon = dataset.inverse_normalize(_data_recon) # (150, 1,)
    data_recon[data_recon>dataset.max_val] = dataset.max_val
    data_recon[data_recon<dataset.min_val] = dataset.min_val
    data_recon_flat = dataset.reconstruct_from_blocks(data_recon.squeeze().cpu())
    # print("data_recon.shape",data_recon_flat.shape)
print("eval finished.")

# quanti error
real_res = data_recon.cpu() - test
calc_res = real_res.clone()
calc_res[real_res > 1.0] = 1.0 # 只对[-1,1]区间的值进行概率建模和熵编码
calc_res[real_res < -1.0] = -1.0
res_quantized = utils.uniform_quantization(calc_res.cpu(), quan_num=1000, min_val=-1.0, max_val=1.0)
res_dequantized = utils.uniform_dequantization(res_quantized, quan_num=1000, min_val=-1.0, max_val=1.0)
res_abs_quantized_diff = torch.abs(res_dequantized.cpu()-calc_res)
print(f"量化平均绝对误差MAE: {torch.mean(res_abs_quantized_diff).item():.6f}")
# res_abs_quantized_diff[res_abs_quantized_diff>5]=0
print(f"量化最大误差MAX: {torch.max(res_abs_quantized_diff).item():.6f}")

print("======================")
# recon error1
real_res_flat = dataset.reconstruct_from_blocks(real_res.squeeze().cpu())
mask_flat = dataset.reconstruct_from_blocks(mask.squeeze().cpu())
mae = torch.mean(torch.abs(real_res_flat[mask_flat]))
print(f"LP平均绝对误差(MAE): {mae:.6f}")
print(f"LP最大误差(MAX error): {torch.abs(real_res_flat[mask_flat]).max():.6f}")

# 计算相对误差
data_range = dataset.max_val-dataset.min_val
# print("==========data range==========",data_range)
test_flat = dataset.reconstruct_from_blocks(test.squeeze().cpu())
# print("test_flat mean: ", torch.mean(test_flat))
# 计算相对误差
relative_error = torch.abs(real_res_flat[mask_flat] / data_range)
# 计算平均相对误差(MRE)
rel = torch.mean(relative_error)
print(f"LP平均相对误差(REL): {rel.item():.6f}", )
print(f"LP最大相对误差(REL max): {torch.abs(real_res_flat[mask_flat]).max()/data_range:.6f}")

# 计算峰值信噪比 (PSNR)
mse = torch.mean(torch.square(real_res_flat[mask_flat]))
psnr = 10 * torch.log10(torch.square(data_range) / mse) if mse > 0 else float('inf')
print(f"LP峰值信噪比 (PSNR): {psnr:.6f} dB")

print(f"LP压缩率: {(16*3120*4320*32)/(16*234*64*64*np.log2(32)+32*32*32):.6f}")
print(f"LP bpp: {(16*234*64*64*np.log2(32)+32*32*32)/(16*3120*4320):.6f}")

# LP可视化对比
import matplotlib.pyplot as plt

# 加载原始数据和LP输出结果
original_data = test_flat.cpu().clone()
lp_output = data_recon_flat.clone()

original_data[~mask_flat] = 0.0
lp_output[~mask_flat] = 0.0

# 创建子图（将子图布局改为横向）
fig, axs = plt.subplots(3, 1, figsize=(5, 15))

# 绘制原始数据
im1 = axs[0].pcolormesh(range(original_data.shape[1]), range(original_data.shape[0]), original_data, cmap='jet', vmin=-1, vmax=33)
axs[0].set_title('Original Data')

# 为原始数据添加colorbar
fig.colorbar(im1, ax=axs[0])

# 绘制LP输出结果
im2 = axs[1].pcolormesh(range(lp_output.shape[1]), range(lp_output.shape[0]), lp_output, cmap='jet', vmin=-1, vmax=33)
axs[1].set_title('LP Reconstruction')

# 为LP输出结果添加colorbar
fig.colorbar(im2, ax=axs[1])

# 追加一个子图来绘制mask_flat
axs = axs.flatten()  # 将axs从2x1的数组变为1维数组
im3 = axs[2].pcolormesh(range(mask_flat.shape[1]), range(mask_flat.shape[0]), mask_flat, cmap='gray', vmin=0, vmax=1)
axs[2].set_title('Mask')

# 为mask_flat添加colorbar
fig.colorbar(im3, ax=axs[2])

# 保存图像
plt.savefig('./saves/SAB_lp_comparison.png')

# 显示图像
plt.show()

diff_np = real_res_flat[mask_flat].numpy()
# 统计落在区间[-1, 1]内的数据个数
count = len([x for x in diff_np if -1 <= x <= 1])
# print("count:",count)
# 计算占比
percentage = count / len(diff_np)
print(f"[-1, 1]区间的数据个数占比为: {percentage:.6f}")

mask_out1 = torch.zeros_like(mask, dtype=torch.bool)
mask_out1[mask & ((real_res > 1.0)|(real_res < -1.0))] = True # 有效数据，且在[-1,1]之外
print(f"[-1,1]之外数据比例: {(torch.sum(mask_out1[:,:,8:-8,8:-8,...])/len(diff_np)).item():.6f}")

print("means.shape",means.shape)

compressed_bits_lst = []
hp_datas = []
start_time = time.time()
for idx in range(means.shape[0]):
    compressed_bits,hp_data_one = utils.get_compressed_bits(res_quantized[idx,:,8:-8,8:-8,...].unsqueeze(0).cpu(),
                                                mask[idx,:,8:-8,8:-8, ...].unsqueeze(0).cpu(),
                                                mask_out1[idx,:,8:-8,8:-8, ...].unsqueeze(0).cpu(),
                                                means[idx,8:-8,8:-8, ...].unsqueeze(0).cpu(),
                                                stds[idx,8:-8,8:-8, ...].unsqueeze(0).cpu(),
                                                weights[idx,8:-8,8:-8, ...].unsqueeze(0).cpu(),
                                                quan_num=1000, HW_size=240)
    compressed_bits_lst.append(compressed_bits)
    hp_datas.append(hp_data_one.squeeze(0))

end_time = time.time()
print("======================")
print(f"HP压缩率: {(1200*480*32)/(np.sum(compressed_bits_lst)+(10*64*64*np.log2(32)+ 32*32*32)):.6f}")
print(f"耗时 {end_time-start_time}")

body_mask = (mask^mask_out1) # 异或操作
body_mask_depad = body_mask[:,:,8:-8,8:-8].squeeze().unsqueeze(1)
body_mask_flat = dataset.reconstruct_from_blocks(body_mask.squeeze().cpu())
mask_out1_flat = dataset.reconstruct_from_blocks(mask_out1.squeeze().cpu())
res_dequantized_depad = res_dequantized[:,:,8:-8,8:-8].squeeze().unsqueeze(1)

# 模拟出重建数据
hp_res_data = torch.stack(hp_datas).squeeze().unsqueeze(1) # (234, 1, 240, 240) 不包含mask_out1
# 把[-1,1]之外的数据替换为real_res
mask_out1_depad = mask_out1[:,:,8:-8,8:-8].squeeze().unsqueeze(1)
real_res_depad = real_res[:,:,8:-8,8:-8].squeeze().unsqueeze(1)
hp_res_data[mask_out1_depad] = torch.round(real_res_depad[mask_out1_depad], decimals=3) # 模拟精度为 0.001

# hp_res_data[body_mask_depad] = res_dequantized_depad[body_mask_depad] # 测试一下[-1,1]直接使用残差

# 残差加上重建数据
data_recon_depad = data_recon.cpu()[:,:,8:-8,8:-8].squeeze().unsqueeze(1)
hp_data = data_recon_depad - hp_res_data # 把残差加上
# 再把pad加回来
hp_data_pad = torch.zeros(res_dequantized.shape, dtype=torch.float32)
hp_data_pad[:,:,8:-8,8:-8] = hp_data
hp_data_flat = dataset.reconstruct_from_blocks(hp_data_pad.squeeze().cpu()) # (3120, 4320)

# 计算平均绝对误差 (MAE)
mae_error = torch.mean(torch.abs(hp_data_flat[mask_flat] - test_flat[mask_flat]))
print(f"HP平均绝对误差 (MAE): {mae_error:.6f}")

# 计算最大误差
max_error = torch.max(torch.abs(hp_data_flat[mask_flat] - test_flat[mask_flat]))
print(f"HP最大误差 (Max Error): {max_error:.6f}")

# 计算平均相对误差 (MRE)
mre_error = torch.mean(np.abs((hp_data_flat[mask_flat] - test_flat[mask_flat]) / data_range))
print(f"HP平均相对误差 (REL_mean): {mre_error:.6f}")

# 计算最大相对误差 (MRE_max)
mre_max_error = torch.max(torch.abs((hp_data_flat[mask_flat] - test_flat[mask_flat]) / data_range))
print(f"HP最大相对误差 (REL_max): {mre_max_error:.6f}")

# 计算峰值信噪比 (PSNR)
signal_power = torch.square(data_range)
noise_power = torch.mean(np.square(hp_data_flat[mask_flat] - test_flat[mask_flat]))
psnr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
print(f"HP峰值信噪比 (PSNR): {psnr:.6f} dB")

# 计算压缩率 (CR)
print(f"**HP压缩率**: {(1200*480*32)/(np.sum(compressed_bits_lst)+(10*64*64*np.log2(32)+32*32*32)):.6f}")

# 计算比特率 (BR)
print(f"比特率 (BR): {(np.sum(compressed_bits_lst)*16+(16*234*64*64*np.log2(32)+32*32*32))/(16*3120*4320):.6f} bits per element")
