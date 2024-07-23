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

from Dataset.iHESP import iHESP_SST_Train,iHESP_SST
from module import Model
import utils
import os
from matplotlib import pyplot as plt

# 先测试重建误差
# load model
print("create model")
device = "cuda:0"
# device = "cpu"
model = Model(num_hiddens=64, num_residual_layers=5, num_residual_hiddens=32,
          num_embeddings=32, embedding_dim=16,
          commitment_cost=0.25, decay=0.99).to(device)
save_dict = torch.load("saves/e2e_model_only_lossy_32_16_good.pth")

print("this model train epoch: ", save_dict['epoch'])
model.load_state_dict(save_dict['model'])
print("create model completed")

# load data
print("loading data...")
dataset = iHESP_SST("E:\MarineDatasets\iHESP_SST")
# dataset.partition()
data,mask = dataset.getitem(index=30, is_norm=True)
test,mask = dataset.getitem(index=30, is_norm=False)
data = data.unsqueeze(1).to(device)
mask = mask.unsqueeze(1)
test = test.unsqueeze(1)
print(data.shape)
print("load data finished.")

# eval
print("start eval...")
model.eval()
with torch.no_grad():
    vq_loss, _data_recon, perplexity, means, stds, weights = model(data)
    print("perplexity: ", perplexity)
    data_recon = dataset.inverse_normalize(_data_recon) # (150, 1,)
    data_recon_flat = dataset.reconstruct_from_blocks(data_recon.squeeze().cpu())
    print("data_recon.shape",data_recon_flat.shape)
print("eval finished.")

# quanti error
real_res = data_recon.cpu() - test
calc_res = real_res.clone()
calc_res[real_res > 1.0] = 1.0 # 只对[-1,1]区间的值进行概率建模和熵编码
calc_res[real_res < -1.0] = -1.0
res_quantized = utils.uniform_quantization(calc_res.cpu(), quan_num=1000, min_val=-1.0, max_val=1.0)
res_dequantized = utils.uniform_dequantization(res_quantized, quan_num=1000, min_val=-1.0, max_val=1.0)
res_abs_quantized_diff = torch.abs(res_dequantized.cpu()-calc_res)
print("quanti mae", torch.mean(res_abs_quantized_diff))
# res_abs_quantized_diff[res_abs_quantized_diff>5]=0
print("quanti max", torch.max(res_abs_quantized_diff).item())

# recon error1
real_res_flat = dataset.reconstruct_from_blocks(real_res.squeeze().cpu())
mask_flat = dataset.reconstruct_from_blocks(mask.squeeze().cpu())
mae = torch.mean(torch.abs(real_res_flat[mask_flat]))
print("1abs max", torch.abs(real_res_flat[mask_flat]).max())
print("1abs min", torch.abs(real_res_flat[mask_flat]).min())
print("1mae: ", mae)

# 计算相对误差
test_flat = dataset.reconstruct_from_blocks(test.squeeze().cpu())
print("test_flat mean: ", torch.mean(test_flat))
# 设置一个小的常数epsilon来避免除零问题
epsilon = 1e-10
# 计算相对误差
relative_error = torch.abs(real_res_flat[mask_flat] / (test_flat[mask_flat]))
# 计算平均相对误差(MRE)
mre = torch.mean(relative_error)
print("MRE: ", mre.item()*100,"%")


# # recon error2
# test_flat = dataset.reconstruct_from_blocks(test.squeeze().cpu())
# mask_flat = dataset.reconstruct_from_blocks(mask.squeeze().cpu())
# diff = test_flat[mask_flat] - data_recon_flat[mask_flat]
# mae = torch.mean(torch.abs(diff))
# print("2abs max", torch.abs(diff).max())
# print("2abs min", torch.abs(diff).min())
# print("2mae: ", mae)

diff_np = real_res_flat[mask_flat].numpy()
# 统计落在区间[-1, 1]内的数据个数
count = len([x for x in diff_np if -1 <= x <= 1])
print("count:",count)
# 计算占比
percentage = count / len(diff_np) * 100
print(f"[-1, 1]区间的数据个数占比为: {percentage:.2f}%")

# # 箱形图
# fig, ax = plt.subplots(figsize=(8, 100),dpi=200)
# ax.boxplot(diff)
# ax.set_title('residual boxplot')
# # 设置x轴的刻度为0.1的倍数
# ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
# # 设置y轴的刻度为0.05的倍数
# ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
# plt.savefig('./saves/residual_boxplot.png')
# plt.show()

# # 数据分布图
# plt.figure(figsize=(100,10))
# plt.hist(diff, bins=200)  # 设置bins的数量来控制直方图的粒度
# plt.title('data distribution')
# plt.xlabel('residual')
# plt.ylabel('frequency')
# # 调整横坐标刻度范围和间隔
# plt.xlim(-10, 10)  # 设置横坐标刻度范围为[-10, 10]
# plt.xticks([i / 10 for i in range(-100, 101, 2)])  # 设置横坐标刻度间隔为0.1
# plt.savefig("./saves/residual_distribution.png")
# plt.show()





# # vis for distribution of res_quantized
# plt.figure()
# plt.hist(res_quantized.flatten().cpu(), bins=100, density=True)
# plt.xlabel('Value')
# plt.ylabel('Density')
# plt.title('Distribution of Data')
# plt.show()

# fig, axs = plt.subplots(1, 2, figsize=(15,5))
# # 显示原始数据
# data_test[~data_mask] = -1
# axs[0].pcolormesh(range(data_test.shape[1]), range(data_test.shape[0]), data_test, cmap='jet', vmin=-1, vmax=33)
# # 添加第一个矩形框框
# rect1 = patches.Rectangle((2500, 1000), 700, 700, linewidth=1, edgecolor='g', facecolor='none')
# axs[0].add_patch(rect1)
# axs[0].set_title('Original Data')
# # 显示一个分区的示例
# data_recon[~data_mask] = -1
# axs[1].pcolormesh(range(data_recon.shape[1]), range(data_recon.shape[0]), data_recon, cmap='jet', vmin=-1, vmax=33)
# rect2 = patches.Rectangle((2500, 1000), 700, 700, linewidth=1, edgecolor='g', facecolor='none')
# axs[1].add_patch(rect2)
# axs[1].set_title('Reconstruction Data')
# plt.savefig("saves/全球对比.png")
# plt.show()

# # 放大可视化
# fig, axs = plt.subplots(1, 2, figsize=(20,10))
# print(data_test[1000:1000+700, 2500:2500+700])
# # 显示原始数据
# axs[0].pcolormesh(range(700), range(700), data_test[1000:1000+700, 2500:2500+700], cmap='jet', vmin=-1, vmax=33)
# axs[0].set_title('Original Data')
# axs[1].pcolormesh(range(700), range(700), data_recon[1000:1000+700, 2500:2500+700], cmap='jet', vmin=-1, vmax=33)
# axs[1].set_title('Reconstruction Data')
# plt.savefig("saves/放大对比.png")
# plt.show()

# # 看看[-1,1]之外数据能不能对上
# mask_in1 = torch.zeros_like(mask, dtype=torch.bool)
# mask_in1[(real_res >= -1.0) & (real_res <= 1.0)] = True
# mask_in1[~mask] = False
# print("[-1,1]之内数据个数:", torch.sum(mask_in1[:,:,8:-8,8:-8,...]))
# print("[-1,1]之内数据比例：",(torch.sum(mask_in1[:,:,8:-8,8:-8,...])/len(diff_np)).item())

mask_out1 = torch.zeros_like(mask, dtype=torch.bool)
mask_out1[mask & ((real_res > 1.0)|(real_res < -1.0))] = True
print("[-1,1]之外数据比例：",(torch.sum(mask_out1[:,:,8:-8,8:-8,...])/len(diff_np)).item())



compressed_bits_lst = []
for idx in range(means.shape[0]):
    # 两个问题，1是没depadding，2是计算出来[-1,1]的比率不一致
    compressed_bits = utils.get_compressed_bits(res_quantized[idx,:,8:-8,8:-8,...].unsqueeze(0).cpu(),
                                                mask[idx,:,8:-8,8:-8, ...].unsqueeze(0).cpu(),
                                                mask_out1[idx,:,8:-8,8:-8, ...].unsqueeze(0).cpu(),
                                                means[idx,8:-8,8:-8, ...].unsqueeze(0).cpu(),
                                                stds[idx,8:-8,8:-8, ...].unsqueeze(0).cpu(),
                                                weights[idx,8:-8,8:-8, ...].unsqueeze(0).cpu(),
                                                quan_num=1000, HW_size=240)

    compressed_bits_lst.append(compressed_bits)

print("CR: ", 2400*3600*32/np.sum(compressed_bits_lst))










