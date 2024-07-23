import time

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

from Dataset.HYCOM import HYCOM,HYCOM_Train
from module import Model,LPModel,HPModel
import utils
import os
from matplotlib import pyplot as plt

if __name__=="__main__":

    print("loading data")
    dataset = HYCOM("E:\MarineDatasets\HYCOM\WTB") # 改
    train_dataset = HYCOM_Train("E:\MarineDatasets\HYCOM\WTB", time=4) # 改
    dataloader  = DataLoader(train_dataset,  batch_size=1, shuffle=True)
    print("len dataloader:",len(dataloader))

    # for data, test, mask in dataloader:
    #     # 创建一个子图网格
    #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #     # 绘制data、test和mask
    #     data_img = axes[0].imshow(data[0, 0], cmap='jet', vmin=0, vmax=1)
    #     axes[0].set_title('Data')
    #     test_img = axes[1].imshow(test[0, 0], cmap='jet', vmin=dataset.min_val, vmax=dataset.max_val)
    #     axes[1].set_title('Test')
    #     mask_img = axes[2].imshow(mask[0, 0], cmap='gray', vmin=0, vmax=1)
    #     axes[2].set_title('Mask')
    #     # 添加colorbar到data、test和mask图像
    #     cbar1 = plt.colorbar(data_img, ax=axes[0])
    #     cbar1.set_label('Data Values')
    #     cbar2 = plt.colorbar(test_img, ax=axes[1])
    #     cbar2.set_label('Test Values')
    #     cbar3 = plt.colorbar(mask_img, ax=axes[2])
    #     cbar3.set_label('Mask Values')
    #     # 添加适当的标签和标题
    #     axes[0].set_xlabel('Width')
    #     axes[0].set_ylabel('Height')
    #     axes[1].set_xlabel('Width')
    #     axes[1].set_ylabel('Height')
    #     axes[2].set_xlabel('Width')
    #     axes[2].set_ylabel('Height')
    #     # 显示图像
    #     plt.show()
    #     break

    # print("data shape in time 0:", data.shape)
    # print("mask shape in time 0:", mask.shape)
    print("loading data completed")

    print("create model")
    device = "cuda:0"
    # model = Model(num_hiddens=64, num_residual_layers=5, num_residual_hiddens=32,
    #           num_embeddings=32, embedding_dim=16,
    #           commitment_cost=0.25, decay=0.99).to(device)
    lpmodel = LPModel(num_hiddens=64, num_residual_layers=5, num_residual_hiddens=32,
                      num_embeddings=16, embedding_dim=48,
                      commitment_cost=0.25, decay=0.99).to(device)
    # hpmodel = HPModel(num_hiddens=64, num_residual_layers=5, num_residual_hiddens=32,
    #                   num_embeddings=32, embedding_dim=16,
    #                   commitment_cost=0.25, decay=0.99).to(device)
    # 加载lpmodel的权重
    # lp_save_dict = torch.load("saves/ablation/HYCOM_WTB_lp_16_48.pth")
    # lpmodel.load_state_dict(lp_save_dict['model'])

    # 加载hpmodel的权重
    # hp_save_dict = torch.load("saves/HYCOM_SAB_hp_32_16.pth")
    # hpmodel.load_state_dict(hp_save_dict['model'])

    # print("this model train epoch: ", save_dict['epoch'])
    # model.load_state_dict(save_dict['model'])
    epochs = 200
    learning_rate = 1e-4   # 改
    optimizer = torch.optim.Adam(lpmodel.parameters(), lr=learning_rate, amsgrad=True) # 记得改
    # optimizer = torch.optim.Adam(hpmodel.parameters(), lr=learning_rate, amsgrad=True)
    print("create model completed")

    print("start training")
    lpmodel.train() # 改
    # hpmodel.train()

    min_loss = float('inf')

    for ep in range(epochs):
        all_losses = []
        recon_losses = []
        prob_losses = []
        start_time = time.time()
        for idx, (data, test, mask) in enumerate(dataloader):
            try:
                data = data.to(device)  # (batch, channel=1, H, W)
                test = test.to(device)
                # print("data.shape",data.shape)
                # with torch.no_grad():
                vq_loss, data_recon, perplexity, quantized = lpmodel(data)
                # vq_loss, data_recon, perplexity, quantized = lpmodel(data)
                # quantized = quantized.detach() # 切断梯度，使损失函数只与HPModel有关
                # means, stds, weights = hpmodel(quantized)
                recon_error = F.mse_loss(data_recon, data)
                # 计算残差
                # denorm_data = dataset.inverse_normalize(data_recon)
                # res_data = denorm_data - test
                # 量化
                # res_quantized = utils.uniform_quantization(res_data, quan_num=1001, min_val=-1.0, max_val=1.0)
                # 计算概率损失
                # probloss = utils.estimate_prob_and_loss(means, stds, weights, res_quantized, quan_num=1001, skip=5,
                #                                                      HW_size=256)
                loss = vq_loss + recon_error
                # loss = probloss
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(hpmodel.parameters(), max_norm=0.1) # 截断梯度防止爆炸
                optimizer.step()
                optimizer.zero_grad()
                all_losses.append(loss.item())
                recon_losses.append(recon_error.item())
                # prob_losses.append(probloss.item())
            except Exception as e:
                print(e)
        # print("idx:",idx, "probloss:",probloss)
        # prob_losses.append(probloss.item())
        # print(f"idx {idx}, recon_loss {recon_error.item()}, probloss {probloss.item()}, all_loss {loss.item()}")
        # except Exception as e:
        #     print("torch.isnan(means).any()", torch.isnan(means).any())
        #     print("此时idx:", idx)
        #     print("data",data)
        #     print("mask",mask)
        end_time = time.time()
        print("time: ", end_time - start_time)

        print(
            f"epoch {ep + 60} finish, all loss {np.mean(all_losses)}, recon loss {np.mean(recon_losses)}, prob loss {np.mean(prob_losses)}")

        if np.mean(all_losses) < min_loss:
            min_loss = np.mean(all_losses)
            print("save model ...")
            utils.save_model(ep=ep + 60, model=lpmodel, model_name=f'ablation/HYCOM_WTB_lp_16_48.pth') # 改2处









