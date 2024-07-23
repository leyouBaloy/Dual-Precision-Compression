import logging
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

from Dataset.HYCOM import HYCOM, HYCOM_Train
from module import Model, LPModel, HPModel
import utils
import os
from matplotlib import pyplot as plt

num_embeddings = 32
embedding_dim = 32
filename = f"HYCOM_SEL_lt_lp_ga_part3"
save_base_path = './saves/ablation/HYCOM_SEL_lt_partition'


# 设置日志记录器，在文件名中加入时间戳
logging.basicConfig(filename=f'{save_base_path}/{filename}.log', level=logging.INFO, format='%(message)s')


if __name__ == "__main__":
    logging.info("loading data")
    dataset = HYCOM("/home/bailey/dataset/HYCOM/SEL_lt")  # 改
    train_dataset = HYCOM_Train("/home/bailey/dataset/HYCOM/SEL_lt", start_time=196, time=105)  # 改
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    logging.info("len dataloader: %d", len(dataloader))


    # print("data shape in time 0:", data.shape)
    # print("mask shape in time 0:", mask.shape)
    logging.info("loading data completed")

    logging.info("create model")
    device = "cuda:0"

    lpmodel = LPModel(num_hiddens=64, num_residual_layers=5, num_residual_hiddens=32,
                      num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                      commitment_cost=0.25, decay=0.99).to(device)
    # hpmodel = HPModel(num_hiddens=64, num_residual_layers=5, num_residual_hiddens=32,
    #                   num_embeddings=32, embedding_dim=16,
    #                   commitment_cost=0.25, decay=0.99).to(device)
    # 加载lpmodel的权重
    # lp_save_dict = torch.load("saves/ablation/HYCOM_WTB_lt_partition_lp/HYCOM_SEL_lt_lp_ga_part3.pth")
    # lpmodel.load_state_dict(lp_save_dict['model'])

    # 加载hpmodel的权重
    # hp_save_dict = torch.load("saves/HYCOM_SAB_hp_32_16.pth")
    # hpmodel.load_state_dict(hp_save_dict['model'])

    # print("this model train epoch: ", save_dict['epoch'])
    # model.load_state_dict(save_dict['model'])
    epochs = 1000
    learning_rate = 1e-4  # 改
    optimizer = torch.optim.Adam(lpmodel.parameters(), lr=learning_rate, amsgrad=True)  # 记得改
    # optimizer = torch.optim.Adam(hpmodel.parameters(), lr=learning_rate, amsgrad=True)
    logging.info("create model completed")

    logging.info("start training")
    lpmodel.train()  # 改
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
                logging.error(e)
        # print("idx:",idx, "probloss:",probloss)
        # prob_losses.append(probloss.item())
        # print(f"idx {idx}, recon_loss {recon_error.item()}, probloss {probloss.item()}, all_loss {loss.item()}")
        # except Exception as e:
        #     print("torch.isnan(means).any()", torch.isnan(means).any())
        #     print("此时idx:", idx)
        #     print("data",data)
        #     print("mask",mask)
        end_time = time.time()
        
        import datetime
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y年%m月%d日 %H时%M分%S秒")
        logging.info(formatted_time)

        logging.info("time: %f", end_time - start_time)

        print(f"epoch {ep + 0} finish, all loss {np.mean(all_losses)}, recon loss {np.mean(recon_losses)}, prob loss {np.mean(prob_losses)}")
        logging.info(
            f"epoch {ep + 0} finish, all loss {np.mean(all_losses)}, recon loss {np.mean(recon_losses)}, prob loss {np.mean(prob_losses)}")

        if np.mean(all_losses) < min_loss:
            min_loss = np.mean(all_losses)
            logging.info("save model ...")
            utils.save_model(ep=ep + 0, model=lpmodel, save_dir=save_base_path, model_name=f'{filename}.pth')  # 改2处