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

from Dataset.iHESP import iHESP_SST_Train,iHESP_SST
from module import Model
import utils
import os

if __name__=="__main__":

    print("loading data")
    dataset = iHESP_SST("E:\MarineDatasets\iHESP_SST") # 主要是为了用逆归一化
    train_dataset = iHESP_SST_Train("E:\MarineDatasets\iHESP_SST")
    dataloader  = DataLoader(train_dataset,  batch_size=1, shuffle=True)

    # print("data shape in time 0:", data.shape)
    # print("mask shape in time 0:", mask.shape)
    print("loading data completed")

    print("create model")
    device = "cuda:0"
    model = Model(num_hiddens=64, num_residual_layers=5, num_residual_hiddens=32,
              num_embeddings=32, embedding_dim=16,
              commitment_cost=0.25, decay=0.99).to(device)
    save_dict = torch.load("saves/e2e_model_only_lossy_32_16_good.pth")
    print("this model train epoch: ", save_dict['epoch'])
    model.load_state_dict(save_dict['model'])
    epochs = 1000
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    print("create model completed")

    print("start training")
    # alpha =
    model.train()

    min_loss = float('inf')

    for ep in range(epochs):
        all_losses = []
        recon_losses = []
        prob_losses = []
        start_time = time.time()
        for idx, (data, test, mask) in enumerate(dataloader):
            data = data.to(device) # (batch, channel=1, H, W)
            test = test.to(device)
            # print("data.shape",data.shape)
            vq_loss, data_recon, perplexity, means, stds, weights = model(data)
            recon_error = F.mse_loss(data_recon, data)
            # 计算残差
            denorm_data = dataset.inverse_normalize(data_recon)
            res_data = denorm_data - test
            # 量化
            res_quantized = utils.uniform_quantization(res_data, quan_num=1000, min_val=-1.0, max_val=1.0)
            # 计算概率损失
            probloss = utils.estimate_prob_and_loss(means, stds, weights, res_quantized, quan_num=1000, skip=5,
                                                    HW_size=256)
            loss = vq_loss + recon_error + 0.08*probloss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            all_losses.append(loss.item())
            recon_losses.append(recon_error.item())
            prob_losses.append(probloss.item())
            # print(f"idx {idx}, recon_loss {recon_error.item()}, probloss {probloss.item()}, all_loss {loss.item()}")
        end_time = time.time()
        print("time: ", end_time-start_time)

        print(f"epoch {ep + 25} finish, all loss {np.mean(all_losses)}, recon loss {np.mean(recon_losses)}, prob loss {np.mean(prob_losses)}")

        if np.mean(all_losses)<min_loss:
            min_loss =  np.mean(all_losses)
            print("save model ...")
            utils.save_model(ep=ep+25, model=model, model_name='tmp.pth')









