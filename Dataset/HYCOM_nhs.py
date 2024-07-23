import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class HYCOM_Train(Dataset):
    def __init__(self):
        self.data_path = "E:/MarineDatasets/hycom_scs/"
        self.data_max = 32.377  # 前30个数据的
        self.data_min = 0.0
        self.time_len = 30
        self.times = sorted(os.listdir(self.data_path))[:self.time_len]



        # Mean-Standard Deviation Normalization
        self.mean = self.data[self.mask].mean()
        self.std = self.data[self.mask].std()
        self.data[self.mask] = (self.data[self.mask] - self.mean) / self.std

        # block size
        self.block_size = (240, 240)
        self.padding = (8, 8)

        # 计算每个维度上分的块数
        self.blocks_in_row = self.data[0].shape[0] // self.block_size[0]  # 在行上的块数
        self.blocks_in_col = self.data[0].shape[1] // self.block_size[1]  # 在列上的块数

    def __getitem__(self, idx):
        # 一维索引转换成二维索引
        level_idx = idx // self.time_len
        t_idx = idx % self.time_len
        time = self.times[t_idx]
        file = nc.Dataset(os.path.join(self.data_path, time, f"hycom_glby_930_{time}_t000_ts3z.nc"),"r")
        water_temp = file.variables['water_temp'][:]
        data = np.nan_to_num(water_temp, nan=0.0)
        # 归一化
        data = (data - self.data_min) / (self.data_max - self.data_min)
        mask = ~np.isnan(data)
        # 分区





        data = self.data[index]
        mask = self.mask[index]

        # 分块
        blocks = [data[x:x + self.block_size[0], y:y + self.block_size[1]] for x in
                  range(0, data.shape[0], self.block_size[0]) for y
                  in range(0, data.shape[1], self.block_size[1])]

        # 分块
        masks = [mask[x:x + self.block_size[0], y:y + self.block_size[1]] for x in
                  range(0, mask.shape[0], self.block_size[0]) for y
                  in range(0, mask.shape[1], self.block_size[1])]

        padded_blocks = [np.pad(block, pad_width=(self.padding, self.padding), mode='reflect') for block in blocks]
        padded_masks  = [np.pad(mask, pad_width=(self.padding, self.padding), mode='reflect') for mask in masks]

        # 将padded blocks转换成PyTorch tensor，并堆叠成一个新的tensor
        padded_blocks_tensor = torch.stack([torch.tensor(block) for block in padded_blocks])
        padded_masks_tensor = torch.stack([torch.tensor(mask) for mask in padded_masks])

        return padded_blocks_tensor, padded_masks_tensor

    def __len__(self):
        return self.data.shape[0]

    def inverse_normalize(self, normalized_data):
        # Inverse the normalization
        return normalized_data * self.std + self.mean

    def reconstruct_from_blocks(self, blocks):
        original_shape = self.data[0].shape
        # 初始化重建图像
        reconstructed = np.zeros(original_shape)
        # Block索引
        block_idx = 0
        for i in range(self.blocks_in_row):
            for j in range(self.blocks_in_col):
                # 计算当前block在原始图像中的位置
                start_row = i * self.block_size[0]
                start_col = j * self.block_size[1]

                # 去掉padding后放入重建图像中
                reconstructed[start_row:start_row + self.block_size[0], start_col:start_col + self.block_size[1]] = \
                    blocks[block_idx][self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
                block_idx += 1

        return torch.tensor(reconstructed)

class iHESP_SST_Test(Dataset):
    def __init__(self, idx=0):
        # read file
        file = nc.Dataset(
            'E:/MarineDatasets/B.E.13.B1850C5.ne120_t12.sehires38.003.sunway_02.pop.h.nday1.SST.046001-046912.nc', 'r')
        sst = file.variables['SST'][idx, :]
        file.close()
        self.data = torch.from_numpy(sst.filled(-1))
        self.mask = (self.data != -1)

        # Mean-Standard Deviation Normalization
        self.mean = self.data[self.mask].mean()
        self.std = self.data[self.mask].std()
        self.data[self.mask] = (self.data[self.mask] - self.mean) / self.std

        # block size
        self.block_size = (240, 240)
        self.padding = (8, 8)

    def __getitem__(self, index=0):
        # (2400,3600) 拆分成10*15个(240*240)，然后每个都增加8的padding
        # padding是通过边缘值重复取得的


        # 计算每个维度上分的块数
        blocks_in_row = self.data.shape[0] // self.block_size[0]  # 在行上的块数
        blocks_in_col = self.data.shape[1] // self.block_size[1]  # 在列上的块数

        # 分块
        blocks = [self.data[x:x + self.block_size[0], y:y + self.block_size[1]] for x in range(0, self.data.shape[0], self.block_size[0]) for y
                  in range(0, self.data.shape[1], self.block_size[1])]

        padded_blocks = [np.pad(block, pad_width=(self.padding, self.padding), mode='reflect') for block in blocks]

        # 将padded blocks转换成PyTorch tensor，并堆叠成一个新的tensor
        padded_blocks_tensor = torch.stack([torch.tensor(block) for block in padded_blocks])
        return padded_blocks_tensor

    def __len__(self):
        return 1

    def inverse_normalize(self, normalized_data):
        # Inverse the normalization
        return normalized_data * self.std + self.mean

    def reconstruct_from_blocks(self, blocks):
        """
        从padding过的blocks中重建原始图像。

        :param blocks: 堆叠的blocks
        :return: 重建的图像
        """
        original_shape = self.data.shape

        # 计算每个维度的block数量
        blocks_in_row = original_shape[0] // self.block_size[0]
        blocks_in_col = original_shape[1] // self.block_size[1]

        # 初始化重建图像
        reconstructed = np.zeros(original_shape)

        # Block索引
        block_idx = 0
        for i in range(blocks_in_row):
            for j in range(blocks_in_col):
                # 计算当前block在原始图像中的位置
                start_row = i * self.block_size[0]
                start_col = j * self.block_size[1]

                # 去掉padding后放入重建图像中
                reconstructed[start_row:start_row + self.block_size[0], start_col:start_col + self.block_size[1]] = \
                    blocks[block_idx][self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
                block_idx += 1

        return torch.tensor(reconstructed)


if __name__=="__main__":
    train_dataset = iHESP_SST_Train()
    print("len(train_dataset",len(train_dataset))
    train_data, train_mask = train_dataset.__getitem__(0)
    print(train_data.shape)
    print(train_mask.shape)
    recon_train_data = train_dataset.reconstruct_from_blocks(train_data)
    print("recon数据是否相等",torch.equal(train_dataset.data[0], recon_train_data))
    recon_train_mask = train_dataset.reconstruct_from_blocks(train_mask)
    print("recon mask是否相等", torch.equal(train_dataset.mask[0], recon_train_mask))


    dataset = iHESP_SST_Test(idx=0)
    partitions = dataset.__getitem__(index=0)
    print("partitions.shape",partitions.shape)

    # 重建数据
    reconstructed_data = dataset.reconstruct_from_blocks(partitions)
    print("testdataset recon数据是否相等", torch.equal(dataset.data, reconstructed_data))

    # 可视化结果
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # 显示原始数据
    axs[0].pcolormesh(range(partitions[0].shape[0]), range(partitions[0].shape[1]), partitions[0], cmap='jet', vmin=-1, vmax=30)
    axs[0].set_title('Original Data')

    # 显示一个分区的示例
    axs[1].pcolormesh(range(partitions[0].shape[0]), range(partitions[0].shape[1]), partitions[90], cmap='jet', vmin=-1, vmax=30)
    axs[1].set_title('A Partition Example')

    plt.show()

