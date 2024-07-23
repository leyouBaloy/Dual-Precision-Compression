import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import os
import json
from torch.utils.data import DataLoader
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class iHESP_SST_Train(Dataset):
    def __init__(self, base_path="E:\MarineDatasets\iHESP_SST"):
        self.base_path = base_path

        # self.dataset = iHESP_SST("E:\MarineDatasets\iHESP_SST")

    def __getitem__(self, idx):
        time_idx = idx // 150
        part_idx = idx % 150
        test_data = torch.load(os.path.join(self.base_path, "partition", f"{time_idx}_test.pt"))
        train_data = torch.load(os.path.join(self.base_path, "partition", f"{time_idx}_train.pt"))
        mask_data = torch.load(os.path.join(self.base_path, "partition", f"{time_idx}_mask.pt"))
        return train_data[part_idx,...].unsqueeze(0), test_data[part_idx,...].unsqueeze(0), mask_data[part_idx,...].unsqueeze(0)

    def __len__(self):
        return 150*30 # 前30个时间点

class iHESP_SST():
    def __init__(self, base_path):
        self.base_path = base_path
        if os.path.exists(os.path.join(self.base_path, "config.json")):
            print("find config file, loading config...")
            with open(os.path.join(self.base_path, "config.json"), 'r') as f:
                config = json.load(f)
                self.origin_data_shape = config.get("origin_data_shape")
                self.mean = torch.tensor(config.get("mean"))
                self.std = torch.tensor(config.get("std"))
                # self.max_val = torch.tensor(config.get("max_val"))
                # self.min_val = torch.tensor(config.get("min_val"))
                self.block_size = config.get("block_size")
                self.padding = config.get("padding")
                self.blocks_in_row = config.get("blocks_in_row")
                self.blocks_in_col = config.get("blocks_in_col")
        else:
            print("not find config")

    def partition(self):
        print("start partition...")
        # save_path = os.path.join(self.base_path, dirname)
        # if not os.path.isdir(save_path):
        #     os.makedirs(save_path)
        # read file
        print("loading original data...")
        file = nc.Dataset(
            os.path.join(self.base_path,'B.E.13.B1850C5.ne120_t12.sehires38.003.sunway_02.pop.h.nday1.SST.046001-046912.nc'),
            'r')
        sst = file.variables['SST'][:]
        file.close()
        original_data = torch.from_numpy(sst.filled(-1))
        original_mask = (original_data != -1)
        self.origin_data_shape = original_data.shape

        # Mean-Standard Deviation Normalization
        self.mean = original_data[original_mask].mean()
        self.std = original_data[original_mask].std()
        self.max_val = original_data[original_mask].max()
        self.min_val = original_data[original_mask].min()

        # block size
        self.block_size = (240, 240)
        self.padding = (8, 8)

        # 计算每个维度上分的块数
        self.blocks_in_row = original_data[0].shape[0] // self.block_size[0]  # 在行上的块数
        self.blocks_in_col = original_data[0].shape[1] // self.block_size[1]  # 在列上的块数


        for time_idx in range(0,40): # 对前40个时间点的数据进行分区
            print(f"partition time_idx {time_idx}")
            dataone = original_data[time_idx]
            maskone = original_mask[time_idx]

            # 分块
            blocks = [dataone[x:x + self.block_size[0], y:y + self.block_size[1]] for x in
                      range(0, dataone.shape[0], self.block_size[0]) for y
                      in range(0, dataone.shape[1], self.block_size[1])]

            # 分块
            masks = [maskone[x:x + self.block_size[0], y:y + self.block_size[1]] for x in
                     range(0, maskone.shape[0], self.block_size[0]) for y
                     in range(0, maskone.shape[1], self.block_size[1])]

            padded_blocks = [np.pad(block, pad_width=(self.padding, self.padding), mode='reflect') for block in blocks]
            padded_masks = [np.pad(mask, pad_width=(self.padding, self.padding), mode='reflect') for mask in masks]

            # 将padded blocks转换成PyTorch tensor，并堆叠成一个新的tensor
            padded_blocks_tensor = torch.stack([torch.tensor(block) for block in padded_blocks])
            padded_masks_tensor = torch.stack([torch.tensor(mask) for mask in padded_masks])

            torch.save(padded_blocks_tensor, os.path.join(self.base_path,"partition",f"{time_idx}_test.pt"))
            torch.save(padded_masks_tensor, os.path.join(self.base_path,"partition",f"{time_idx}_mask.pt"))
            # 归一化
            padded_blocks_tensor[padded_masks_tensor] = (padded_blocks_tensor[padded_masks_tensor] - self.mean) / self.std
            torch.save(padded_blocks_tensor, os.path.join(self.base_path,"partition",f"{time_idx}_train.pt"))
        print("saving config...")
        config = {
            "origin_data_shape":self.origin_data_shape,
            "mean": self.mean.item(),
            "std": self.std.item(),
            "max_val": self.max_val.item(),
            "min_val": self.min_val.item(),
            "block_size": self.block_size,
            "padding": self.padding,
            "blocks_in_row": self.blocks_in_row,
            "blocks_in_col": self.blocks_in_col
        }
        with open(os.path.join(self.base_path,"config.json"), 'w') as f:
            json.dump(config, f)
        print("partition completed!")

    def getitem(self, index, is_norm=False):
        _mask = torch.load(os.path.join(self.base_path, "partition", f"{index}_mask.pt"))
        if is_norm:
            _data = torch.load(os.path.join(self.base_path,"partition",f"{index}_train.pt"))
        else:
            _data = torch.load(os.path.join(self.base_path,"partition",f"{index}_test.pt"))
        return _data, _mask

    def __len__(self):
        return 40

    def inverse_normalize(self, normalized_data):
        return normalized_data * self.std + self.mean

    # 已验证
    def reconstruct_from_blocks(self, blocks):
        assert blocks.dim() == 3
        original_shape = (2400, 3600)
        # 初始化数组
        if blocks.dtype==torch.float32:
            reconstructed = np.zeros(original_shape, dtype=np.float32)
        if blocks.dtype==torch.bool:
            reconstructed = np.zeros(original_shape, dtype=np.bool_)
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

    def export_data(self, savepath):
        file = nc.Dataset(
            os.path.join(self.base_path,
                         'B.E.13.B1850C5.ne120_t12.sehires38.003.sunway_02.pop.h.nday1.SST.046001-046912.nc'),
            'r')
        sst = file.variables['SST'][:]
        file.close()
        original_data = torch.from_numpy(sst.filled(-1))
        original_mask = (original_data != -1)
        savedata = original_data[30].numpy()
        savemask = original_mask[30].numpy()
        savedata.tofile(os.path.join(savepath, '30th_data.bin'))
        savemask.tofile(os.path.join(savepath, '30th_mask.bin'))
        print("save completely !")

# class iHESP_SST_Test(Dataset):
#     def __init__(self, idx=0):
#         # read file
#         file = nc.Dataset(
#             'E:/MarineDatasets/B.E.13.B1850C5.ne120_t12.sehires38.003.sunway_02.pop.h.nday1.SST.046001-046912.nc', 'r')
#         sst = file.variables['SST'][idx, :]
#         file.close()
#         self.data = torch.from_numpy(sst.filled(-1))
#         self.mask = (self.data != -1)
#
#         # Mean-Standard Deviation Normalization
#         self.mean = self.data[self.mask].mean()
#         self.std = self.data[self.mask].std()
#         self.data[self.mask] = (self.data[self.mask] - self.mean) / self.std
#
#         # block size
#         self.block_size = (240, 240)
#         self.padding = (8, 8)
#
#     def __getitem__(self, index=0):
#         # (2400,3600) 拆分成10*15个(240*240)，然后每个都增加8的padding
#         # padding是通过边缘值重复取得的
#
#
#         # 计算每个维度上分的块数
#         blocks_in_row = self.data.shape[0] // self.block_size[0]  # 在行上的块数
#         blocks_in_col = self.data.shape[1] // self.block_size[1]  # 在列上的块数
#
#         # 分块
#         blocks = [self.data[x:x + self.block_size[0], y:y + self.block_size[1]] for x in range(0, self.data.shape[0], self.block_size[0]) for y
#                   in range(0, self.data.shape[1], self.block_size[1])]
#
#         padded_blocks = [np.pad(block, pad_width=(self.padding, self.padding), mode='reflect') for block in blocks]
#
#         # 将padded blocks转换成PyTorch tensor，并堆叠成一个新的tensor
#         padded_blocks_tensor = torch.stack([torch.tensor(block) for block in padded_blocks])
#         return padded_blocks_tensor
#
#     def __len__(self):
#         return 1
#
#     def inverse_normalize(self, normalized_data):
#         # Inverse the normalization
#         return normalized_data * self.std + self.mean
#
#     def reconstruct_from_blocks(self, blocks):
#         """
#         从padding过的blocks中重建原始图像。
#
#         :param blocks: 堆叠的blocks
#         :return: 重建的图像
#         """
#         original_shape = self.data.shape
#
#         # 计算每个维度的block数量
#         blocks_in_row = original_shape[0] // self.block_size[0]
#         blocks_in_col = original_shape[1] // self.block_size[1]
#
#         # 初始化重建图像
#         reconstructed = np.zeros(original_shape)
#
#         # Block索引
#         block_idx = 0
#         for i in range(blocks_in_row):
#             for j in range(blocks_in_col):
#                 # 计算当前block在原始图像中的位置
#                 start_row = i * self.block_size[0]
#                 start_col = j * self.block_size[1]
#
#                 # 去掉padding后放入重建图像中
#                 reconstructed[start_row:start_row + self.block_size[0], start_col:start_col + self.block_size[1]] = \
#                     blocks[block_idx][self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
#                 block_idx += 1
#
#         return torch.tensor(reconstructed)


if __name__=="__main__":
    # # 测试iHESP
    # dataset = iHESP_SST("E:\MarineDatasets\iHESP_SST")
    # # dataset.partition()
    # trainone,maskone = dataset.getitem(0, is_norm=True)
    # testone,maskone = dataset.getitem(0, is_norm=False)
    # print(testone.max())
    # print(trainone.max())
    # diff = testone[maskone]- dataset.inverse_normalize(trainone)[maskone]
    # print("diff max ",diff.max())
    #
    # # 验证一下分区能不能恢复
    # recon_one = dataset.reconstruct_from_blocks(testone)
    # file = nc.Dataset(
    #     os.path.join(dataset.base_path,
    #                  'B.E.13.B1850C5.ne120_t12.sehires38.003.sunway_02.pop.h.nday1.SST.046001-046912.nc'),
    #     'r')
    # sst = file.variables['SST'][0,...]
    # file.close()
    # original_dataone = torch.from_numpy(sst.filled(-1))
    # original_maskone = (original_dataone != -1)
    # diff_recon = original_dataone[original_maskone] - recon_one[original_maskone]
    # print(diff_recon.max())

    # 测试train dataloader
    train_dataset = iHESP_SST_Train()
    dataloader  = DataLoader(train_dataset,  batch_size=1, shuffle=True)

    for data, test, mask in dataloader:
        print(data.shape)
        print(test.dtype)
        print(mask.shape)
        break

