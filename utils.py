import torch
from torch.distributions import Normal
import numpy as np

import os

# 已验证
def get_estimate_prob_slow(means, stds, coeffs, quan_num=1000, HW_size=512):
    # means, stds, coeffs (batch, HW_size, HW_size, 3)
    # 计算每个像素,每个取值的概率,该算法极慢
    device = means.device
    batch_size = means.shape[0]
    estimate_prob = torch.zeros((batch_size, 1, HW_size, HW_size, quan_num),
                                device=device)  # (batch, 1, H, W, quan_num)
    for b in range(batch_size):
        for h in range(HW_size):
            for w in range(HW_size):
                for idx in range(quan_num):
                    if idx == 0:  # 处理第一个
                        for mean, std, coeff in zip(means[b, h, w, :], stds[b, h, w, :], coeffs[b, h, w, :]):
                            normal_dist = Normal(mean, std)
                            estimate_prob[b, 0, h, w, idx] += coeff * normal_dist.cdf(torch.tensor(0.5))
                    elif idx == quan_num - 1:  # 处理最后一个
                        estimate_prob[b, 0, h, w, idx] = 1
                        for mean, std, coeff in zip(means[b, h, w, :], stds[b, h, w, :], coeffs[b, h, w, :]):
                            normal_dist = Normal(mean, std)
                            estimate_prob[b, 0, h, w, idx] -= coeff * normal_dist.cdf(torch.tensor(idx - 0.5))
                    else:
                        for mean, std, coeff in zip(means[b, h, w, :], stds[b, h, w, :], coeffs[b, h, w, :]):
                            normal_dist = Normal(mean, std)
                            estimate_prob[b, 0, h, w, idx] += coeff * (
                                        normal_dist.cdf(torch.tensor(idx + 0.5)) - normal_dist.cdf(
                                    torch.tensor(idx - 0.5)))
    return estimate_prob


# 已验证
def get_estimate_prob_fast(means, stds, coeffs, quan_num=1000, HW_size=512):
    device = means.device
    batch_size = means.shape[0]
    estimate_prob = torch.zeros((batch_size, 1, HW_size, HW_size, quan_num),
                                device=device)  # (batch, 1, H, W, quan_num)

    # Precompute the range for quantization numbers
    idx_range = torch.arange(start=0.5, end=quan_num - 1, step=1, device=device).float()
    #     print(idx_range)

    # Compute the cdf values for each combination of (mean, std, coeff) across all quantization steps
    cdf_values = Normal(means.unsqueeze(-1), stds.unsqueeze(-1)).cdf(idx_range.view(1, 1, 1, 1, -1))
    #     print(cdf_values.shape)

    # Calculate the difference between adjacent CDF values for each pixel
    cdf_diff = cdf_values[..., 1:] - cdf_values[..., :-1]

    # Compute the final estimate probability by summing over the channels
    estimate_prob[..., 1:-1] = torch.sum(coeffs.unsqueeze(-1) * cdf_diff, dim=3)

    # Handling the first and last probabilities separately
    estimate_prob[..., 0] = torch.sum(coeffs * cdf_values[..., 0], dim=3)
    estimate_prob[..., -1] = 1 - torch.sum(coeffs * cdf_values[..., -1], dim=3)

    return estimate_prob


# 已验证
def get_estimate_prob_sample_slow(means, stds, coeffs, quan_num=1000, skip=2, HW_size=512):
    assert quan_num % skip == 0
    # means, stds, coeffs (batch, HW_size, HW_size, 3)
    # 计算每个像素,每个取值的概率,该算法极慢
    device = means.device
    batch_size = means.shape[0]
    estimate_prob = torch.zeros((batch_size, 1, HW_size, HW_size, quan_num),
                                device=device)  # (batch, 1, H, W, quan_num)
    for b in range(batch_size):
        for h in range(HW_size):
            for w in range(HW_size):
                for idx in range(skip - 1, quan_num, skip):
                    if idx == skip - 1:  # 处理第一个
                        for mean, std, coeff in zip(means[b, h, w, :], stds[b, h, w, :], coeffs[b, h, w, :]):
                            normal_dist = Normal(mean, std)
                            estimate_prob[b, 0, h, w, 0:idx + 1] += coeff * normal_dist.cdf(
                                torch.tensor(skip - 0.5)) / skip
                    elif idx == quan_num - 1:  # 处理最后一个
                        estimate_prob[b, 0, h, w, idx - skip + 1:idx + 1] = 1 / skip
                        for mean, std, coeff in zip(means[b, h, w, :], stds[b, h, w, :], coeffs[b, h, w, :]):
                            normal_dist = Normal(mean, std)
                            estimate_prob[b, 0, h, w, idx - skip + 1:idx + 1] -= coeff * normal_dist.cdf(
                                torch.tensor(idx - (skip - 0.5))) / skip
                    else:
                        for mean, std, coeff in zip(means[b, h, w, :], stds[b, h, w, :], coeffs[b, h, w, :]):
                            normal_dist = Normal(mean, std)
                            estimate_prob[b, 0, h, w, idx - skip + 1:idx + 1] += coeff * (
                                        normal_dist.cdf(torch.tensor(idx + 0.5)) - normal_dist.cdf(
                                    torch.tensor(idx - (skip - 0.5)))) / skip
    return estimate_prob


# 已验证
def get_estimate_prob_sample_fast(means, stds, coeffs, quan_num=1000, skip=2, HW_size=512):
    assert quan_num % skip == 0 and means.dim() == 4 # (1,1,HW_size,HW_size,quan_num)
    device = means.device
    batch_size = means.shape[0]
    # print("means.shape",means.shape)
    # print("分配size",(batch_size, 1, HW_size, HW_size, quan_num))
    estimate_prob = torch.zeros((batch_size, 1, HW_size, HW_size, quan_num),
                                device=device, dtype=torch.float32)  # (batch, 1, H, W, quan_num)

    # Precompute the range for quantization numbers with the specified skip
    idx_range = torch.arange(start=skip - 0.5, end=quan_num, step=skip, device=device).float()

    # Compute the cdf values for each combination of (mean, std, coeff) across all quantization steps
    if torch.isnan(means).any():
        print(means)
    cdf_values = Normal(means.unsqueeze(-1), stds.unsqueeze(-1)).cdf(idx_range.view(1, 1, 1, 1, -1))

    # Calculate the difference between adjacent CDF values for each pixel
    cdf_diff = cdf_values[..., 1:] - cdf_values[..., :-1]

    # print("estimate_prob[..., skip:]",estimate_prob[..., skip:].shape)
    # print("second", (torch.sum(coeffs.unsqueeze(-1) * cdf_diff, dim=3) / skip).repeat_interleave(skip,dim=-1).shape)
    estimate_prob[..., skip:] = (torch.sum(coeffs.unsqueeze(-1) * cdf_diff, dim=3) / skip).repeat_interleave(skip,dim=-1)

    # Handling the first and last probabilities separately
    first_cdf = Normal(means, stds).cdf(torch.tensor(skip - 0.5, device=device))
    estimate_prob[..., 0:skip] = torch.sum(coeffs * first_cdf, dim=3).unsqueeze(-1) / skip
    last_cdf = Normal(means, stds).cdf(torch.tensor(quan_num - skip - 0.5, device=device))
    estimate_prob[..., -skip:] = (1 - torch.sum(coeffs * last_cdf, dim=3)).unsqueeze(-1) / skip

    return estimate_prob



def get_prob_loss_slow(estimate_prob, quantized_tensor, quan_num=1000, HW_size=512):
    device = quantized_tensor.device
    thresholds = torch.linspace(0, quan_num, quan_num + 1, device=device)
    nl_loss = torch.zeros(1, device=device)
    for h in range(HW_size):
        for w in range(HW_size):
            # 先找到对应区间
            bin_index = torch.sum(torch.lt(thresholds, quantized_tensor[0, 0, h, w]), dim=-1)
            # 再计算负对数
            nl_loss -= torch.log(estimate_prob[0, 0, h, w, bin_index])
    return nl_loss / (HW_size * HW_size)


# 已验证
def get_prob_loss_fast(estimate_prob, quantized_tensor, quan_num=1000, HW_size=512):
    assert estimate_prob.dim() == 5, "estimate_prob should be 5-dimensional (1, 1, H, W, 1000)"
    assert quantized_tensor.dim() == 4, "quantized_tensor should be 4-dimensional (1, 1, H, W)"
    # quantized_tensor = quantized_tensor.squeeze(0) # (1,1,240,240)
    device = estimate_prob.device
    # 计算每个像素标签对应的区间索引
    thresholds = torch.linspace(0, quan_num, quan_num + 1, device=device)
    bin_indices = torch.sum(thresholds[:-1] < quantized_tensor.unsqueeze(-1), dim=-1) - 1
    bin_indices = bin_indices.clamp(min=0, max=quan_num - 1)

    # bin_indices匹配estimate_prob
    bin_indices = bin_indices.unsqueeze(-1)  # 形状变为[1, 1, HW_size, HW_size, 1]

    # 使用advanced indexing从estimate_prob中提取对应的概率值
    selected_probs = estimate_prob.gather(-1, bin_indices).squeeze(-1)

    # 计算负对数损失
    nl_loss = -torch.log(selected_probs.clamp(min=1e-6)).mean()  # 加入clamp以防止对数计算中的数值问题
    # nl_loss = -torch.log(selected_probs).sum()
    return nl_loss


# 未验证
def get_prob_loss_fast_savespace(estimate_prob, quantized_tensor, quan_num=1000, HW_size=512):
    assert quantized_tensor.min() == 0
    device = estimate_prob.device
    # 计算每个像素标签对应的区间索引，避免创建大型中间张量
    bin_indices = quantized_tensor.clamp(max=quan_num-1).long()  # 假设quantized_tensor已经是0到quan_num之间的整数

    # bin_indices匹配estimate_prob
    bin_indices = bin_indices.view(1, 1, HW_size, HW_size, 1)  # 调整形状以匹配estimate_prob

    # 使用advanced indexing从estimate_prob中提取对应的概率值
    selected_probs = estimate_prob.gather(-1, bin_indices).squeeze(-1)

    # 计算负对数损失，使用就地操作clamp_()以减少内存占用
    nl_loss = -torch.log(selected_probs.clamp_(min=1e-6)).sum()
    return nl_loss / (HW_size * HW_size)

def estimate_prob_and_loss(means, stds, weights, res_quantized, quan_num=1000, skip=5, HW_size=240):
    total_loss = 0
    batch_size = means.shape[0]
    # print(means.s)

    for i in range(batch_size):
        means_one = means[i,...].unsqueeze(0) # (1, H, W, 3)
        # print("means_one",means_one.shape)
        # print("caculate i-th loss: ", i)
        # if i == 42:
            # print(means_one = means[i,...].unsqueeze(0))
        stds_one = stds[i,...].unsqueeze(0)
        weights_one = weights[i,...].unsqueeze(0)
        res_quantized_one = res_quantized[i,...].unsqueeze(0)

        # 计算每个批次的估计概率
        estimate_prob_one = get_estimate_prob_sample_fast(means_one, stds_one, weights_one, quan_num, skip, HW_size)

        # 计算每个批次的概率损失
        prob_loss_one = get_prob_loss_fast(estimate_prob_one, res_quantized_one, quan_num, HW_size)

        # 累积损失
        total_loss += prob_loss_one

        # # 删除不再需要的变量来节省内存
        # del means_one, stds_one, weights_one, res_quantized_one, estimate_prob_one, prob_loss_one

    # 返回平均损失
    return total_loss / batch_size
def compute_cdf(estimate_prob):
    # 确保estimate_prob的shape正确
    assert estimate_prob.shape[-1] == 1000, "The last dimension of estimate_prob must be 1000."

    # 获取batch, channel, H, W
    batch, channel, H, W = estimate_prob.shape[:-1]

    # 初始化estimate_cdf张量，形状为(batch, channel, H, W, 1001)
    estimate_cdf = torch.zeros((batch, channel, H, W, 1001), dtype=estimate_prob.dtype, device=estimate_prob.device)

    # 计算累积分布
    estimate_cdf[..., :-1] = torch.cumsum(estimate_prob, dim=-1)

    # 设置最后一个元素为1.0
    estimate_cdf[..., -1] = 1.00000

    return estimate_cdf

# 均匀量化
def uniform_quantization(input, quan_num,  min_val, max_val):
    input = torch.clamp(input, min_val, max_val)  # 限制输入值的范围
    scale = (max_val - min_val) / (quan_num - 1)
    q = torch.clamp(torch.round((input - min_val) / scale), 0, quan_num - 1)
    return q.to(torch.int16)


# 线性逆量化
def uniform_dequantization(q, quan_num, min_val, max_val):
    scale = (max_val - min_val) / (quan_num - 1)
    return (scale * q + min_val).float()


# tanh量化,需修改为中点
def tanh_quantization(tensor, quan_num):
    device = tensor.device
    tensor = tensor.cpu()
    quantized_tensor = (quan_num / 2) * (1 * torch.tanh(1.5 * tensor) + 1)
    quantized_tensor = torch.floor(quantized_tensor).to(torch.int16)
    quantized_tensor = torch.clamp(quantized_tensor, min=0, max=quan_num - 1)
    return quantized_tensor.to(device)


# tanh逆量化
def tanh_dequantization(quantized_tensor, quan_num):
    # 将量化张量转换回浮点数，并缩放回[-1, 1]的范围
    dequantized_tensor = (quantized_tensor / (quan_num / 2)) - 1

    # 调整输入以确保它落在arctanh的有效范围内
    # 避免-1和1，因为arctanh在这些点是未定义的
    dequantized_tensor = np.clip(dequantized_tensor, -0.999999, 0.999999)

    # 应用arctanh来反转tanh操作，处理缩放因子
    dequantized_tensor = np.arctanh(dequantized_tensor) / 1.5

    return dequantized_tensor

# 残差编码
def get_compressed_bits(res_quantized, mask, mask_out1, means, stds, weights, quan_num, HW_size):
    import torchac.torchac
    esti_prob = get_estimate_prob_fast(means, stds,
                                             weights, quan_num=quan_num, HW_size=HW_size)
    estimate_cdf = compute_cdf(esti_prob)
    body_mask = (mask^mask_out1) # 只存储有效数据且在[-1,1]范围内,^是异或
    # 筛选有效部分
    estimate_cdf = estimate_cdf[body_mask, ...]
    # 熵编码
    byte_stream = torchac.encode_float_cdf(estimate_cdf, res_quantized[body_mask, ...],
                                           check_input_bounds=False)
    # 计算压缩后的理论大小
    body_bits = len(byte_stream) * 8
    # 计算mask_out1所需的bit
    index_bit = torch.sum(mask_out1) * 16 # 需要存储索引，最大为256*256，每个使用16bit存储
    # 计算超出[-1,1]部分所需的bit
    out1_bit = torch.sum(mask_out1) * 17 # 17位存储即可存储
    num_bits = body_bits+index_bit+out1_bit

    # 模拟解压后的数组
    _decom = torchac.decode_float_cdf(estimate_cdf, byte_stream)
    decom = uniform_dequantization(_decom, quan_num=1000, min_val=-1.0, max_val=1.0)
    decom_res = torch.zeros(res_quantized.shape,dtype=torch.float32)
    decom_res[body_mask, ...] = decom.clone()
    # decom_data[out1_bit] =
    return num_bits,decom_res

def save_model(ep, model, save_dir, model_name):
    save_path = os.path.join(save_dir, model_name)
    save_dict = {
        "epoch": ep,
        "model": model.state_dict()
    }
    torch.save(save_dict, save_path)

def direct_save_model(ep, model, model_name):
    save_path = model_name
    save_dict = {
        "epoch": ep,
        "model": model.state_dict()
    }
    torch.save(save_dict, save_path)

def calc_lp_cr(time, h, w, blocks, z_size, num_embeddings, embedding_dim):
    cr = (time * h * w * 32) / (time * blocks * z_size * z_size * np.log2(num_embeddings) + num_embeddings * embedding_dim * 32)
    bitrate = (time*blocks*z_size*z_size*np.log2(num_embeddings)+num_embeddings*embedding_dim*32)/(time * h * w)
    return cr, bitrate