import torch
import math

# 假设的稀疏bool类型tensor
sparse_tensor = torch.tensor([[False, False, True], [False, False, False], [True, False, False]])

# 计算原始tensor所需的Bit数
original_bits = sparse_tensor.numel() * 1  # 每个元素1 bit

# 压缩
non_zero_indices = torch.nonzero(sparse_tensor, as_tuple=False)
# 假设每个索引是以32-bit整数存储（这取决于实际情况，这里只是为了示例）
index_bits = 16
# 计算压缩后所需的Bit数（每个非零元素的索引需要index_bits个Bit）
compressed_bits = non_zero_indices.numel() * index_bits

# 计算压缩率
compression_ratio =  original_bits / compressed_bits

print("原始所需Bit数:", original_bits)
print("压缩后所需Bit数:", compressed_bits)
print("压缩率:", compression_ratio)

print(3758989/(256*256*150))