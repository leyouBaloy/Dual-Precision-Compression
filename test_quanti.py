import torch

class UniformQuantTorchMappedToInt16:
    def __init__(self, levels=1000) -> None:
        self.levels = levels  # 设置为1000个量化级别

    def quant(self, input, min_val=-1.0, max_val=1.0):
        input = torch.clamp(input, min_val, max_val)  # 限制输入值的范围
        scale = (max_val - min_val) / (self.levels - 1)
        q = torch.clamp(torch.round((input - min_val) / scale), 0, self.levels - 1)
        return q, scale  # 在这里q已经映射到了[0, 999]

    def dequant(self, q, scale, min_val=-1.0):
        return (scale * q + min_val).float()

    def fakequant(self, input, min_val=-1.0, max_val=1.0):
        q, scale = self.quant(input, min_val, max_val)
        fakequant_input = self.dequant(q, scale, min_val)
        return fakequant_input

# 使用PyTorch张量
test_input_torch = torch.linspace(-10.0, 10.0, 10000)  # 生成测试数据

# 实例化量化对象
quant_torch_mapped_to_int16 = UniformQuantTorchMappedToInt16(levels=1000)

# 进行伪量化
fakequant_input_torch_mapped_int16 = quant_torch_mapped_to_int16.fakequant(test_input_torch)

# 计算误差
error_torch_mapped_int16 = torch.abs(test_input_torch - fakequant_input_torch_mapped_int16)

# 检验最大误差
max_error_torch_mapped_int16 = torch.max(error_torch_mapped_int16)
max_error_torch_mapped_int16
