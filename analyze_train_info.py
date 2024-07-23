# import matplotlib.pyplot as plt
#
# # 读取训练日志文件
# with open('saves/ablation/HYCOM_WTB_lp_64_64.log', 'r') as f:
#     log_lines = f.readlines()
#
# # 初始化变量
# epochs = []
# recon_losses = []
#
# # 提取每个epoch的recon loss数值
# for line in log_lines:
#     if 'recon loss' in line:
#         print(line.split(',')[2].split(' '))
#         recon_loss = float(line.split(',')[2].split(' ')[-1])
#         epochs.append(len(recon_losses) + 1)
#         recon_losses.append(recon_loss)
#
# # 绘制可视化图表
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, recon_losses, marker='o', linewidth=2)
# plt.xlabel('Epoch')
# plt.ylabel('Reconstruction Loss')
# plt.title('Reconstruction Loss Over Epochs')
# plt.grid(True)
# plt.savefig("saves/HYCOM_WTB_lp_32_32.jpg")
# plt.show()

import matplotlib.pyplot as plt

# 读取训练日志文件
with open('saves/ablation/HYCOM_WTB_hp_probsample_1.log', 'r') as f:
    log_lines = f.readlines()

# 初始化变量
epochs = []
recon_losses = []

# 提取每个epoch的recon loss数值
for line in log_lines:
    if 'recon loss' in line:
        print(line.split(',')[1].split(' '))
        recon_loss = float(line.split(',')[1].split(' ')[-1])
        epochs.append(len(recon_losses) + 1)
        recon_losses.append(recon_loss)

# 绘制可视化图表
plt.figure(figsize=(10, 6))
plt.plot(epochs, recon_losses, marker='o', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Reconstruction Loss')
plt.title('Reconstruction Loss Over Epochs')
plt.grid(True)
# plt.savefig("saves/HYCOM_WTB_lp_32_32.jpg")
plt.show()