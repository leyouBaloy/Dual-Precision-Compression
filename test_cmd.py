import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import torchac.torchac
import matplotlib.patches as patches
from Dataset.HYCOM import HYCOM
from module import LPModel, HPModel
import utils
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HYCOM model performance")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--lp_model_path', type=str, required=True, help='Path to load pretrained LPModel')
    parser.add_argument('--hp_model_path', type=str, required=True, help='Path to load pretrained HPModel')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for evaluation (e.g., "cuda:0" or "cpu")')
    parser.add_argument('--num_embeddings', type=int, default=32, help='Number of embeddings')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Dimension of embeddings')
    parser.add_argument('--num_hiddens', type=int, default=64, help='Number of hidden units in models')
    parser.add_argument('--num_residual_layers', type=int, default=5, help='Number of residual layers in models')
    parser.add_argument('--num_residual_hiddens', type=int, default=32, help='Number of residual hidden units in models')
    parser.add_argument('--n_component', type=int, default=2, help='Number of components in HPModel')
    parser.add_argument('--quan_num', type=int, default=1000, help='Number of quantization levels')
    parser.add_argument('--hw_size', type=int, default=240, help='Height and width size for quantization')
    parser.add_argument('--block_index', type=int, default=0, help='Index of the block to evaluate')
    return parser.parse_args()

def main():
    args = parse_args()

    # 加载模型
    print("create model")
    device = args.device
    lpmodel = LPModel(num_hiddens=args.num_hiddens, num_residual_layers=args.num_residual_layers,
                      num_residual_hiddens=args.num_residual_hiddens, num_embeddings=args.num_embeddings,
                      embedding_dim=args.embedding_dim, commitment_cost=0.25, decay=0.99).to(device)
    hpmodel = HPModel(num_hiddens=args.num_hiddens, num_residual_layers=args.num_residual_layers,
                      num_residual_hiddens=args.num_residual_hiddens, num_embeddings=args.num_embeddings,
                      embedding_dim=args.embedding_dim, commitment_cost=0.25, decay=0.99,
                      n_component=args.n_component).to(device)

    lp_save_dict = torch.load(args.lp_model_path)
    hp_save_dict = torch.load(args.hp_model_path)

    print("lp model train epoch: ", lp_save_dict['epoch'])
    print("hp model train epoch: ", hp_save_dict['epoch'])

    lpmodel.load_state_dict(lp_save_dict['model'])
    hpmodel.load_state_dict(hp_save_dict['model'])

    print("create model completed")

    # 加载数据
    print("loading data...")
    dataset = HYCOM(args.data_path)
    data, mask = dataset.getitem(index=args.block_index, is_norm=True)
    test, mask = dataset.getitem(index=args.block_index, is_norm=False)

    data = data.unsqueeze(1).to(device)
    mask = mask.unsqueeze(1)
    test = test.unsqueeze(1)

    print(data.shape)
    print("load data finished.")

    # 评估
    print("start eval...")
    lpmodel.eval()
    hpmodel.eval()
    with torch.no_grad():
        vq_loss, _data_recon, perplexity, quantized = lpmodel(data)
        means, stds, weights = hpmodel(quantized)
        print("perplexity: ", perplexity.item())
        data_recon = dataset.inverse_normalize(_data_recon)
        data_recon[data_recon > dataset.max_val] = dataset.max_val
        data_recon[data_recon < dataset.min_val] = dataset.min_val
        data_recon_flat = dataset.reconstruct_from_blocks(data_recon.squeeze().cpu())
    print("eval finished.")

    real_res = data_recon.cpu() - test
    calc_res = real_res.clone()
    calc_res[real_res > 1.0] = 1.0
    calc_res[real_res < -1.0] = -1.0
    res_quantized = utils.uniform_quantization(calc_res.cpu(), quan_num=args.quan_num, min_val=-1.0, max_val=1.0)
    res_dequantized = utils.uniform_dequantization(res_quantized, quan_num=args.quan_num, min_val=-1.0, max_val=1.0)
    res_abs_quantized_diff = torch.abs(res_dequantized.cpu() - calc_res)
    print(f"量化平均绝对误差MAE: {torch.mean(res_abs_quantized_diff).item():.6f}")
    print(f"量化最大误差MAX: {torch.max(res_abs_quantized_diff).item():.6f}")

    print("======================")
    real_res_flat = dataset.reconstruct_from_blocks(real_res.squeeze().cpu())
    mask_flat = dataset.reconstruct_from_blocks(mask.squeeze().cpu())
    mae = torch.mean(torch.abs(real_res_flat[mask_flat]))
    print(f"LP平均绝对误差(MAE): {mae:.6f}")
    print(f"LP最大误差(MAX error): {torch.abs(real_res_flat[mask_flat]).max():.6f}")

    data_range = dataset.max_val - dataset.min_val
    test_flat = dataset.reconstruct_from_blocks(test.squeeze().cpu())
    relative_error = torch.abs(real_res_flat[mask_flat] / data_range)
    rel = torch.mean(relative_error)
    print(f"LP平均相对误差(REL): {rel.item():.6f}")
    print(f"LP最大相对误差(REL max): {torch.abs(real_res_flat[mask_flat]).max() / data_range:.6f}")

    mse = torch.mean(torch.square(real_res_flat[mask_flat]))
    psnr = 10 * torch.log10(torch.square(data_range) / mse) if mse > 0 else float('inf')
    print(f"LP峰值信噪比 (PSNR): {psnr:.6f} dB")

    print(f"LP压缩率: {(16*3120*4320*32)/(16*234*64*64*np.log2(32)+32*32*32):.6f}")
    print(f"LP bpp: {(16*234*64*64*np.log2(32)+32*32*32)/(16*3120*4320):.6f}")

    mask_out1 = torch.zeros_like(mask, dtype=torch.bool)
    mask_out1[mask & ((real_res > 1.0) | (real_res < -1.0))] = True
    print(f"[-1,1]之外数据比例: {(torch.sum(mask_out1[:, :, 8:-8, 8:-8, ...]) / len(real_res_flat[mask_flat])).item():.6f}")

    print("means.shape", means.shape)

    compressed_bits_lst = []
    hp_datas = []
    start_time = time.time()
    for idx in range(means.shape[0]):
        compressed_bits, hp_data_one = utils.get_compressed_bits(res_quantized[idx, :, 8:-8, 8:-8, ...].unsqueeze(0).cpu(),
                                                                 mask[idx, :, 8:-8, 8:-8, ...].unsqueeze(0).cpu(),
                                                                 mask_out1[idx, :, 8:-8, 8:-8, ...].unsqueeze(0).cpu(),
                                                                 means[idx, 8:-8, 8:-8, ...].unsqueeze(0).cpu(),
                                                                 stds[idx, 8:-8, 8:-8, ...].unsqueeze(0).cpu(),
                                                                 weights[idx, 8:-8, 8:-8, ...].unsqueeze(0).cpu(),
                                                                 quan_num=args.quan_num, HW_size=args.hw_size)
        compressed_bits_lst.append(compressed_bits)
        hp_datas.append(hp_data_one.squeeze(0))

    end_time = time.time()
    print("======================")
    print(f"HP压缩率: {(3120 * 4320 * 32) / (np.sum(compressed_bits_lst) + (234 * 64 * 64 * np.log2(32) + 32 * 32 * 32)):.6f}")
    print(f"耗时 {end_time - start_time}")

    body_mask = (mask ^ mask_out1)
    body_mask_depad = body_mask[:, :, 8:-8, 8:-8].squeeze().unsqueeze(1)
    body_mask_flat = dataset.reconstruct_from_blocks(body_mask.squeeze().cpu())
    mask_out1_flat = dataset.reconstruct_from_blocks(mask_out1.squeeze().cpu())
    res_dequantized_depad = res_dequantized[:, :, 8:-8, 8:-8].squeeze().unsqueeze(1)

    hp_res_data = torch.stack(hp_datas).squeeze().unsqueeze(1)
    mask_out1_depad = mask_out1[:, :, 8:-8, 8:-8].squeeze().unsqueeze(1)
    real_res_depad = real_res[:, :, 8:-8, 8:-8].squeeze().unsqueeze(1)
    hp_res_data[mask_out1_depad] = torch.round(real_res_depad[mask_out1_depad], decimals=3)

    data_recon_depad = data_recon.cpu()[:, :, 8:-8, 8:-8].squeeze().unsqueeze(1)
    hp_data = data_recon_depad - hp_res_data
    hp_data_pad = torch.zeros(res_dequantized.shape, dtype=torch.float32)
    hp_data_pad[:, :, 8:-8, 8:-8] = hp_data
    hp_data_flat = dataset.reconstruct_from_blocks(hp_data_pad.squeeze().cpu())

    mae_error = torch.mean(torch.abs(hp_data_flat[mask_flat] - test_flat[mask_flat]))
    print(f"HP平均绝对误差 (MAE): {mae_error:.6f}")

    max_error = torch.max(torch.abs(hp_data_flat[mask_flat] - test_flat[mask_flat]))
    print(f"HP最大误差 (Max Error): {max_error:.6f}")

    mre_error = torch.mean(np.abs((hp_data_flat[mask_flat] - test_flat[mask_flat]) / data_range))
    print(f"HP平均相对误差 (REL_mean): {mre_error:.6f}")

    mre_max_error = torch.max(torch.abs((hp_data_flat[mask_flat] - test_flat[mask_flat]) / data_range))
    print(f"HP最大相对误差 (REL_max): {mre_max_error:.6f}")

    signal_power = torch.square(data_range)
    noise_power = torch.mean(np.square(hp_data_flat[mask_flat] - test_flat[mask_flat]))
    psnr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    print(f"HP峰值信噪比 (PSNR): {psnr:.6f} dB")

    print(f"**HP压缩率**: {(16 * 3120 * 4320 * 32) / (np.sum(compressed_bits_lst) * 16 + (16 * 234 * 64 * 64 * np.log2(32) + 32 * 32 * 32)):.6f}")

    print(f"比特率 (BR): {(np.sum(compressed_bits_lst) * 16 + (16 * 234 * 64 * 64 * np.log2(32) + 32 * 32 * 32)) / (16 * 3120 * 4320):.6f} bits per element")

if __name__ == "__main__":
    main()
