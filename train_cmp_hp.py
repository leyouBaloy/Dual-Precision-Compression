import argparse
import logging
import time
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from Dataset.HYCOM import HYCOM, HYCOM_Train
from module import LPModel, HPModel
import utils
import os
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Train HYCOM HP model")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--log_file', type=str, required=True, help='Log file name')
    parser.add_argument('--num_embeddings', type=int, default=32, help='Number of embeddings')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Dimension of embeddings')
    parser.add_argument('--n_component', type=int, default=2, help='Number of components in HPModel')
    parser.add_argument('--n_prob_sample', type=int, default=5, help='Number of probability samples')
    parser.add_argument('--num_hiddens', type=int, default=64, help='Number of hidden units in models')
    parser.add_argument('--num_residual_layers', type=int, default=5, help='Number of residual layers in models')
    parser.add_argument('--num_residual_hiddens', type=int, default=32, help='Number of residual hidden units in models')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training (e.g., "cuda:0" or "cpu")')
    parser.add_argument('--time', type=int, default=16, help='Time parameter for HYCOM_Train')
    parser.add_argument('--lp_model_path', type=str, required=True, help='Path to load pretrained LPModel')
    parser.add_argument('--hp_model_path', type=str, required=True, help='Path to load pretrained HPModel')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval (in epochs) to save model checkpoints')
    parser.add_argument('--start_epoch', type=int, default=321, help='Starting epoch number for saving models')
    return parser.parse_args()

def main():
    args = parse_args()

    # 设置日志记录器
    logging.basicConfig(filename=args.log_file, level=logging.INFO, format='%(message)s')

    logging.info("loading data")
    dataset = HYCOM(args.data_path)
    train_dataset = HYCOM_Train(args.data_path, time=args.time)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    logging.info("len dataloader: %d", len(dataloader))
    logging.info("loading data completed")

    logging.info("create model")
    lpmodel = LPModel(num_hiddens=args.num_hiddens, num_residual_layers=args.num_residual_layers,
                      num_residual_hiddens=args.num_residual_hiddens, num_embeddings=args.num_embeddings,
                      embedding_dim=args.embedding_dim, commitment_cost=0.25, decay=0.99).to(args.device)
    hpmodel = HPModel(num_hiddens=args.num_hiddens, num_residual_layers=args.num_residual_layers,
                      num_residual_hiddens=args.num_residual_hiddens, num_embeddings=args.num_embeddings,
                      embedding_dim=args.embedding_dim, commitment_cost=0.25, decay=0.99,
                      n_component=args.n_component).to(args.device)

    # 加载预训练的模型权重
    lp_save_dict = torch.load(args.lp_model_path)
    lpmodel.load_state_dict(lp_save_dict['model'])

    hp_save_dict = torch.load(args.hp_model_path)
    hpmodel.load_state_dict(hp_save_dict['model'])

    optimizer = torch.optim.Adam(hpmodel.parameters(), lr=args.learning_rate, amsgrad=True)
    logging.info("create model completed")

    logging.info("start training")
    lpmodel.eval()
    hpmodel.train()

    min_loss = float('inf')

    for ep in range(args.epochs):
        all_losses = []
        recon_losses = []
        prob_losses = []
        start_time = time.time()

        for idx, (data, test, mask) in enumerate(dataloader):
            try:
                data = data.to(args.device)
                test = test.to(args.device)
                with torch.no_grad():
                    vq_loss, data_recon, perplexity, quantized = lpmodel(data)
                quantized = quantized.detach()  # 切断梯度，使损失函数只与HPModel有关
                means, stds, weights = hpmodel(quantized)
                recon_error = F.mse_loss(data_recon, data)
                # 计算残差
                denorm_data = dataset.inverse_normalize(data_recon)
                res_data = denorm_data - test
                # 量化
                res_quantized = utils.uniform_quantization(res_data, quan_num=1000, min_val=-1.0, max_val=1.0)
                # 计算概率损失
                probloss = utils.estimate_prob_and_loss(means, stds, weights, res_quantized, quan_num=1000,
                                                        skip=args.n_prob_sample, HW_size=256)
                loss = probloss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                all_losses.append(loss.item())
                recon_losses.append(recon_error.item())
                prob_losses.append(probloss.item())
            except Exception as e:
                print(e)

        end_time = time.time()
        logging.info("time: %f", end_time - start_time)

        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y年%m月%d日 %H时%M分%S秒")
        logging.info(formatted_time)
        print(
            f"epoch {ep} finish, all loss {np.mean(all_losses)}, recon loss {np.mean(recon_losses)}, prob loss {np.mean(prob_losses)}")
        logging.info(
            f"epoch {ep} finish, all loss {np.mean(all_losses)}, recon loss {np.mean(recon_losses)}, prob loss {np.mean(prob_losses)}")

        # 每隔指定的 epoch 保存一次模型
        if (ep + args.start_epoch) % args.save_interval == 0:
            logging.info(f"Saving model at epoch {ep + args.start_epoch}...")
            utils.direct_save_model(ep=ep + args.start_epoch, model=hpmodel, model_name=args.model_path)

if __name__ == "__main__":
    main()
