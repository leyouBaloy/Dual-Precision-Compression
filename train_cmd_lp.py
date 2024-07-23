import argparse
import logging
import time
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from Dataset.HYCOM import HYCOM, HYCOM_Train
from module import LPModel
import utils

def parse_args():
    parser = argparse.ArgumentParser(description="Train HYCOM model")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--log_file', type=str, required=True, help='Log file name')
    parser.add_argument('--model_path', type=str, required=True, help='Model save path')
    parser.add_argument('--num_embeddings', type=int, default=32, help='Number of embeddings')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Dimension of embeddings')
    parser.add_argument('--num_hiddens', type=int, default=64, help='Number of hidden units in LPModel')
    parser.add_argument('--num_residual_layers', type=int, default=5, help='Number of residual layers in LPModel')
    parser.add_argument('--num_residual_hiddens', type=int, default=32, help='Number of residual hidden units in LPModel')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training (e.g., "cuda:0" or "cpu")')
    parser.add_argument('--time', type=int, default=16, help='Time parameter for HYCOM_Train')
    parser.add_argument('--load_model', type=str, help='Path to load pretrained model')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval (in epochs) to save model checkpoints')
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

    if args.load_model:
        lp_save_dict = torch.load(args.load_model)
        lpmodel.load_state_dict(lp_save_dict['model'])

    optimizer = torch.optim.Adam(lpmodel.parameters(), lr=args.learning_rate, amsgrad=True)
    logging.info("create model completed")

    logging.info("start training")
    lpmodel.train()

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
                vq_loss, data_recon, perplexity, quantized = lpmodel(data)
                recon_error = F.mse_loss(data_recon, data)
                loss = vq_loss + recon_error
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                all_losses.append(loss.item())
                recon_losses.append(recon_error.item())
            except Exception as e:
                logging.error(e)

        end_time = time.time()
        logging.info("time: %f", end_time - start_time)

        print(f"epoch {ep} finish, all loss {np.mean(all_losses)}, recon loss {np.mean(recon_losses)}")
        logging.info(f"epoch {ep} finish, all loss {np.mean(all_losses)}, recon loss {np.mean(recon_losses)}")

        if np.mean(all_losses) < min_loss:
            min_loss = np.mean(all_losses)
            logging.info("save model ...")
            utils.direct_save_model(ep=ep, model=lpmodel, model_name=args.model_path)

if __name__ == "__main__":
    main()
