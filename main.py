# _*_ coding: utf-8 _*_
# This file is created by C. Zhang for personal use.
# @Time         : 25/03/2024 19:47
# @Author       : Chuanting Zhang
# @File         : main.py
# @Affiliation  : Shandong University
import sys
import torch
import numpy as np
import random
import copy
from sklearn import metrics
import pandas as pd
import h5py
import time
import tqdm

sys.path.append('../')
from fedgcc.utils.options import args_parser
from fedgcc.utils.data import process_isolated
from fedgcc.nodes.server import ServerNode
from fedgcc.nodes.client import ClientNode
from fedgcc.compressor import topk, dgc, randomk, terngrad
from fedgcc.utils.models import MLP, TransformerModel
from fedgcc.utils.logger import AverageMeter

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    args = args_parser()
    data = pd.read_csv('./data/{:}_traffic_nid.csv'.format(args.file), index_col=0)
    data.index = pd.to_datetime(data.index)

    # seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    log_id = '{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},'.format(
        args.file, args.alg, args.strategy, args.ratio, args.mu, args.local_epoch, args.compressed, args.epsilon,
        args.exp, args.close_size, args.lr, args.clip, args.seed, args.tkv, args.thv, args.comp, args.hidden_dim
    )
    # print(log_id)

    device = 'cuda' if args.gpu else 'cpu'
    selected_cells = data.columns.values.tolist()
    n_samples = data.shape[0]
    n_test = 24 * args.granularity * (args.test_days + args.val_days)

    train_sample = data.iloc[:-n_test]
    mean = train_sample.mean()
    std = train_sample.std()
    df_scaled = (data - mean) / std

    # print('Begin processing')
    train, val, test = process_isolated(args, df_scaled)
    # print('Finished processing')

    model = MLP(args) if args.model.lower() == 'mlp' else TransformerModel(ninp=args.close_size)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.comp.lower() == 'topk':
        comp = topk.TopKCompressor(compress_ratio=args.ratio)
    elif args.comp.lower() == 'randomk':
        comp = randomk.RandomKCompressor(compress_ratio=args.ratio)
    elif args.comp.lower() == 'terngrad':
        comp = terngrad.TernGradCompressor()
    elif args.comp.lower() == 'dgc':
        comp = dgc.DGCCompressor(compress_ratio=args.ratio)
    else:
        comp = None

    # initialize server and clients
    clients = dict()
    server = ServerNode(args, train[selected_cells[0]], copy.deepcopy(model), comp)

    for cell in selected_cells:
        clients[cell] = ClientNode(args, train[cell], test[cell], copy.deepcopy(model), comp)

    losses = AverageMeter()
    communications = AverageMeter()
    acc = AverageMeter()
    elapsed_time = AverageMeter()
    history = []
    bs = 223 if args.file.lower() == 'trentino' else 88

    for epoch in tqdm.tqdm(range(args.epochs)):
        epoch_time = 0
        m = max(int(args.frac * bs), 1)
        cell_idx = random.sample(selected_cells, m)

        round_loss = []
        round_grads = []
        bits = 0
        begin_time = time.time()
        for cell in cell_idx:
            # client locally update its model parameters by using SGD
            client_loss, client_grads, up_bits = clients[cell].batch_update(server, epoch)
            # print('cell: {:}, up bits: {:}'.format(cell, up_bits))
            round_loss.append(client_loss)
            round_grads.append(np.concatenate(client_grads))
            bits += up_bits

        epoch_clients = [clients[cell] for cell in cell_idx]

        train_time = time.time()
        epoch_time += (train_time - begin_time)

        if args.alg.lower() == 'fedgcc':
            epoch_clients = server.agg_personalized(round_grads, clients, cell_idx, epoch)
            personalized_time = time.time() - train_time
            extra_time = personalized_time / m  # time could be reduced m times by parallelization
            epoch_time += extra_time

        server_opt = time.time()
        down_bits = server.agg_global(epoch_clients, epoch)
        # print('down bits: {:}'.format(down_bits))
        bits += down_bits
        epoch_time += (time.time() - server_opt)

        avg_loss = sum(round_loss) / len(round_loss)

        eval_loss = [1.0]

        eval_avg_loss = sum(eval_loss) / len(eval_loss)
        acc.update(eval_avg_loss)
        losses.update(avg_loss)
        communications.update(bits)
        elapsed_time.update(epoch_time)
        history.append((epoch + 1, losses.val, losses.avg,
                        acc.val, acc.avg,
                        communications.val*4/1000000, communications.sum*4/1000000,
                        elapsed_time.val, elapsed_time.sum))

    df_log = pd.DataFrame(history,
                          columns=['epoch', 'loss val', 'loss avg',
                                   'acc val', 'acc avg',
                                   'bit val', 'bit sum',
                                   'time val', 'time sum'])

    # Test model accuracy
    pred, truth = {}, {}
    for cell in selected_cells:
        cell_loss, pred_, truth_ = clients[cell].inference(server)
        pred[cell] = pred_
        truth[cell] = truth_

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)

    rmse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel()) ** 0.5
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    r2 = metrics.r2_score(df_pred.values.ravel(), df_truth.values.ravel())
    corr = np.corrcoef(df_pred.values.ravel(), df_truth.values.ravel())[0, 1]
    # nrmse = NRMSE / len(selected_cells)
    print(log_id + '{:.4f},{:.4f},{:.4f},{:.4f}'.format(rmse, mae, r2, corr))

    file_name = 'ratio-{:}-{:}-{:}-{:}-{:}-{:}-{:}-{:}-{:}-{:}-{:}-{:}-{:}-{:}-{:}'.format(
        args.file, args.alg, args.strategy, args.ratio, args.mu, args.local_epoch, args.compressed, args.epsilon,
        args.exp, args.close_size, args.lr, args.clip, args.seed, args.tkv, args.thv
    )

    # df_log.to_csv(args.directory + file_name + '.csv', index=False)
    f = h5py.File(args.directory + file_name + '.h5', 'w')
    f.create_dataset(name='pred', data=df_pred)
    f.create_dataset(name='truth', data=df_truth)
    f.create_dataset(name='log', data=df_log)
    f.close()
