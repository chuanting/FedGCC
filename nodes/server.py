# _*_ coding: utf-8 _*_
import sys
sys.path.append('../../')
from fedgcc.utils.sgd import SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.special import softmax
import pandas as pd
from scipy import linalg
import numpy as np
import copy
import torch


class ServerNode(object):
    def __init__(self, args, train, model, comp=None):
        super(ServerNode, self).__init__()
        self.args = args
        self.device = 'cuda' if args.gpu else 'cpu'
        self.model = model.to(self.device)
        self.opt = SGD(self.model.parameters(), lr=args.lr, momentum=0)
        self.sparse = comp
        data = list(zip(*train))
        loader = DataLoader(data, shuffle=True, batch_size=self.args.local_bs)
        xc, y = next(iter(loader))
        xc = xc.float().to(self.device)
        y = y.float().to(self.device)
        self.opt.zero_grad(set_to_none=False)
        if self.args.model.lower() == 'transformer':
            xc = torch.permute(xc, [1, 0, 2])
            y = y[None, :, :]
            xc_y = torch.concat([xc, y], dim=0)
            xc = xc_y[:-1, :, :]
            y = xc_y[1:, :, :]
        pred = self.model(xc)
        loss = F.mse_loss(y, pred)
        loss.backward()

        self.grad_acc = None

    def enable_grad(self, dataset):
        data = list(zip(*dataset))
        loader = DataLoader(data, shuffle=True, batch_size=self.args.local_bs)
        xc, y = next(iter(loader))
        xc = xc.float().to(self.device)
        y = y.float().to(self.device)
        self.opt.zero_grad(set_to_none=False)
        if self.args.model.lower() == 'transformer':
            xc = torch.permute(xc, [1, 0, 2])
            y = y[None, :, :]
            xc_y = torch.concat([xc, y], dim=0)
            xc = xc_y[:-1, :, :]
            y = xc_y[1:, :, :]
        pred = self.model(xc)
        loss = F.mse_loss(y, pred)
        loss.backward()
        # self.opt.zero_grad()

    def agg_global(self, local_clients, epoch):
        # print('Start aggregation')
        self.model.train()
        self.opt.zero_grad(set_to_none=False)
        n = len(local_clients)
        bits = 0
        # the following for loop calculates the average of local gradients w or w/o compression
        for client in local_clients:
            for server_param, client_param, memory_param in zip(self.model.parameters(),
                                                                client.model.parameters(),
                                                                client.memory.parameters()):
                param_diff = (server_param.data - client_param.data) / n
                with torch.no_grad():
                    if self.args.compressed:
                        gradient = param_diff + memory_param / n
                        # gradient = param_diff
                        gradient_compressed, ctx = self.sparse.compress(gradient, 'seed')
                        gradient_decompressed = self.sparse.decompress(gradient_compressed, ctx)
                        server_param.grad.data.add_(gradient_decompressed)
                        # broadcast aggregated gradients
                        bits += np.prod(gradient_compressed[0].size())
                        bits += np.prod(gradient_compressed[1].size())
                        # bits += sys.getsizeof(ctx)
                    else:
                        server_param.grad.data.add_(param_diff)
                        bits += np.prod(param_diff.size())

        bits /= len(local_clients)

        temp_bits = 0
        for client in local_clients:
            for server_param, client_param, gt_param, memory_param in zip(self.model.parameters(),
                                                                          client.model.parameters(),
                                                                          client.gt.parameters(),
                                                                          client.memory.parameters()):
                with torch.no_grad():
                    gt_param.data += (server_param.data -
                                      server_param.grad.data -
                                      client_param.data) / (self.args.lr * self.args.local_epoch)
                    # model size, global aggregated gradients
                    temp_bits += np.prod(server_param.data.size())
                    if self.args.compressed:
                        memory_param.data += server_param.data - client_param.data - server_param.grad.data

        temp_bits /= 10
        bits += temp_bits
        self.opt.step(apply_lr=False, scale=self.args.epsilon)

        return bits

    def agg_avg(self, local_clients):
        # print('Start aggregation')
        self.opt.zero_grad()
        n = len(local_clients)
        self.model.train()
        # the following for loop calculates the average of local gradients w or w/o compression
        for client in local_clients:
            for server_param, client_param in zip(self.model.parameters(), client.model.parameters()):
                param_diff = (server_param.data - client_param.data) / n
                if self.args.compressed:
                    gradient_compressed, ctx = self.sparse.compress(param_diff, 'seed')
                    gradient_decompressed = self.sparse.decompress(gradient_compressed, ctx)
                    server_param.grad.data.add_(gradient_decompressed)
                else:
                    server_param.grad.data.add_(param_diff)
        self.opt.step(apply_lr=False, scale=self.args.epsilon)

    def agg_personalized(self, round_grads, clients, cell_idx, epoch):
        df_gradient = pd.DataFrame(round_grads).T
        df_gradient.columns = cell_idx
        # df_gradient.to_csv('epoch_{:}.csv'.format(epoch), index=False)
        round_clients = []
        df_corr = df_gradient.corr()
        if self.args.strategy.lower() == 'th':
            for cell in cell_idx:
                corr = df_corr[cell]
                target_cell = corr.loc[corr.abs() > self.args.thv].index
                ws = [clients[cell].get_model() for cell in target_cell]
                ws_grouped = self.average_weights(ws)
                clients[cell].update_model(ws_grouped)
                round_clients.append(clients[cell])
        elif self.args.strategy.lower() == 'kb':
            for cell in cell_idx:
                corr = df_corr[cell].sort_values(ascending=False)
                target_cell = corr[:self.args.tkv].index
                ws = [clients[cell].get_model() for cell in target_cell]
                ws_grouped = self.average_weights(ws)
                clients[cell].update_model(ws_grouped)
                round_clients.append(clients[cell])
        elif self.args.strategy.lower() == 'pr':
            corr_soft = softmax(df_corr.values, axis=1)
            for i, cell_src in enumerate(cell_idx):
                ws = [(clients[cell_dst].get_model(), corr_soft[i, j]) for j, cell_dst in enumerate(cell_idx)]
                ws_grouped = self.average_weights_pearson(ws)
                clients[cell_src].update_model(ws_grouped)
                round_clients.append(clients[cell_src])
        else:
            round_clients = [clients[cell] for cell in cell_idx]
        return round_clients

    def average_weights(self, w):
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg

    def average_weights_pearson(self, w):
        params, weights = [], []
        for param, score in w:
            params.append(param)
            weights.append(score)
        w_avg = dict()
        for name, param in params[0].items():
            w_avg[name] = torch.zeros_like(param)
        for key in w_avg.keys():
            for i in range(0, len(params)):
                w_avg[key] += params[i][key] * weights[i]
        return w_avg

    def average_weights_att(self, w_clients):
        w_server = self.model.state_dict()
        w_next = copy.deepcopy(w_server)
        att = {}
        for k in w_server.keys():
            w_next[k] = torch.zeros_like(w_server[k])
            att[k] = torch.zeros(len(w_clients))

        for k in w_next.keys():
            for i in range(0, len(w_clients)):
                att[k][i] = torch.from_numpy(np.array(linalg.norm(w_server[k].cpu() - w_clients[i][k].cpu())))

        for k in w_next.keys():
            att[k] = F.softmax(att[k], dim=0)

        for k in w_next.keys():
            att_weight = torch.zeros_like(w_server[k])
            for i in range(0, len(w_clients)):
                att_weight += torch.mul(w_server[k] - w_clients[i][k], att[k][i])

            w_next[k] = w_server[k] - torch.mul(att_weight, self.args.epsilon)

        return w_next

    def average_weights_dual_att(self, w_clients, warm_server, epsilon=1.0, rho=0.1):
        w_server = self.model.state_dict()
        w_next = copy.deepcopy(w_server)
        att = {}
        att_warm = {}
        for k in w_server.keys():
            w_next[k] = torch.zeros_like(w_server[k])
            att[k] = torch.zeros(len(w_clients))

        for k in w_next.keys():
            for i in range(0, len(w_clients)):
                att[k][i] = torch.from_numpy(np.array(linalg.norm(w_server[k].cpu() - w_clients[i][k].cpu())))
            sw_diff = w_server[k].cpu() - warm_server[k].cpu()
            att_warm[k] = torch.FloatTensor(np.array(linalg.norm(sw_diff)))

        warm_tensor = torch.FloatTensor([v for k, v in att_warm.items()])
        layer_w = F.softmax(warm_tensor, dim=0)

        for i, k in enumerate(w_next.keys()):
            att[k] = F.softmax(att[k], dim=0)
            att_warm[k] = layer_w[i]

        for k in w_next.keys():
            att_weight = torch.zeros_like(w_server[k])
            for i in range(0, len(w_clients)):
                att_weight += torch.mul(w_server[k] - w_clients[i][k], att[k][i])

            att_weight += torch.mul(w_server[k] - warm_server[k], rho * att_warm[k])

            w_next[k] = w_server[k] - torch.mul(att_weight, epsilon)
        return w_next

    def update_weights(self, w):
        self.model.load_state_dict(w)

    def get_weights(self):
        return self.model.state_dict()
