import sys
import copy
sys.path.append('../')
from fedgcc.utils.sgd import SGD
import torch
import numpy as np
from torch.utils.data import DataLoader


class ClientNode(object):
    def __init__(self, args, train, test, model, comp):
        super(ClientNode, self).__init__()
        self.args = args
        self.train = train
        self.test = test
        self.train_loader = self.process_data(train, shuffle=True)
        self.test_loader = self.process_data(test, shuffle=False)
        self.loader = iter(self.process_data(train, shuffle=True))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.model = model.to(self.device)
        self.criterion = torch.nn.MSELoss().to(self.device)
        self.gt = self.copy_model()
        self.memory = self.copy_model()
        self.global_rounds = args.epochs
        self.sparse = comp

    def process_data(self, dataset, shuffle):
        data = list(zip(*dataset))
        loader = DataLoader(data, shuffle=shuffle, batch_size=self.args.local_bs)
        return loader

    def update_model(self, weight):
        self.model.load_state_dict(weight)

    def get_model(self):
        return self.model.state_dict()

    def batch_update(self, server, com_rounds):
        if com_rounds >= int(self.args.epochs * 0.75):
            lr = self.args.lr / 100.0
        elif com_rounds >= int(self.args.epochs / 2):
            lr = self.args.lr / 10.0
        else:
            lr = self.args.lr

        self.model.load_state_dict(server.model.state_dict())
        self.model.train()
        opt = SGD(self.model.parameters(), lr=lr, momentum=self.args.momentum, mu=self.args.mu)

        # for p in opt.param_groups:
        #     print('LR: {:.4f}'.format(p['lr']))
        epoch_loss = []
        for epoch in range(self.args.local_epoch):
            try:
                xc, y = next(self.loader)
            except StopIteration:
                self.loader = iter(self.process_data(self.train, shuffle=True))
                xc, y = next(self.loader)
            xc = xc.float().to(self.device)
            y = y.float().to(self.device)
            opt.zero_grad()
            if self.args.model.lower() == 'transformer':
                xc = torch.permute(xc, [1, 0, 2])
                y = y[None, :, :]
                xc_y = torch.concat([xc, y], dim=0)
                xc = xc_y[:-1, :, :]
                y = xc_y[1:, :, :]
            pred = self.model(xc)
            loss = self.criterion(y, pred)
            loss.backward()
            if self.args.alg.lower() == 'fedgcc':
                for client_param, gt_param in zip(self.model.parameters(),
                                                  self.gt.parameters()):
                    client_param.grad.data -= gt_param.data
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip, norm_type=2)
            opt.step(apply_lr=True, apply_momentum=True)
            epoch_loss.append(loss.item())

        grads = []
        bits = 0
        for client_param, server_param in zip(self.model.parameters(),
                                              server.model.parameters()):
            gradient = (server_param.data - client_param.data) / self.args.lr

            if self.args.compressed:
                gradient_compressed, ctx = self.sparse.compress(gradient, 'seed')
                gradient_decompressed = self.sparse.decompress(gradient_compressed, ctx)
                grads.append(gradient_decompressed.detach().cpu().numpy().ravel())
                # calculate used memory of non zeros elements
                # values
                bits += np.prod(gradient_compressed[0].size())
                # indices
                bits += np.prod(gradient_compressed[1].size())
            else:
                # non-compressed memory
                bits += np.prod(gradient.size())
                grads.append(gradient.detach().cpu().numpy().ravel())

        return sum(epoch_loss) / len(epoch_loss), grads, bits

    def inference(self, server=None):
        if server is not None:
            self.model.load_state_dict(server.model.state_dict())
        self.model.eval()
        loss = 0.0
        pred_list = []
        truth_list = []
        with torch.no_grad():
            for batch_idx, (xc, y) in enumerate(self.test_loader):
                xc = xc.float().to(self.device)
                y = y.float().to(self.device)
                if self.args.model.lower() == 'transformer':
                    xc = torch.permute(xc, [1, 0, 2])
                    y = y[None, :, :]
                    xc_y = torch.concat([xc, y], dim=0)
                    xc = xc_y[:-1, :, :]
                    y = xc_y[1:, :, :]

                pred = self.model(xc)
                batch_loss = self.criterion(y, pred)
                loss += batch_loss.item()
                if self.args.model.lower() == 'transformer':
                    pred_list.append(pred[-1, :, :].detach().cpu())
                    truth_list.append(y[-1, :, :].detach().cpu())
                else:
                    pred_list.append(pred.detach().cpu())
                    truth_list.append(y.detach().cpu())

            prediction = np.concatenate(pred_list).ravel()
            truth = np.concatenate(truth_list).ravel()
            avg_loss = loss / len(self.test_loader)
        return avg_loss, prediction, truth

    def copy_model(self):
        local_model = copy.deepcopy(self.model)
        for param in local_model.parameters():
            param.data = torch.zeros_like(param.data)
        return local_model
