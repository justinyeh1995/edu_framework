#!/usr/bin/env python
# coding: utf-8
import copy
from tqdm import tqdm
import numpy as np
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
'''
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

'''
# TODO: how to save model?
# - pytorch_lightening有pre-train routine
#　


class Trainer:
    def __init__(self, model, optimizer, device='cpu'):
        self.model = model.to(device)
        # TODO: define here but move to forward
        self.criterion_reg = nn.MSELoss()
        self.criterion_clf = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        # TODO: move to configure_optimizers
        self.device = device
        # TODO: remove

    def fit(self, train_loader, test_loader=None, epoch=1, early_stop=-1, scaler=None):
        history = {
            'train': [],
            'test': []
        }

        best_eval = 9e9
        early_cnt = 0
        best_model_params = copy.deepcopy(self.model.state_dict())

        for ep in tqdm(range(epoch)):
            # print('Epoch:{}'.format(ep+1))

            self.model.train()
            # TODO: remove
            for batch in tqdm(train_loader, leave=False):
                self.optimizer.zero_grad()
                # TODO: remove
                x_dense, x_sparse, objmean, tscnt, label_0 = [b.to(self.device) for b in batch]  # , spcnt
                # TODO: remove
                outputs = self.model(x_dense, x_sparse)
                # TODO: define in forward
                loss = self.criterion_reg(outputs[0], objmean)
                # TODO: define in training_step
                if ep >= 20:
                    loss += self.criterion_reg(outputs[1], tscnt)
                    # TODO: define in training_step
                # if ep >= 30:
                #    loss += self.criterion_reg(outputs[2], spcnt)
                if ep >= 40:
                    loss += self.criterion_clf(outputs[3], label_0)
                    # TODO: define in training_step
                # TODO: find EPOCH parameter in pl.lightening or calculate epoch here

                loss.backward()
                # TODO: remove
                self.optimizer.step()
                # TODO: remove
            train_result, _, _ = self.evaluate(train_loader)
            history['train'].append(train_result)
            if test_loader:
                test_result, _, _ = self.evaluate(test_loader)
                history['test'].append(test_result)
                if ep % 5 == 0 or ep == epoch - 1:
                    print('Epoch:{}'.format(ep + 1))
                    print('\ttest\t' + ' '.join(['{}:{:.3f}'.format(k, v) for k, v in test_result.items()]))

                if test_result['total_loss'] < best_eval:
                    early_cnt = 0
                    best_eval = test_result['total_loss']
                    best_model_params = copy.deepcopy(self.model.state_dict())
                    # print('\tbetter!')

                elif early_stop > 0 and ep >= 40:
                    early_cnt += 1

            if early_stop > 0 and early_cnt >= early_stop and ep >= 40:
                break

        self.model.load_state_dict(best_model_params)

        return history

    def evaluate(self, loader):
        true_list = [[], [], [], []]
        pred_list = [[], [], [], []]
        total_loss = 0
        loss_list = [0] * 4

        self.model.eval()
        for batch in loader:
            x_dense, x_sparse, objmean, tscnt, label_0 = [b.to(self.device) for b in batch]  # , spcnt
            outputs = self.model(x_dense, x_sparse)

            for i, (y, output) in enumerate(zip([objmean, tscnt, label_0], outputs)):  # , spcnt
                true_list[i].append(y.cpu().detach().numpy())
                pred_list[i].append(output.cpu().detach().numpy())

                if i < 3:
                    batch_loss = self.criterion_reg(output, y).cpu().detach().item() * y.shape[0]
                else:
                    batch_loss = self.criterion_clf(output, y).cpu().detach().item() * y.shape[0]

                total_loss += batch_loss
                loss_list[i] += batch_loss
        objmean_scaler = MinMaxScaler((0, 1))
        true_list[0] = objmean_scaler.inverse_tinverse_transformransform(np.concatenate(true_list[0], axis=0).reshape(-1, 1))
        pred_list[0] = objmean_scaler.inverse_tinverse_transformransform(np.concatenate(pred_list[0], axis=0).reshape(-1, 1))
        true_list[0] = np.expm1(true_list[0].flatten())
        pred_list[0] = np.expm1(pred_list[0].flatten())

        for i in range(1, 3):
            true_list[i] = np.expm1(np.concatenate(true_list[i], axis=0))
            pred_list[i] = np.expm1(np.concatenate(pred_list[i], axis=0))

        true_list[3] = np.concatenate(true_list[3], axis=0).reshape(-1, 1)
        pred_list[3] = np.argmax(np.concatenate(pred_list[3], axis=0), axis=1).reshape(-1, 1)

        result = {
            'total_loss': total_loss / len(loader.dataset),
            'objmean': mean_squared_error(true_list[0], pred_list[0], squared=False),
            'objmean_loss': loss_list[0] / len(loader.dataset),
            'tscnt': mean_squared_error(true_list[1], pred_list[1], squared=False),
            'tscnt_loss': loss_list[1] / len(loader.dataset),
            # 'spcnt': mean_squared_error(true_list[2], pred_list[2], squared=False),
            # 'spcnt_loss': loss_list[2]/len(loader.dataset),
            'label_0': accuracy_score(true_list[3], pred_list[3]),
            'label_0_loss': loss_list[3] / len(loader.dataset)
        }

        return result, true_list, pred_list
