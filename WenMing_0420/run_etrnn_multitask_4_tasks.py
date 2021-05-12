#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from time import time
# from tqdm.notebook import tqdm
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


# In[ ]:


import torch
from torch import nn
from torch.autograd import Variable

class ET_Rnn(torch.nn.Module):
    def __init__(self, dense_dims, sparse_dims, hidden_dims, n_layers=1, use_chid=True, cell='GRU', bi=False, dropout=0, device='cpu'):
        super(ET_Rnn, self).__init__()
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.cell = cell
        self.use_chid = use_chid
        self.device = device
        self.bi = bi

        self.embedding_list = nn.ModuleList([nn.Embedding(fd, ed, padding_idx=0) for fd, ed in sparse_dims])

        if use_chid:
            rnn_in_dim = dense_dims + sum([ed for fd, ed in sparse_dims[1:]])
            self.out_dim = hidden_dims*(bi+1) + sparse_dims[0][1] # chid embed dim
            self.user_layer = nn.Linear(sparse_dims[0][1], sparse_dims[0][1])

        else:
            rnn_in_dim = dense_dims + sum([ed for fd, ed in sparse_dims[:]])
            self.out_dim = hidden_dims*(bi+1)

        if self.cell == 'LSTM':
            self.rnn = nn.LSTM(rnn_in_dim, hidden_dims, n_layers, batch_first=True, bidirectional=bi, dropout=dropout)
        elif self.cell == 'GRU':
            self.rnn = nn.GRU(rnn_in_dim, hidden_dims, n_layers, batch_first=True, bidirectional=bi, dropout=dropout)

        self.init_embedding()

    def init_embedding(self):
        for embed in self.embedding_list:
            embed.reset_parameters()

    def init_hidden(self, x):
        if self.cell == 'LSTM':
            hidden = Variable(torch.zeros(self.n_layers*(self.bi+1), x.size(0), self.hidden_dims)).to(self.device)
            context = Variable(torch.zeros(self.n_layers*(self.bi+1), x.size(0), self.hidden_dims)).to(self.device)
            ret = (hidden, context)
        elif self.cell == 'GRU':
            hidden = Variable(torch.zeros(self.n_layers*(self.bi+1), x.size(0), self.hidden_dims)).to(self.device)
            ret = hidden

        return ret

    def forward(self, x_dense, x_sparse):
        if self.use_chid:
            x = torch.cat([x_dense]+[embed(x_sparse[:, :, i+1]) for i, embed in enumerate(self.embedding_list[1:])], dim=-1)
        else:
            x = torch.cat([x_dense]+[embed(x_sparse[:, :, i]) for i, embed in enumerate(self.embedding_list[:])], dim=-1)

        self.hidden = self.init_hidden(x)
        logits, self.hidden = self.rnn(x, self.hidden)

        if self.use_chid:
            user_embed = self.user_layer(self.embedding_list[0](x_sparse[:,0,0]))
            last_logits = torch.cat([logits[:, -1], user_embed], dim=-1)
        else:
            last_logits = logits[:, -1]

        return last_logits


# In[ ]:


class MLP(nn.Module):
    def __init__(self, input_dims, hidden_dims=[1], out_dim=1):
        super(MLP, self).__init__()
        hidden_dims = [input_dims] + hidden_dims

        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(idim, odim),
                nn.ReLU()
            ) for idim, odim in zip(hidden_dims[:-1], hidden_dims[1:])
        ])

        self.out_layer = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x):
        out = self.layers(x)
        out = self.out_layer(out)

        return out


# In[ ]:


class MultiTaskModel(nn.Module):
    def __init__(self, dense_dims, sparse_dims, hidden_dims, out_dims=[1], n_layers=1, use_chid=True, cell='GRU', bi=False, dropout=0, device='cpu'):
        super(MultiTaskModel, self).__init__()
        self.rnn = ET_Rnn(dense_dims, sparse_dims, hidden_dims, n_layers=n_layers, use_chid=use_chid,
                          cell=cell, bi=bi, dropout=dropout, device=device)

        self.mlps = nn.ModuleList([MLP(self.rnn.out_dim, hidden_dims=[self.rnn.out_dim//2], out_dim=od) for od in out_dims])

    def forward(self, x_dense, x_sparse):
        logits = self.rnn(x_dense, x_sparse)
        outs = [mlp(logits)for mlp in self.mlps]

        return outs


# In[ ]:


class Trainer:
    def __init__(self, model, optimizer, device='cpu'):
        self.model = model.to(device)
        self.criterion_reg = nn.MSELoss()
        self.criterion_clf = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.device = device

    def fit(self, train_loader, test_loader=None, epoch=1, early_stop=-1, scaler=None):
        history = {
            'train': [],
            'test': []
        }

        best_eval = 9e9
        early_cnt = 0
        best_model_params = copy.deepcopy(self.model.state_dict())

        for ep in tqdm(range(epoch)):
            #print('Epoch:{}'.format(ep+1))

            self.model.train()
            for batch in train_loader:
                self.optimizer.zero_grad()

                x_dense, x_sparse, objmean, tscnt, label_0 = [b.to(self.device) for b in batch] # , spcnt

                outputs = self.model(x_dense, x_sparse)

                loss = self.criterion_reg(outputs[0], objmean)
                if ep >= 20:
                    loss += self.criterion_reg(outputs[1], tscnt)
                # if ep >= 30:
                #    loss += self.criterion_reg(outputs[2], spcnt)
                if ep >= 40:
                    loss += self.criterion_clf(outputs[3], label_0)

                loss.backward()
                self.optimizer.step()

            train_result, _, _ = self.evaluate(train_loader)
            history['train'].append(train_result)
            #print('\ttrain\t'+' '.join(['{}:{:.2f}'.format(k, v) for k, v in train_result.items()]))

            if test_loader:
                test_result, _, _ = self.evaluate(test_loader)
                history['test'].append(test_result)
                if ep%5 == 0 or ep == epoch-1:
                    print('Epoch:{}'.format(ep+1))
                    print('\ttest\t'+' '.join(['{}:{:.3f}'.format(k, v) for k, v in test_result.items()]))

                if test_result['total_loss'] < best_eval:
                    early_cnt = 0
                    best_eval = test_result['total_loss']
                    best_model_params = copy.deepcopy(self.model.state_dict())
                    #print('\tbetter!')

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
        loss_list = [0]*4

        self.model.eval()
        for batch in loader:
            x_dense, x_sparse, objmean, tscnt, label_0 = [b.to(self.device) for b in batch] # , spcnt
            outputs = self.model(x_dense, x_sparse)

            for i, (y, output) in enumerate(zip([objmean, tscnt, label_0], outputs)): # , spcnt
                true_list[i].append(y.cpu().detach().numpy())
                pred_list[i].append(output.cpu().detach().numpy())

                if i < 3:
                    batch_loss = self.criterion_reg(output, y).cpu().detach().item() * y.shape[0]
                else:
                    batch_loss = self.criterion_clf(output, y).cpu().detach().item() * y.shape[0]

                total_loss += batch_loss
                loss_list[i] += batch_loss

        true_list[0] = objmean_scaler.inverse_transform(np.concatenate(true_list[0], axis=0).reshape(-1, 1))
        pred_list[0] = objmean_scaler.inverse_transform(np.concatenate(pred_list[0], axis=0).reshape(-1, 1))
        true_list[0] = np.expm1(true_list[0].flatten())
        pred_list[0] = np.expm1(pred_list[0].flatten())

        for i in range(1, 3):
            true_list[i] = np.expm1(np.concatenate(true_list[i], axis=0))
            pred_list[i] = np.expm1(np.concatenate(pred_list[i], axis=0))

        true_list[3] = np.concatenate(true_list[3], axis=0).reshape(-1, 1)
        pred_list[3] = np.argmax(np.concatenate(pred_list[3], axis=0), axis=1).reshape(-1, 1)

        result = {
            'total_loss': total_loss/len(loader.dataset),
            'objmean': mean_squared_error(true_list[0], pred_list[0], squared=False),
            'objmean_loss': loss_list[0]/len(loader.dataset),
            'tscnt': mean_squared_error(true_list[1], pred_list[1], squared=False),
            'tscnt_loss': loss_list[1]/len(loader.dataset),
            # 'spcnt': mean_squared_error(true_list[2], pred_list[2], squared=False),
            # 'spcnt_loss': loss_list[2]/len(loader.dataset),
            'label_0': accuracy_score(true_list[3], pred_list[3]),
            'label_0_loss': loss_list[3]/len(loader.dataset)
        }

        return result, true_list, pred_list

print('Model Built')

# In[ ]:


sample_path = './data/sample_50k'

x_train = np.load(os.path.join(sample_path, 'RNN', 'x_train.npy'), allow_pickle=True)
x_test = np.load(os.path.join(sample_path, 'RNN', 'x_test.npy'), allow_pickle=True)

print('x dataset loaded')
#f_train = np.load(os.path.join(sample_path, 'RNN', 'f_train.npy'), allow_pickle=True)
#f_test = np.load(os.path.join(sample_path, 'RNN', 'f_test.npy'), allow_pickle=True)

Y_train = np.load(os.path.join(sample_path, 'RNN', 'y_train.npy'), allow_pickle=True)
Y_test = np.load(os.path.join(sample_path, 'RNN', 'y_test.npy'), allow_pickle=True)

print('y dataset loaded')

chid_mapper = np.load(os.path.join(sample_path, 'sample_50k_chid_idx_map.npy'), allow_pickle=True).item()
feat_mapper = np.load(os.path.join(sample_path, 'RNN', 'feature_map.npy'), allow_pickle=True).item()
cust_feature_map = np.load(os.path.join(sample_path, 'RNN', 'cust_feature_map.npy'), allow_pickle=True).item()

columns = np.load(os.path.join(sample_path, 'RNN', 'columns.npy'), allow_pickle=True).item()

print('data info loaded')

print(x_train.shape, x_test.shape, Y_train.shape, Y_test.shape, len(chid_mapper))
print([(k, len(v)) for k, v in feat_mapper.items()], [(k, len(v)) for k, v in cust_feature_map.items()])



# In[ ]:


category_cols = columns['x_columns'][:-4]
numeric_cols = columns['x_columns'][-4:]

print(columns['x_columns'])
print(category_cols, numeric_cols)


# In[ ]:


print(columns['y_columns'])


# In[ ]:


# regession
index = columns['y_columns'].index('objam_sum')
train_objsum = Y_train[:, [index]].astype(np.float64)
test_objsum = Y_test[:, [index]].astype(np.float64)

index = columns['y_columns'].index('objam_mean')
train_objmean = Y_train[:, [index]].astype(np.float64)
test_objmean = Y_test[:, [index]].astype(np.float64)

index = columns['y_columns'].index('trans_count')
train_tscnt = Y_train[:, [index]].astype(np.float64)
test_tscnt = Y_test[:, [index]].astype(np.float64)

# index = columns['y_columns'].index('shop_count')
# train_spcnt = Y_train[:, [index]].astype(np.float64)
# test_spcnt = Y_test[:, [index]].astype(np.float64)

print('Data Transformed: regression')
# In[ ]:


# log transform
train_objsum = np.log1p(train_objsum)
test_objsum = np.log1p(test_objsum)

train_objmean = np.log1p(train_objmean)
test_objmean = np.log1p(test_objmean)

objmean_scaler = MinMaxScaler((0, 1))
train_objmean = objmean_scaler.fit_transform(train_objmean)
test_objmean = objmean_scaler.transform(test_objmean)

train_tscnt = np.log1p(train_tscnt)
test_tscnt = np.log1p(test_tscnt)

# train_spcnt = np.log1p(train_spcnt)
# test_spcnt = np.log1p(test_spcnt)

print('Data Transformed: log')
# In[ ]:


#classfication
bounds = [0]
lable_trans = np.vectorize(lambda x: sum([x > bound for bound in bounds]))

train_label_0 = lable_trans(train_objsum)
test_label_0 = lable_trans(test_objsum)

print('Data Transformed: classification')
# In[ ]:


train_label_num = [np.sum(train_label_0 == label) for label in sorted(np.unique(train_label_0))]
test_label_num = [np.sum(test_label_0 == label) for label in sorted(np.unique(train_label_0))]

'''with sns.axes_style("darkgrid"):
    plt.figure(figsize=(4, 4))
    sns.barplot(x=[0, 1], y=train_label_num)
    plt.xticks(range(len(train_label_num)))
    plt.show()

with sns.axes_style("darkgrid"):
    plt.figure(figsize=(4, 4))
    sns.barplot(x=[0, 1], y=train_label_num)
    plt.xticks(range(len(test_label_num)))
    plt.show()'''


# In[ ]:j


sparse_feat = category_cols[:5]#+['stonc_tag', 'stonc_6_label']
dense_feat = numeric_cols

keys = list(feat_mapper.keys())
for key in keys:
    if key not in sparse_feat:
        del feat_mapper[key]

print(sparse_feat, [(k, len(v)) for k, v in feat_mapper.items()])


# In[ ]:


USE_CHID = True
idx_start = 1-int(USE_CHID)
sparse_index = [category_cols.index(feat) for feat in sparse_feat][idx_start:]

chid_embed_dim = 64
feat_embed_dim = 16

dense_dims = len(dense_feat) # number of dense feature
feat_dims = np.array([len(chid_mapper)] + [len(v) for v in feat_mapper.values()])+1 # 0 is padding index, so add 1 dims
embed_dims = [chid_embed_dim]+[feat_embed_dim]*len(feat_mapper) # dims of chid and other sparse feature

sparse_dims = [(fd, ed) for fd, ed in zip(feat_dims[idx_start:], embed_dims[idx_start:])]

dense_dims, sparse_dims, sparse_index


# In[ ]:


# x_data
w_size = x_train.shape[1]

x_scaler = MinMaxScaler(feature_range=(0, 1))

x_train_dense = x_train[:, -w_size:, len(category_cols):].astype(np.float64) # split dense feature
x_train_sparse = x_train[:, -w_size:, sparse_index].astype(np.int64) # split sparse feature

x_train_dense = np.log1p(x_train_dense - x_train_dense.min(axis=0))
x_train_dense = x_scaler.fit_transform(x_train_dense.reshape(-1, x_train_dense.shape[-1])).reshape(x_train_dense.shape)

x_test_dense = x_test[:, -w_size:, len(category_cols):].astype(np.float64)
x_test_sparse = x_test[:, -w_size:, sparse_index].astype(np.int64)

x_test_dense = np.log1p(x_test_dense - x_test_dense.min(axis=0))
x_test_dense = x_scaler.transform(x_test_dense.reshape(-1, x_test_dense.shape[-1])).reshape(x_test_dense.shape)


# In[ ]:


print(x_train_dense.shape, x_train_sparse.shape)
print(train_objsum.shape, train_objmean.shape, train_tscnt.shape, train_label_0.shape) # train_spcnt.shape,


# In[ ]:


print(x_test_dense.shape, x_test_sparse.shape)
print(test_objsum.shape, test_objmean.shape, test_tscnt.shape, test_label_0.shape) # test_spcnt.shape

print('Data Transformed')
# In[ ]:


batch_size = 8192

train_dataset = TensorDataset(torch.FloatTensor(x_train_dense), torch.LongTensor(x_train_sparse),
                              torch.FloatTensor(train_objmean), torch.FloatTensor(train_tscnt),
                              torch.LongTensor(train_label_0.flatten())) # torch.FloatTensor(train_spcnt),
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=8)

test_dataset = TensorDataset(torch.FloatTensor(x_test_dense), torch.LongTensor(x_test_sparse),
                              torch.FloatTensor(test_objmean), torch.FloatTensor(test_tscnt),
                              torch.LongTensor(test_label_0.flatten())) # torch.FloatTensor(test_spcnt),
test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, num_workers=8)


# In[ ]:


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
out_dims = [1, 1, 1] # , 2

model = MultiTaskModel(dense_dims, sparse_dims, hidden_dims=64, out_dims=out_dims, n_layers=2,
                       use_chid=USE_CHID, cell='GRU', bi=False, dropout=0.1, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

print(device)
model


# In[ ]:


t0 = time()
trainer = Trainer(model, optimizer, device)
history = trainer.fit(train_loader, test_loader, epoch=100, early_stop=20)
t1 = time()

print('cost: {:.2f}'.format(t1 - t0))


# In[ ]:


df_history = pd.DataFrame(history['test'])

'''with sns.axes_style("darkgrid"):
    show_cols = ['tscnt_loss',  'label_0_loss'] # 'spcnt_loss',

    plt.figure(figsize=(8, 4))
    sns.lineplot(data=pd.DataFrame(df_history[show_cols]))
    plt.show()

    plt.figure(figsize=(8, 4))
    show_cols = ['objmean_loss']
    sns.lineplot(data=pd.DataFrame(df_history[show_cols]))
    plt.show()'''


# In[ ]:


'''with sns.axes_style("darkgrid"):
    plt.figure(figsize=(8, 4))
    show_cols = ['tscnt']
    sns.lineplot(data=pd.DataFrame(df_history[show_cols]))
    plt.show()'''


# In[ ]:

'''
with sns.axes_style("darkgrid"):
    plt.figure(figsize=(8, 4))
    show_cols = ['spcnt']
    sns.lineplot(data=pd.DataFrame(df_history[show_cols]))
    plt.show()

'''
# In[ ]:


'''with sns.axes_style("darkgrid"):
    plt.figure(figsize=(8, 4))
    show_cols = ['label_0']
    sns.lineplot(data=pd.DataFrame(df_history[show_cols]))
    plt.show()'''


# In[ ]:


train_result, train_true_list, train_pred_list = trainer.evaluate(train_loader)
test_result, test_true_list, test_pred_list = trainer.evaluate(test_loader)

print('train\t'+' '.join(['{}:{:.2f}'.format(k, v) for k, v in train_result.items()]))
print('test\t'+' '.join(['{}:{:.2f}'.format(k, v) for k, v in test_result.items()]))


# In[ ]:


cf_matrix = confusion_matrix(test_true_list[-1].reshape(-1, 1), test_pred_list[-1].reshape(-1, 1), normalize='true')

'''with sns.axes_style("darkgrid"):
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(cf_matrix, linewidths=.01, annot=True)
    ax.set_xlabel('Predict', fontsize=16)
    ax.set_ylabel('True', fontsize=16)
    plt.show()'''


# In[ ]:


{
    'AccuracyScore': accuracy_score(test_true_list[-1], test_pred_list[-1]),
    'RecallScore': recall_score(test_true_list[-1], test_pred_list[-1], average='macro'),
    'PrecisionScore': precision_score(test_true_list[-1], test_pred_list[-1], average='macro'),
    'F1Score': f1_score(test_true_list[-1], test_pred_list[-1], average='macro'),
}


# In[ ]:


torch.save({
    'dense_dims': dense_dims,
    'sparse_dims': sparse_dims,
    'hidden_dims': 64,
    'n_layers': 2,
    'use_chid': True,
    'cell': 'GRU',
    'bi': False,
    'dropout': 0.1,
    'model_state_dict': model.rnn.cpu().state_dict()
}, './models/rnn.pt')


# In[ ]:


checkpoint = torch.load('./models/rnn.pt')
checkpoint

