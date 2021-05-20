#!/usr/bin/env python
# coding: utf-8
from module import MultiTaskModel

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
# from MultiTasksLightningModel import MultiTasksLightningModel
import dataset_builder 

if __name__ == "__main__":
    batch_size = 64
    USE_CHID = True 
    train_loader = DataLoader(dataset=dataset_builder.train_dataset.run()[0], shuffle=True, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(dataset=dataset_builder.test_dataset.run()[0], shuffle=False, batch_size=batch_size, num_workers=4)

    print('DataLoader Built', train_loader)

    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = MultiTaskModel(dataset_builder.dense_dims.run()[0], dataset_builder.sparse_dims.run()[0], hidden_dims=64, out_dims=[1, 1, 1], n_layers=2, use_chid=USE_CHID, cell='GRU', bi=False, dropout=0.1)

    print('Model Built')
    # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    # trainer = pl.Trainer(auto_lr_find=True)
    # trainer = Trainer(model, optimizer, device)
    logger = TensorBoardLogger('/home/ai/work/logs/tensorboard', 'run_pytorch_lightening')
    # model.inputs
    single_batch = next(iter(test_loader))
    logger.experiment.add_graph(model, [single_batch[0], single_batch[1]])
    trainer = pl.Trainer(logger=logger)
    lr_finder = trainer.tuner.lr_find(model, train_dataloader=train_loader)
    # print(lr_finder.results)
    new_lr = lr_finder.suggestion()
    print('suggested lr:', new_lr)
    model.hparams.lr = new_lr

    print('Trainer Built')

    # t0 = time()
    # history = trainer.fit(train_loader, test_loader, epoch=1, early_stop=20)
    # t1 = time()
    trainer.fit(model, train_loader, test_loader)

    # print('cost: {:.2f}'.format(t1 - t0))

# processed_x_data.run()
'''
print('Apply Split and Dense Feature Transformation again on Test Data')
x_test_dense = x_test[:, -w_size:, len(category_cols):].astype(np.float64)
x_test_sparse = x_test[:, -w_size:, sparse_index].astype(np.int64)

x_test_dense = np.log1p(x_test_dense - x_test_dense.min(axis=0))
x_test_dense = x_scaler.transform(x_test_dense.reshape(-1, x_test_dense.shape[-1])).reshape(x_test_dense.shape)
'''

# print(x_train_dense.shape, x_train_sparse.shape)
# print(train_objmean.shape, train_tscnt.shape, train_label_0.shape)  # train_spcnt.shape,


# print(x_test_dense.shape, x_test_sparse.shape)
# print(test_objmean.shape, test_tscnt.shape, test_label_0.shape)  # test_spcnt.shape
