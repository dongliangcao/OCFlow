import pytorch_lightning as pl
from models.model import FlowStageModel, InpaintingStageModel
from models.lightning_datamodule import DatasetModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--network_type', type=str, help='Type of network: [flow, inpainting]', default='flow')
    parser.add_argument('--model', type=str, help='Type of model', default='simple')
    parser.add_argument('--dataset_name', type=str, help='Name of dataset', default = 'MpiSintelClean')
    parser.add_argument('--root', type=str, help='Data root')
    
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    parser.add_argument('--epochs', type=int, help='Epochs to train', default=100)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=1e-3) 
    parser.add_argument('--overfit_batches', type=int, help='Mode of training', default =0)
    parser.add_argument('--find_best_lr', action = 'store_true', help='Use Trainer to find the best learning')


    args = parser.parse_args()
    
    hparams = dict(network_type = args.network_type, model=args.model, epochs = args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)

    network_type = args.network_type
    assert network_type in ['flow', 'inpainting'], 'Unknown network type'
    if network_type == 'flow':
        assert hparams['model'] in ['simple', 'flownets', 'flownetc', 'pwc', 'flownet', 'eflownet', 'eflownet2']
        hparams['smoothness_weight'] = 0.05
        hparams['second_order_weight'] = 0.0
        hparams['with_occ'] = False
        model = FlowStageModel(hparams=hparams)
    else:
        hparams['second_order_weight'] = 0.0
        model = InpaintingStageModel(hparams=hparams)
    max_epochs = args.epochs
    #specify data module
    dataset_name = args.dataset_name
    image_size = (64,128)
    assert dataset_name in ['ImgFlowOcc', 'MpiSintelClean', 'MpiSintelFinal', 'MpiSintelCleanOcc', 'MpiSintelFinalOcc', 'MpiSintelCleanFlowOcc', 'MpiSintelFinalFlowOcc', 'MpiSintelCleanInpainting', 'MpiSintelFinalInpainting']
    data_module = DatasetModule(root=args.root,image_size= image_size, batch_size=args.batch_size, dataset_name=dataset_name)
    data_module.prepare_data()
    data_module.setup()

    early_stop_callback = EarlyStopping(monitor='val_loss',
    min_delta=0.00,
    patience=70,
    verbose=False,
    mode='min')
    tb_logger = pl_loggers.TensorBoardLogger('tensorboard_logs/')
    #specify Trainer and start training
    if not args.find_best_lr: 
        trainer = pl.Trainer(max_epochs=max_epochs, gpus=1, overfit_batches=args.overfit_batches, logger = tb_logger)
        trainer.fit(model, datamodule = data_module)
    else: 
        trainer = pl.Trainer(gpus =1, max_epochs = max_epochs, logger = tb_logger)
        lr_finder = trainer.tuner.lr_find(model, datamodule = data_module, early_stop_threshold=None, num_training=100)
        suggested_lr = lr_finder.suggestion()
        print(suggested_lr)
        model.hparams['lr'] = suggested_lr
        trainer.fit(model, datamodule = data_module)