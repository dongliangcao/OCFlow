import pytorch_lightning as pl
from models.inpainting_model import InpaintingModel
from models.flow_model import FlowModel
from models.occlusion_model import OcclusionModel
from models.flow_occ_model import FlowOccModel
from models.lightning_datamodule import DatasetModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import yaml
import torch
# flow occ model should be ['simple', 'flowoccnets', 'flowoccnetc', 'pwoc', 'flowoccnet']
# optical flow model should be ['simple', 'flownets', 'flownetc', 'pwc', 'flownet']
# occlusion model should be ['simple', 'occnets', 'occnetc']

if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('--network_type', type=str, help='Type of network: [flow, occ, flow-occ, inpainting]', default='flow')
#     parser.add_argument('--model', type=str, help='Type of model', default='simple')
#     parser.add_argument('--epochs', type=int, help='Epochs to train', default=100)
#     parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
#     parser.add_argument('--learning_rate', type=float, help='Learning rate', default=1e-3) 
#     parser.add_argument('--dataset_name', type=str, help='Name of dataset', default = 'MpiSintelClean')
#     parser.add_argument('--root', type=str, help='Data root')
#     parser.add_argument('--overfit_batches', type=int, help='Mode of training', default =0)
#     parser.add_argument('--find_best_lr', action = 'store_true', help='Use Trainer to find the best learning')
                        
#     args = parser.parse_args()
    
    file_name = r'config\supervised_config.yml'
    with open(file_name) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    
    hparams = dict(network_type=args['network_type'], model=args['model'], epochs = args['epochs'], batch_size=args['batch_size'], learning_rate=args['learning_rate'], displacement=args['displacement'])
    
    network_type = args['network_type']
    assert network_type in ['flow', 'occ', 'flow-occ', 'inpainting'], 'Unknown network type'
    if network_type == 'flow':
        assert hparams['model'] in ['simple', 'flownets', 'flownetc', 'pwc', 'flownet', 'eflownet', 'eflownet2']
        model = FlowModel(hparams=hparams)
    elif network_type == 'occ':
        assert hparams['model'] in ['simple', 'occnets', 'occnetc']
        model = OcclusionModel(hparams=hparams)
    elif network_type == 'flow-occ':
        assert hparams['model'] in ['simple', 'flowoccnets', 'flowoccnetc', 'pwoc', 'flowoccnet']
        model = FlowOccModel(hparams=hparams)
    else:
        model = InpaintingModel(hparams=hparams)
    
    max_epochs = args['epochs']
    #specify data module
    dataset_name = args['dataset_name']
    image_size = args['image_size']
    assert dataset_name in ['ImgFlowOcc', 'MpiSintelClean', 'MpiSintelFinal', 'MpiSintelCleanOcc', 'MpiSintelFinalOcc', 'MpiSintelCleanFlowOcc', 'MpiSintelFinalFlowOcc']
    data_module = DatasetModule(root=args['root'],image_size=image_size, batch_size=args['batch_size'], dataset_name=dataset_name, overfit = args['overfit'])
    data_module.prepare_data()
    data_module.setup()
    #specify early stopping callbacks
    early_stop_callback = EarlyStopping(monitor='monitored_loss',
    min_delta=0.00,
    patience=60,
    verbose=False,
    mode='min')
    #specify checkpoint callbacks
    checkpoint_callback = ModelCheckpoint(
    monitor='monitored_loss',
    save_top_k=1,
    mode='min',verbose = True)
    #specify logger
    tb_logger = pl_loggers.TensorBoardLogger('tensorboard_logs/')
    #specify Trainer and start training
    if not args['find_best_lr']: 
        trainer = pl.Trainer(max_epochs=max_epochs, gpus=1, logger=tb_logger, callbacks=[checkpoint_callback])
        trainer.fit(model, datamodule=data_module)
    else: 
        trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, logger=tb_logger, callbacks=[checkpoint_callback])
        lr_finder = trainer.tuner.lr_find(model, datamodule=data_module, early_stop_threshold=None, num_training=100)
        suggested_lr = lr_finder.suggestion()
        print(suggested_lr)
        model.lr = suggested_lr
        trainer.fit(model, datamodule=data_module)
    print(trainer.logged_metrics)