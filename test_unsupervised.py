import pytorch_lightning as pl
from models.model import FlowStageModel, InpaintingStageModel, TwoStageModel, TwoStageModelGC, InpaintingGConvModel
from models.lightning_datamodule import DatasetModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import yaml
import torch
import time

if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     #parameters for all models
#     parser.add_argument('--network_type', type=str, help='Type of network: [flow, inpainting, twostage]', default='flow')
#     parser.add_argument('--model', type=str, help='Type of model', default='simple')
#     parser.add_argument('--dataset_name', type=str, help='Name of dataset', default = 'MpiSintelClean')
#     parser.add_argument('--root', type=str, help='Data root')
    
#     parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
#     parser.add_argument('--epochs', type=int, help='Epochs to train', default=100)
#     parser.add_argument('--learning_rate', type=float, help='Learning rate', default=1e-3) 
#     parser.add_argument('--overfit_batches', type=int, help='Mode of training', default =0)
#     parser.add_argument('--find_best_lr', action='store_true', help='Use Trainer to find the best learning')
#     parser.add_argument('--log_every_n_steps', type=int, help='Log every n steps', default = 20)
#     parser.add_argument('--image_size', type=int, nargs=2, help='Image size', default=[64, 128])
#     #parameters for each model
#     parser.add_argument('--smoothness_weight', type=float, help='Weight for smoothness loss', default=0.05)
#     parser.add_argument('--second_order_weight', type=float, help='Weight for gradient of image', default=0.0)
#     parser.add_argument('--reconst_weight', type=float, help='Weight for reconstruction loss', default=2.0)
#     parser.add_argument('--with_occ', action='store_true', help='Training flownet with ground truth occlusion mask or not')
#     parser.add_argument('--flow_root', type=str, help='Path to flow model checkpoint', default=None)
#     parser.add_argument('--supervised_flow', action='store_true', help='Used trained supervised model in TwoStageModel instead of unsupervised one')
#     parser.add_argument('--inpainting_root', type=str, help='Path to inpainting model checkpoint', default=None)
#     parser.add_argument('--org', action='store_true', help='Use the original version of the gated convolution')


#     args = parser.parse_args()
    pl.seed_everything(42)
    file_name = r'config/unsupervised_config.yml'
    with open(file_name) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    time_stamp = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    result_dir = '{}/{}'.format(args['result_dir'], time_stamp)
    print('result directory is {}'.format(result_dir))
    hparams = dict(network_type = args['network_type'], model=args['model'], epochs = args['epochs'], batch_size=args['batch_size'], learning_rate=args['learning_rate'], log_every_n_steps = args['log_every_n_steps'], img_size=args['image_size'], org=args['org'])

    network_type = args['network_type']
    automatic_optimization = True
    image_size = args['image_size']
    assert network_type in ['flow', 'inpainting', 'twostage'], 'Unknown network type'
    if network_type == 'flow':
        assert hparams['model'] in ['simple', 'flownets', 'flownetc', 'pwc', 'flownet', 'eflownet', 'eflownet2']
        hparams['photo_weight'] = args['photo_weight']
        hparams['smooth1_weight'] = args['smooth1_weight']
        hparams['smooth2_weight'] = args['smooth2_weight']
        hparams['with_occ'] = args['with_occ']
        hparams['occ_aware'] = args['occ_aware']
        hparams['displacement'] = args['displacement']
        model = FlowStageModel(hparams=hparams)
    elif network_type == 'inpainting':
        assert hparams['model'] in ['simple', 'gated']
        if hparams['model'] == 'simple': 
            hparams['second_order_weight'] = args['second_order_weight']
            hparams['n_display_images'] = args['n_display_images']
            hparams['result_dir'] = result_dir
            hparams['log_image_every_epoch'] = args['log_image_every_epoch']
            model = InpaintingStageModel(hparams=hparams)
        else: 
            hparams['n_display_images'] = args['n_display_images']
            hparams['result_dir'] = result_dir
            hparams['log_image_every_epoch'] = args['log_image_every_epoch']
            model = InpaintingGConvModel(hparams=hparams)
            automatic_optimization = False
    else: 
        assert hparams['model'] in ['with_gt_flow', 'no_gt_flow']
        hparams['reconst_weight'] = args['reconst_weight']
        hparams['inpainting_root'] = args['inpainting_root']
        if hparams['model'] == 'no_gt_flow': 
            hparams['smoothness_weight'] = args['smoothness_weight']
            hparams['flow_root'] = args['flow_root']
            if args['supervised_flow']: 
                hparams['supervised_flow'] = True
            else: 
                hparams['supervised_flow'] = False
            model = TwoStageModel(hparams=hparams)
        else: 
            hparams['inpainting_stage'] = args['inpainting_stage']
            assert hparams['inpainting_stage'] in ['gated', 'gated_org', 'simple']
            model = TwoStageModelGC(hparams=hparams)

    max_epochs = args['epochs']
    #specify data module
    dataset_name = args['dataset_name']
    
    assert dataset_name in ['ImgFlowOcc', 'MpiSintelClean', 'MpiSintelFinal', 'MpiSintelCleanOcc', 'MpiSintelFinalOcc', 'MpiSintelCleanFlowOcc', 'MpiSintelFinalFlowOcc', 'MpiSintelCleanInpainting', 'MpiSintelFinalInpainting', 'FlyingChairsInpainting', 'FlyingChairs2', 'FlyingChairs']
    data_module = DatasetModule(root=args['root'],image_size=image_size, batch_size=args['batch_size'], dataset_name=dataset_name, static_occ=args['static_occ'], overfit=args['overfit'], occlusion_ratio=args['occlusion_ratio'])
    data_module.prepare_data()
    data_module.setup()
    #specify early stopping callback
    early_stop_callback = EarlyStopping(monitor='monitored_loss',
    min_delta=0.00,
    patience=70,
    verbose=False,
    mode='min')
    #specify model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
    monitor='monitored_loss',
    save_top_k=1,
    mode='min',)
    #specify logger
    tb_logger = pl_loggers.TensorBoardLogger('tensorboard_logs/')
    #specify Trainer and start training
    if not args['find_best_lr']: 
        #trainer = pl.Trainer(max_epochs=max_epochs, gpus=-1, accelerator='ddp', logger=tb_logger, callbacks=[checkpoint_callback], automatic_optimization=automatic_optimization)
        trainer = pl.Trainer(max_epochs=max_epochs, gpus=1, logger=tb_logger, callbacks=[checkpoint_callback], automatic_optimization=automatic_optimization)
        trainer.fit(model, datamodule=data_module)
    else: 
        trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, logger=tb_logger, callbacks=[checkpoint_callback], automatic_optimization= automatic_optimization)
        #trainer = pl.Trainer(max_epochs=max_epochs, gpus=-1, accelerator='ddp', logger=tb_logger, callbacks=[checkpoint_callback], automatic_optimization=automatic_optimization)
        lr_finder = trainer.tuner.lr_find(model, datamodule=data_module, early_stop_threshold=None, num_training=100)
        suggested_lr = lr_finder.suggestion()
        print(suggested_lr)
        model.lr = suggested_lr
        trainer.fit(model, datamodule=data_module)
