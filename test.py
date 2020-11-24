import pytorch_lightning as pl
from models.inpainting_model import InpaintingModel
from models.flow_model import FlowModel
from models.occlusion_model import OcclusionModel
from models.flow_occ_model import FlowOccModel
from models.networks.lightning_datamodule import DatasetModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse
# flow occ model should be ['simple', 'flowoccnets', 'flowoccnetc', 'pwoc', 'flowoccnet']
# optical flow model should be ['simple', 'flownets', 'flownetc', 'pwc', 'flownet']
# occlusion model should be ['simple', 'occnets', 'occnetc']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--network_type', type=str, help='Type of network: [flow, occ, flow-occ, inpainting]', default='flow')
    parser.add_argument('--model', type=str, help='Type of model', default='simple')
    parser.add_argument('--epochs', type=int, help='Epochs to train', default=100)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=1e-3)
    parser.add_argument('--dataset_name', type=str, help='Name of dataset', default = 'MpiSintelClean')
    parser.add_argument('--root', type=str, help='Data root')
    parser.add_argument('--overfit_batches', type=int, help='Mode of training', default =0.0)
    parser.add_argument('--find_best_lr', action = 'store_true', help='Use Trainer to find the best learning')
                        
    args = parser.parse_args()
    
    hparams = dict(network_type = args.network_type, model=args.model, epochs = args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
    
    network_type = args.network_type
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
    
    max_epochs = args.epochs
    #specify data module
    data_module = DatasetModule(root = args.root, image_size= (32,32), batch_size=args.batch_size, dataset_name = args.dataset_name)
    data_module.prepare_data()
    data_module.setup()
    #specify early stopping
    early_stop_callback = EarlyStopping(monitor='train_loss',
    min_delta=0.00,
    patience=50,
    verbose=False,
    mode='min')
    
    #specify Trainer and start training
    if not args.find_best_lr: 
        trainer = pl.Trainer(max_epochs=max_epochs, gpus=1, callbacks=[early_stop_callback], overfit_batches=args.overfit_batches)
        trainer.fit(model, datamodule = data_module)
    else: 
        trainer = pl.Trainer(gpus =1, max_epochs = max_epochs)
        lr_finder = trainer.tuner.lr_find(model, datamodule = data_module, early_stop_threshold=None)
        suggested_lr = lr_finder.suggestion()
        print(suggested_lr)