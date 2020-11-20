import pytorch_lightning as pl
from models.inpainting_model import InpaintingModel
from models.flow_model import FlowModel
from models.occlusion_model import OcclusionModel
from models.flow_occ_model import FlowOccModel
from models.networks.lightning_datamodule import ImageFlowOccModule
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
    parser.add_argument('--root', type=str, help='Data root')
                        
    args = parser.parse_args()
    
    hparams = dict(learning_rate=args.learning_rate, batch_size=args.batch_size, image_size=(512, 1024), model=args.model)
    
    network_type = args.network_type
    assert network_type in ['flow', 'occ', 'flow-occ', 'inpainting'], 'Unknown network type'
    if network_type == 'flow':
        assert hparams['model'] in ['simple', 'flownets', 'flownetc', 'pwc', 'flownet']
        model = FlowModel(root=args.root, hparams=hparams)
    elif network_type == 'occ':
        assert hparams['model'] in ['simple', 'occnets', 'occnetc']
        model = OcclusionModel(root=args.root, hparams=hparams)
    elif network_type == 'flow-occ':
        assert hparams['model'] in ['simple', 'flowoccnets', 'flowoccnetc', 'pwoc', 'flowoccnet']
        model = FlowOccModel(root=args.root, hparams=hparams)
    else:
        model = InpaintingModel(root=args.root, hparams=hparams)
    
    max_epochs = args.epochs
    data_module = ImageFlowOccModule(root = args.root, image_size= (512,1024), batch_size=args.batch_size)
    data_module.prepare_data()
    data_module.setup()
    trainer = pl.Trainer(max_epochs=max_epochs, gpus=1, overfit_batches = 1)
    trainer.fit(model, datamodule = data_module)