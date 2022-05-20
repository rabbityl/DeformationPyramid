import os, torch, argparse, shutil
torch.multiprocessing.set_sharing_strategy('file_system')
from easydict import EasyDict as edict
import yaml
from datasets.dataloader import get_dataloader, get_datasets
from lepard.pipeline import Pipeline as Lepard
from outlier_rejection.pipeline import Outlier_Rejection
from lib.utils import setup_seed
from lib.tester import get_trainer
from lib.tictok import Timers

from torch import optim



setup_seed(0)

def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])

yaml.add_constructor('!join', join)


if __name__ == '__main__':

    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help= 'Path to the config file.')
    args = parser.parse_args()
    with open(args.config,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
        with open( config['matcher_config'], 'r' ) as f_ :
            matcher_config = yaml.load(f_, Loader=yaml.Loader)
            matcher_config=edict(matcher_config)
        config['kpfcn_config'] = matcher_config['kpfcn_config']

    config['snapshot_dir'] = 'snapshot/%s/%s' % (config['folder'], config['exp_dir'])
    config['tboard_dir'] = 'snapshot/%s/%s/tensorboard' % (config['folder'], config['exp_dir'])
    config['save_dir'] = 'snapshot/%s/%s/checkpoints' % (config['folder'], config['exp_dir'])
    config = edict(config)

    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)

    if config.gpu_mode:
        config.device = torch.device("cuda:0")
    else:
        config.device = torch.device('cpu')
    
    # backup the experiment
    if config.mode == 'train':
        os.system(f'cp -r configs {config.snapshot_dir}')
        os.system(f'cp -r cpp_wrappers {config.snapshot_dir}')
        os.system(f'cp -r datasets {config.snapshot_dir}')
        os.system(f'cp -r kernels {config.snapshot_dir}')
        os.system(f'cp -r lepard {config.snapshot_dir}')
        os.system(f'cp -r lib {config.snapshot_dir}')
        os.system(f'cp -r outlier_rejection {config.snapshot_dir}')
        shutil.copy2('main.py', config.snapshot_dir)

    
    # model initialization
    config.matcher = Lepard(matcher_config) # pretrained point cloud matcher model
    config.model = Outlier_Rejection(config.model)
    # config.model = NonLocalNet( in_dim=6, num_layers=6, num_channels=128)  # Model from PointDSC Bai+ 2021
    matcher_params_cnt = sum(p.numel() for p in config.matcher.parameters())
    model_params_cnt = sum(p.numel() for p in config.model.parameters() if p.requires_grad)
    print("#param in matcher", matcher_params_cnt)
    print("#param in model", model_params_cnt)


    # create optimizer 
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    

    #create learning rate scheduler
    if  'overfit' in config.exp_dir :
        config.scheduler = optim.lr_scheduler.MultiStepLR(
            config.optimizer,
            milestones=[config.max_epoch-1], # fix lr during overfitting
            gamma=0.1,
            last_epoch=-1)

    else:
        config.scheduler = optim.lr_scheduler.ExponentialLR(
            config.optimizer,
            gamma=config.scheduler_gamma,
        )


    config.timers = Timers()

    # create dataset and dataloader
    train_set, val_set, test_set = get_datasets(config)
    config.train_loader, neighborhood_limits = get_dataloader(train_set,config, shuffle=False)
    config.val_loader, _ = get_dataloader(val_set, config, shuffle=False, neighborhood_limits=neighborhood_limits)
    config.test_loader, _ = get_dataloader(test_set, config, shuffle=False, neighborhood_limits=neighborhood_limits)
    

    trainer = get_trainer(config)
    if(config.mode=='train'):
        trainer.train()
    else:
        trainer.test()