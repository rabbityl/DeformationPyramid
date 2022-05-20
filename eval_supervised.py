from model.geometry import *
import os
import torch
import sys
sys.path.append("correspondence")


from tqdm import tqdm
import argparse



from model.registration import Registration
import  yaml
from easydict import EasyDict as edict
from model.loss import compute_flow_metrics

from utils.benchmark_utils import setup_seed
from utils.utils import Logger, AverageMeter
from utils.tiktok import Timers

from correspondence.landmark_estimator import Landmark_Model



def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])
yaml.add_constructor('!join', join)

setup_seed(0)



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help= 'Path to the config file.')
    parser.add_argument('--visualize', action = 'store_true', help= 'visualize the registration results')
    args = parser.parse_args()
    with open(args.config,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config['snapshot_dir'] = 'snapshot/%s/%s' % (config['folder'], config['exp_dir'])
    os.makedirs(config['snapshot_dir'], exist_ok=True)


    config = edict(config)


    # backup the experiment
    os.system(f'cp -r config {config.snapshot_dir}')
    os.system(f'cp -r data {config.snapshot_dir}')
    os.system(f'cp -r model {config.snapshot_dir}')
    os.system(f'cp -r utils {config.snapshot_dir}')


    if config.gpu_mode:
        config.device = torch.cuda.current_device()
    else:
        config.device = torch.device('cpu')


    ldmk_model =  Landmark_Model(config_file = config.ldmk_config, device=config.device)
    config['kpfcn_config'] = ldmk_model.kpfcn_config

    model = Registration(config)
    timer = Timers()

    stats_meter = None


    from correspondence.datasets._4dmatch import _4DMatch
    from correspondence.datasets.dataloader import get_dataloader


    splits = [ '4DMatch-F', '4DLoMatch-F' ]


    for split in splits:

        config.split['test'] = split


        test_set = _4DMatch(config, 'test', data_augmentation=False)
        test_loader, _ = get_dataloader(test_set, config, shuffle=False)


        logger = Logger(os.path.join(config.snapshot_dir, config.split["test"] + ".log"))

        num_iter =  len(test_set)
        c_loader_iter = test_loader.__iter__()

        for c_iter in tqdm(range(num_iter)):

            inputs = c_loader_iter.next()


            for k, v in inputs.items():
                if type(v) == list:
                    inputs [k] = [item.to(config.device) for item in v]
                elif type(v) in [dict, float, type(None), np.ndarray]:
                    pass
                else:
                    inputs [k] = v.to(config.device)


            """predict landmarks"""
            ldmk_s, ldmk_t, inlier_rate, inlier_rate_2 = ldmk_model.inference (inputs, reject_outliers=config.reject_outliers, inlier_thr=config.inlier_thr, timer=timer)


            src_pcd, tgt_pcd = inputs["src_pcd_list"][0], inputs["tgt_pcd_list"][0]
            s2t_flow = inputs['sflow_list'][0]
            rot, trn = inputs['batched_rot'][0],  inputs['batched_trn'][0]
            correspondence = inputs['correspondences_list'][0]


            """compute scene flow GT"""
            src_pcd_deformed = src_pcd + s2t_flow
            s_pc_wrapped = ( rot @ src_pcd_deformed.T + trn ).T
            s2t_flow = s_pc_wrapped - src_pcd
            flow_gt = s2t_flow.to(config.device)


            """compute overlap mask"""
            overlap = torch.zeros(len(src_pcd))
            overlap[correspondence[:, 0]] = 1
            overlap = overlap.bool()
            overlap =  overlap.to(config.device)



            if config.deformation_model in ["NDP"]:
                model.load_pcds(src_pcd, tgt_pcd, landmarks=(ldmk_s, ldmk_t))

                timer.tic("registration")
                warped_pcd, iter, timer = model.register(visualize=args.visualize, timer = timer)
                timer.toc("registration")
                flow = warped_pcd - model.src_pcd

                for key, value in iter.items():
                    timer.tictoc(key, value)


            elif config.deformation_model == "ED": # Lepard+NICP

                model.load_pcds(src_pcd, tgt_pcd)

                depth_paths = inputs['depth_paths_list'][0]
                cam_intrin = inputs['cam_intrin']

                # get pixel landmarks
                uv_src = xyz_2_uv(ldmk_s, cam_intrin)
                uv_tgt = xyz_2_uv(ldmk_t, cam_intrin)
                landmarks = (uv_src.to(config.device), uv_tgt.to(config.device))


                timer.tic("graph construction")
                model.load_raw_pcds_from_depth(depth_paths[0], depth_paths[1], cam_intrin, landmarks=landmarks)
                timer.toc("graph construction")


                timer.tic("registration")
                warped_pcd, point_mask = model.register(visualize=args.visualize)
                timer.toc("registration")

                flow = warped_pcd - model.src_pcd[point_mask]
                flow_gt = flow_gt[point_mask]
                overlap = overlap[point_mask]


            else:
                raise KeyError()



            metric_info = compute_flow_metrics(flow, flow_gt, overlap=overlap)


            if stats_meter is None:
                stats_meter = dict()
                for key, _ in metric_info.items():
                    stats_meter[key] = AverageMeter()
            for key, value in metric_info.items():
                stats_meter[key].update(value)




        message = f'{c_iter}/{len(test_set)}: '
        for key, value in stats_meter.items():
            message += f'{key}: {value.avg:.3f}\t'
        logger.write(message + '\n')

        print("score on ", split, '\n', message)




    # note down average time cost
    print('time cost average')
    for ele in timer.get_strings():
        logger.write(ele + '\n')
        print(ele)
