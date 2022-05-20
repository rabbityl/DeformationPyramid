from lib.trainer import Trainer
import torch
from tqdm import tqdm
from lepard.loss import MatchMotionLoss as MML
import numpy as np
from lepard.matching import Matching as CM
import math
from lib.benchmark_utils import to_o3d_pcd
import open3d as o3d


def blend_anchor_motion (query_loc, reference_loc, reference_flow , knn=3, search_radius=0.1) :
    '''approximate flow on query points
    this function assume query points are sub- or un-sampled from reference locations
    @param query_loc:[m,3]
    @param reference_loc:[n,3]
    @param reference_flow:[n,3]
    @param knn:
    @return:
        blended_flow:[m,3]
    '''
    from datasets.utils import knn_point_np
    dists, idx = knn_point_np (knn, reference_loc, query_loc)
    dists[dists < 1e-10] = 1e-10
    mask = dists>search_radius
    dists[mask] = 1e+10
    weight = 1.0 / dists
    weight = weight / np.sum(weight, -1, keepdims=True)  # [B,N,3]
    blended_flow = np.sum (reference_flow [idx] * weight.reshape ([-1, knn, 1]), axis=1, keepdims=False)

    mask = mask.sum(axis=1)<3

    return blended_flow, mask

def compute_nrfmr( match_pred, data, recall_thr=0.04):


    s_pcd, t_pcd = data['s_pcd'], data['t_pcd']

    s_pcd_raw = data ['src_pcd_list']
    sflow_list = data['sflow_list']
    metric_index_list = data['metric_index_list']

    batched_rot = data['batched_rot']  # B,3,3
    batched_trn = data['batched_trn']


    nrfmr = 0.

    for i in range ( len(s_pcd_raw)):

        # get the metric points' transformed position
        metric_index = metric_index_list[i]
        sflow = sflow_list[i]
        s_pcd_raw_i = s_pcd_raw[i]
        metric_pcd = s_pcd_raw_i [ metric_index ]
        metric_sflow = sflow [ metric_index ]
        metric_pcd_deformed = metric_pcd + metric_sflow
        metric_pcd_wrapped_gt = ( torch.matmul( batched_rot[i], metric_pcd_deformed.T) + batched_trn[i] ).T


        # use the match prediction as the motion anchor
        match_pred_i = match_pred[ match_pred[:, 0] == i ]
        s_id , t_id = match_pred_i[:,1], match_pred_i[:,2]
        s_pcd_matched= s_pcd[i][s_id]
        t_pcd_matched= t_pcd[i][t_id]
        motion_pred = t_pcd_matched - s_pcd_matched
        metric_motion_pred, valid_mask = blend_anchor_motion(
            metric_pcd.cpu().numpy(), s_pcd_matched.cpu().numpy(), motion_pred.cpu().numpy(), knn=3, search_radius=0.1)
        metric_pcd_wrapped_pred = metric_pcd + torch.from_numpy(metric_motion_pred).to(metric_pcd)

        debug = False
        if debug:
            import mayavi.mlab as mlab
            c_red = (224. / 255., 0 / 255., 125 / 255.)
            c_pink = (224. / 255., 75. / 255., 232. / 255.)
            c_blue = (0. / 255., 0. / 255., 255. / 255.)
            scale_factor = 0.013
            metric_pcd_wrapped_gt = metric_pcd_wrapped_gt.cpu()
            metric_pcd_wrapped_pred = metric_pcd_wrapped_pred.cpu()
            err = metric_pcd_wrapped_pred - metric_pcd_wrapped_gt
            mlab.points3d(metric_pcd_wrapped_gt[:, 0], metric_pcd_wrapped_gt[:, 1], metric_pcd_wrapped_gt[:, 2], scale_factor=scale_factor, color=c_pink)
            mlab.points3d(metric_pcd_wrapped_pred[ :, 0] , metric_pcd_wrapped_pred[ :, 1], metric_pcd_wrapped_pred[:,  2], scale_factor=scale_factor , color=c_blue)
            mlab.quiver3d(metric_pcd_wrapped_gt[:, 0], metric_pcd_wrapped_gt[:, 1], metric_pcd_wrapped_gt[:, 2], err[:, 0], err[:, 1], err[:, 2],
                          scale_factor=1, mode='2ddash', line_width=1.)
            mlab.show()

        dist = torch.sqrt( torch.sum( (metric_pcd_wrapped_pred - metric_pcd_wrapped_gt)**2, dim=1 ) )

        r = (dist < recall_thr).float().sum() / len(dist)
        nrfmr = nrfmr + r

    nrfmr = nrfmr /len(s_pcd_raw)

    return  nrfmr

class _4DMatchTester(Trainer):
    """
    3DMatch tester
    """
    def __init__(self,args):
        Trainer.__init__(self, args)

    def test(self):

        # for thr in [  0.05, 0.1, 0.2]:
        for thr in [ 0.1 ]:
            import time
            start = time.time()
            ir, fmr, nspl = self.test_thr(thr)
            print( "conf_threshold", thr,  "NFMR:", fmr, " Inlier rate:", ir, "Number sample:", nspl)
            print( "time costs:", time.time() - start)

    def test_thr(self, conf_threshold=None):

        num_iter = math.ceil(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        c_loader_iter = self.loader['test'].__iter__()


        self.matcher.eval()
        self.model.eval()


        assert self.loader['test'].batch_size == 1

        IR=0.
        NR_FMR=0.

        inlier_thr = recall_thr = 0.04

        n_sample = 0.

        with torch.no_grad():
            for idx in tqdm(range(num_iter)): # loop through this epoch


                ##################################
                if self.timers: self.timers.tic('load batch')
                inputs = c_loader_iter.next()
                for k, v in inputs.items():
                    if type(v) == list:
                        inputs[k] = [item.to(self.device) for item in v]
                    elif type(v) in [ dict, float, type(None), np.ndarray]:
                        pass
                    else:
                        inputs[k] = v.to(self.device)
                if self.timers: self.timers.toc('load batch')
                ##################################

                with torch.no_grad():
                    if self.timers: self.timers.tic('matcher')
                    data = self.matcher(inputs, timers=None)  # [N1, C1], [N2, C2]
                    # covert matched 3D pairs to 6D points
                    ind = data['coarse_match_pred']
                    bi, si, ti = ind[:, 0], ind[:, 1], ind[:, 2]
                    s_pos = data['s_pcd'][0][si]
                    t_pos = data['t_pcd'][0][ti]
                    vec_6d = torch.cat([s_pos, t_pos], dim=1)[None]
                    data['vec_6d'] = vec_6d
                    if self.timers: self.timers.toc('matcher')


                    if self.timers: self.timers.tic('inlier model')
                    confidence = self.model(data)
                    data["inlier_conf"] = confidence
                    if self.timers: self.timers.toc('inlier model')

                    if self.timers: self.timers.tic('backprop')
                    loss_info = self.loss(data)
                    if self.timers: self.timers.toc('backprop')







def get_trainer(config):
    if config.dataset == '3dmatch':
        return None #_3DMatchTester(config)
    elif config.dataset == '4dmatch' or  config.dataset == '4dmatch_mv':
        return _4DMatchTester(config)
    else:
        raise NotImplementedError
