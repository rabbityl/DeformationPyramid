import torch
import torch.nn as nn
import numpy as np



class NeCoLoss(nn.Module):
    def __init__(self, config):
        super().__init__()


        self.inlier_thr = config['inlier_thr']
        self.mutual_nearest = config['mutual_nearest']
        self.dataset = config['dataset']
        self.balanced = config['balanced_bce']


    def forward(self, data):
        loss_info = {}


        s2t_flow = torch.zeros_like(data['s_pcd'])
        for i, cflow in enumerate(data['coarse_flow']):
            s2t_flow[i][: len(cflow)] = cflow


        ###################################
        # evaluate matcher
        ###################################
        match_pred= data['coarse_match_pred']
        '''Inlier Ratio (IR)'''
        inlier_mask, inlier_rate = self.compute_inlier_mask( data, self.inlier_thr, s2t_flow=s2t_flow)
        loss_info.update({"IR Lepard": torch.stack(inlier_rate).mean()})  # only check the first frame pair in the batch


        vis = False
        if vis:
            self.multiview_corr_vis( data, inlier_mask)

        ###################################
        # compute loss of outlier filter
        ###################################
        labels = torch.cat( inlier_mask )
        inlier_conf= data["inlier_conf"].reshape(-1) [  data["vec_6d_mask"].reshape(-1)]
        loss = self.get_weighted_bce_loss(inlier_conf, labels.float())


        ###################################
        # evaluate metrtics after outlier filter
        ###################################
        ir = []
        for i in range ( len(inlier_mask)) :
            match_filtered_i = inlier_mask[i] [ data["inlier_conf"][i][data["vec_6d_mask"][i]] > 0.5 ]
            if match_filtered_i.shape[0] > 0:
                ir.append( match_filtered_i.sum()/(match_filtered_i.shape[0]))

        if len(ir) > 0:
            loss_info.update({"IR NeCo": torch.stack(ir).mean() })
        else :
            loss_info.update({"IR NeCo": 0 })


        loss_info.update({ 'loss': loss })
        return loss_info




    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')

        class_loss = loss(prediction, gt)

        weights = torch.ones_like(gt)
        w_negative = gt.sum() / gt.size(0)
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        return w_class_loss #, cls_precision, cls_recall




    def compute_correspondence_loss(self, conf, conf_gt, weight=None):
        '''
        @param conf: [B, L, S]
        @param conf_gt: [B, L, S]
        @param weight: [B, L, S]
        @return:
        '''
        pos_mask = conf_gt == 1
        neg_mask = conf_gt == 0

        pos_w, neg_w = self.pos_w, self.neg_w

        #corner case assign a wrong gt
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            neg_w = 0.

        # focal loss
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.focal_alpha
        gamma = self.focal_gamma

        if self.match_type == "dual_softmax":
            pos_conf = conf[pos_mask]
            loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
            loss =  pos_w * loss_pos.mean()
            return loss

        elif self.match_type == "sinkhorn":
            # no supervision on dustbin row & column.
            loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
            loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
            loss = pos_w * loss_pos.mean() + neg_w * loss_neg.mean()
            return loss

    def match_2_conf_matrix(self, matches_gt, matrix_pred):
        matrix_gt = torch.zeros_like(matrix_pred)
        for b, match in enumerate (matches_gt) :
            matrix_gt [ b][ match[0],  match[1] ] = 1
        return matrix_gt


    @staticmethod
    def compute_match_recall(conf_matrix_gt, match_pred) : #, s_pcd, t_pcd, search_radius=0.3):
        '''
        @param conf_matrix_gt:
        @param match_pred:
        @return:
        '''

        pred_matrix = torch.zeros_like(conf_matrix_gt)

        b_ind, src_ind, tgt_ind = match_pred[:, 0], match_pred[:, 1], match_pred[:, 2]
        pred_matrix[b_ind, src_ind, tgt_ind] = 1.

        true_positive = (pred_matrix == conf_matrix_gt) * conf_matrix_gt

        recall = true_positive.sum() / conf_matrix_gt.sum()

        precision = true_positive.sum() / max(len(match_pred), 1)

        return recall, precision



    @staticmethod
    def compute_inlier_mask( data, inlier_thr, s2t_flow=None):

        s_pcd, t_pcd = data['s_pcd'], data['t_pcd'] #B,N,3
        batched_rot = data['batched_rot'] #B,3,3
        batched_trn = data['batched_trn']
        bsize = len(s_pcd)

        s_pcd_deformed = s_pcd + s2t_flow
        s_pcd_wrapped = (torch.matmul(batched_rot, s_pcd_deformed.transpose(1, 2)) + batched_trn).transpose(1,2)


        batch_vec6d = data['vec_6d']
        batch_mask = data['vec_6d_mask']
        batch_index = data['vec_6d_ind']

        inlier_rate = []
        inlier_mask = []

        for i in range(bsize):

            s_pcd_match_warp_gt = s_pcd_wrapped[i][batch_index[i][:,0]] [batch_mask[i]]
            t_pcd_matched = batch_vec6d[i][:,3:] [batch_mask[i]]
            inlier = torch.sum( (s_pcd_match_warp_gt - t_pcd_matched)**2 , dim= 1) <  inlier_thr**2

            inlier_rate.append(inlier.sum().float() / t_pcd_matched.shape[0])
            inlier_mask.append(inlier)


        return  inlier_mask, inlier_rate





    @staticmethod
    def tensor2numpy(tensor):
        if tensor.requires_grad:
            tensor=tensor.detach()
        return tensor.cpu().numpy()



    def multiview_corr_vis(self, data, inlier_mask):

        def viz_coarse_nn_correspondence_mayavi(s_pc, t_pc, good_c, bad_c, f_src_pcd=None, f_tgt_pcd=None,
                                                scale_factor=0.02):

            import mayavi.mlab as mlab
            c_red = (224. / 255., 0 / 255., 0 / 255.)
            c_pink = (224. / 255., 75. / 255., 232. / 255.)
            c_blue = (0. / 255., 0. / 255., 255. / 255.)
            c_green = (0. / 255., 255. / 255., 0. / 255.)
            c_gray1 = (255 / 255., 255 / 255., 125 / 255.)
            c_gray2 = (175. / 255., 175. / 255., 175. / 255.)

            if f_src_pcd is not None:
                mlab.points3d(f_src_pcd[:, 0], f_src_pcd[:, 1], f_src_pcd[:, 2], scale_factor=scale_factor * 0.3,
                              color=c_gray2)
            else:
                mlab.points3d(s_pc[:, 0], s_pc[:, 1], s_pc[:, 2], scale_factor=scale_factor * 0.75, color=c_gray2)

            if f_tgt_pcd is not None:
                mlab.points3d(f_tgt_pcd[:, 0], f_tgt_pcd[:, 1], f_tgt_pcd[:, 2], scale_factor=scale_factor * 0.3,
                              color=c_gray2)
            else:
                mlab.points3d(t_pc[:, 0], t_pc[:, 1], t_pc[:, 2], scale_factor=scale_factor * 0.75, color=c_gray2)

            s_cpts_god = s_pc[good_c[:,0]]
            t_cpts_god = t_pc[good_c[:,1]]
            flow_good = t_cpts_god - s_cpts_god



            def match_draw(s_cpts, t_cpts, flow, color):

                mlab.points3d(s_cpts[:, 0], s_cpts[:, 1], s_cpts[:, 2], scale_factor=scale_factor * 0.5, color=c_blue)
                mlab.points3d(t_cpts[:, 0], t_cpts[:, 1], t_cpts[:, 2], scale_factor=scale_factor * 0.5, color=c_pink)
                mlab.quiver3d(s_cpts[:, 0], s_cpts[:, 1], s_cpts[:, 2], flow[:, 0], flow[:, 1], flow[:, 2],
                              scale_factor=1, mode='2ddash', line_width=1., color=color)




            match_draw(s_cpts_god, t_cpts_god, flow_good, c_green)

            if len( bad_c) > 0 :
                s_cpts_bd = s_pc[bad_c[:,0]]
                t_cpts_bd = t_pc[bad_c[:,1]]
                flow_bad = t_cpts_bd - s_cpts_bd
                match_draw(s_cpts_bd, t_cpts_bd, flow_bad, c_red)

            # mlab.show()




        import mayavi.mlab as mlab

        mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))

        src_pcd_list = data['src_pcd_list']
        tgt_pcd_list = data['tgt_pcd_list']
        s_pcd = data['s_pcd'].cpu().numpy()
        t_pcd = data['t_pcd'].cpu().numpy()


        edge = data['pcd_pairs']
        batch_vec6d = data['vec_6d']
        batch_mask = data['vec_6d_mask']
        n_pairs, n_match = batch_mask.shape
        n_pcd = edge.max().cpu().numpy() + 1


        """apply offset"""
        radius = 1.5
        pi = 3.14
        div = pi*2/n_pcd
        angels = np.array( [ div * i for i in range ( n_pcd )] )
        x_offset = np.cos(angels) * radius
        y_offset = np.sin(angels) * radius
        z_offset = np.zeros_like(x_offset)
        offsets = np.stack( [x_offset, y_offset, z_offset], axis=-1)

        for i, e in enumerate ( edge ):
            # i,j = e
            src_pcd_list[i] =  src_pcd_list[i].cpu().numpy() + offsets[e[0]:e[0]+1]
            tgt_pcd_list[i] =  tgt_pcd_list[i].cpu().numpy() + offsets[e[1]:e[1]+1]
            s_pcd[i] = s_pcd[i] + offsets[e[0]:e[0]+1]
            t_pcd[i] = t_pcd[i] + offsets[e[1]:e[1]+1]


        batch_mask = data['vec_6d_mask'].cpu().numpy()
        batch_index = data['vec_6d_ind'].cpu().numpy()
        inlier_mask = inlier_mask


        max = 25

        for i in range(len(s_pcd)):
            corrs = batch_index[i][batch_mask[i]]
            ir_m = inlier_mask[i].cpu().numpy()

            if len(corrs)>max:

                perm = np.random.permutation(corrs.shape[0])
                ind = perm[:max]

                corrs = corrs[ ind]
                ir_m = ir_m[ind]


            good_c = corrs[ir_m]
            bad_c = corrs[~ir_m]

            viz_coarse_nn_correspondence_mayavi (s_pcd[i], t_pcd[i], good_c, bad_c, f_src_pcd=src_pcd_list[i], f_tgt_pcd=tgt_pcd_list[i], scale_factor=0.05 )


        mlab.show()
