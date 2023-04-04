import itertools
import os
import os.path as osp
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from models import *
from models.decode import mot_decode
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.utils import *
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process, reback_ctdet_post_process

from tracker import matching
from .common import intersect_dicts 
from .basetrack import Model, SiamMot
from .basetrack import BaseTrack, TrackState

import seaborn as sns
import copy


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.score_deque = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        # self.score_deque.append(new_track.score)
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    #训练好模型之后就可以开始跟踪流程，FairMOT跟踪流程与Deep Sort类似，直接看代码：
    def __init__(self, opt, frame_rate=30):   #初始化轨迹
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        
        # ckpt = torch.load(opt.weights, map_location=opt.device)  # load checkpoint
        # model_base = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=1).to(opt.device)  # create
        # self.model = SiamMot(self.opt,model_base)  # create
        # exclude = ['anchor'] if opt.cfg else []  # exclude keys
        # if type(ckpt['model']).__name__ == "OrderedDict":
        #     state_dict = ckpt['model']
        # else:
        #     state_dict = ckpt['model'].float().state_dict()  # to FP32
        # state_dict = intersect_dicts(state_dict, self.model.state_dict(), exclude=exclude)  # intersect
        # self.model.load_state_dict(state_dict, strict=False)  # load
        # self.model.cuda().eval()
        
        # total_params = sum(p.numel() for p in self.model.parameters())
        # print(f'{total_params:,} total parameters.')
        
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        # self.det_thresh = opt.conf_thres + 0.1
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]
    
    def reback_post_process(self, pre_tracks, meta_reback, meta):
        pre_tracks_det = pre_tracks.tlbr
        center_inds = reback_ctdet_post_process(pre_tracks_det.copy(), [meta_reback['c']], [meta_reback['s']],
                                                meta_reback['out_height'], meta_reback['out_width'], meta['out_height'], meta['out_width'])
        return center_inds    

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    # def update(self, im_blob, img0, seq_num, save_dir):
    def update(self, im_blob, img0):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        dets = []

        width = img0.shape[1] # 1920
        height = img0.shape[0] # 1080
        inp_height = im_blob.shape[2] # 608
        inp_width = im_blob.shape[3] # 1088
        c = np.array([width / 2., height / 2.], dtype=np.float32) # [960, 540]
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0 
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        
        c_back = np.array([meta['out_width']/2., meta['out_height']/2.], dtype=np.float32)
        s_back = max(float(width)/float(height)*meta['out_height'], meta['out_width']) * 1.0
        meta_reback = {'c': c_back, 's': s_back,
                        'out_height': height,
                        'out_width': width}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]    #检查网络输出结果
            hm = output['hm'].sigmoid_()        #检查网络的输出热力图
            wh = output['wh']                   #检查网络的目标宽高值
            id_feature = output['id']           #检查网络输出的Re-ID特征
            id_feature_net = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None   #检查网络的输出偏移量
            # dets, inds = mot_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
            
            id_feature = _tranpose_and_gather_feat(id_feature_net, inds)
            id_feature = id_feature.squeeze(0)
            id_feature_origin = id_feature.cpu().numpy()
            # pred, train_out = output[1]

        #后处理
        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]
        # inds_ = inds.cpu().numpy().reshape(-1,1)

        #置信度过滤
        remain_inds = dets[:, 4] > self.opt.conf_thres
        inds_low = dets[:, 4] > 0.2
        inds_high = dets[:, 4] < self.opt.conf_thres
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = dets[inds_second]
        id_feature_second = id_feature_origin[inds_second]
        dets = dets[remain_inds]
        id_feature = id_feature_origin[remain_inds]
        
        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []        
        # pred = pred[pred[:, :, 4] > self.opt.conf_thres]
        # detections = []
        # if len(pred) > 0:
        #     dets,x_inds,y_inds = non_max_suppression_and_inds(pred[:,:6].unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres,method='cluster_diou')
        #     if len(dets) != 0:
        #         scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
        #         id_feature = output[0][0, y_inds, x_inds, :].cpu().numpy()

        #         detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
        #                       (tlbrs, f) in zip(dets[:, :5], id_feature)]
        #     else:
        #         detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

#这里首先通过backbone获取到对应的各个head的输出，接着进行后处理及置信度过滤（NMS），将新的目标加入轨迹。

        ''' Step 2: First association, with embedding/CIOU'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool) # 卡尔曼预测
        dists_embedding = matching.embedding_distance(strack_pool, detections) # 计算新检测出来的目标和tracked_tracker之间的cosine距离
        dists_ciou = matching.iou_distance(strack_pool, detections, "ciou")
        dists_ciou = dists_ciou/2
        dists_fuse = dists_embedding + dists_ciou        
        #dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        # dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections) # 利用卡尔曼计算detection和pool_stacker直接的距离代价索引，
        matches, u_track, u_detection = matching.linear_assignment(dists_fuse, thresh=0.7)
        # matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7) # 匈牙利匹配 // 将跟踪框和检测框进行匹配 // u_track是未匹配的tracker的索引，

        for itracked, idet in matches: # matches:63*2 , 63:detections的维度，2：第一列为tracked_tracker索引，第二列为detection的索引
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id) # 匹配的pool_tracker和detection，更新特征和卡尔曼状态
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False) # 如果是在lost中的，就重新激活
                refind_stracks.append(track)
#这里同时应用了Kalman Filter来筛选detection，如果预测的跟踪框与检测框距离过远，则将其cost设为无穷大，
# 接着通过特征余弦距离以及马氏距离来作为数据关联的cost，接着通过匈牙利算法来进行关联，然后进行轨迹的更新。

        '''vis'''
        # track_features, det_features, cost_matrix, cost_matrix_det, cost_matrix_track = [],[],[],[],[]
        # if self.opt.vis_state == 1 and self.frame_id % 20 == 0:
        #     if len(dets) != 0:
        #         for i in range(0, dets.shape[0]):
        #             bbox = dets[i][0:4]
        #             cv2.rectangle(img0, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0, 255, 0), 2)
        #         track_features, det_features, cost_matrix, cost_matrix_det, cost_matrix_track = matching.vis_id_feature_A_distance(strack_pool, detections)
        #     vis_feature(self.frame_id,seq_num,img0,track_features,
        #                           det_features, cost_matrix, cost_matrix_det, cost_matrix_track, max_num=5, out_path=save_dir)

        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                                 (tlbrs, f) in zip(dets_second[:, :5], id_feature_second)] # 处于0.2到conf_thres之间的进行二次匹配
        else:
            detections_second = []
        
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists_ciou = matching.iou_distance(r_tracked_stracks, detections_second, 'ciou')
        dists_ciou = dists_ciou/2
        dists_embedding = matching.embedding_distance(r_tracked_stracks, detections_second)
        dists_fuse = dists_ciou + dists_embedding
        matches, u_track, u_detection_second = matching.linear_assignment(dists_fuse, thresh=0.4)

        
        ''' Step 3: Second association, with IOU'''
        # detections = [detections[i] for i in u_detection]  # u_detection是未匹配的detection的索引
        # r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # dists = matching.iou_distance(r_tracked_stracks, detections)
        # matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            # det = detections[idet]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False) # 前面已经限定了是TrackState.Tracked，这里是不用运行到的。
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track) # 将和tracked_tracker iou未匹配的tracker的状态改为lost
                                            #将未匹配上的检测框与跟踪框再通过IOU进行一次匹配，防止遗漏的目标
                                            
        # reback
        for it in u_track:
            track = r_tracked_stracks[it]
            ## 最后进行一次找回
            track_index = self.reback_post_process(track, meta_reback, meta)
            if 0 < track_index < meta['out_width']*meta['out_height']:
                with torch.no_grad():
                    reback_feature = _tranpose_and_gather_feat(id_feature_net, torch.tensor(track_index[None,:], dtype=torch.int64).cuda())
                    reback_feature = reback_feature.squeeze(0)
                    reback_feature = reback_feature.cpu().numpy()
                dists_reback = matching.embedding_distance_track_back([track], reback_feature)
                if dists_reback[0] < 0.1:
                    det = STrack(STrack.tlbr_to_tlwh(track.tlbr), track.score, reback_feature[0,:], 2) 
                    if track.state == TrackState.Tracked:
                        track.update(det, self.frame_id)
                        activated_starcks.append(track)
                    else:
                        track.re_activate(det, self.frame_id, new_id=False)
                        refind_stracks.append(track)
                    # print("dists_reback",dists_reback)
                    continue
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]  # 将cosine/iou未匹配的detection和unconfirmed_tracker进行匹配
        dists = matching.iou_distance(unconfirmed, detections, 'iou')
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        '''接着处理未匹配上的轨迹（通常是新增目标）'''

        """ Step 4: Init new stracks初始化新增目标轨迹并更新所有轨迹状态"""
        for inew in u_detection: # 对cosine/iou/uncofirmed_tracker都未匹配的detection重新初始化一个unconfimed_tracker
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id) # 激活track，第一帧的activated=T，其他为False
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed() # 消失15帧之后
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb, 'iou')
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


def remove_fp_stracks(stracksa, n_frame=10):
    remain = []
    for t in stracksa:
        score_5 = t.score_deque
        score_5 = np.array(score_5, dtype=np.float32)
        score_5 = score_5[-n_frame:]
        index = score_5 < 0.35
        num = np.sum(index)
        if num < n_frame:
            remain.append(t)
    return remain


def vis_feature(frame_id,seq_num,img,track_features, det_features, cost_matrix, cost_matrix_det, cost_matrix_track,max_num=5, out_path='/home/XX/'):
    num_zero = ["0000","000","00","0"]
    img = cv2.resize(img, (778, 435))

    if len(det_features) != 0:
        max_f = det_features.max()
        min_f = det_features.min()
        det_features = np.round((det_features - min_f) / (max_f - min_f) * 255)
        det_features = det_features.astype(np.uint8)
        d_F_M = []
        cutpff_line = [40]*512
        for d_f in det_features:
            for row in range(45):
                d_F_M += [[40]*3+d_f.tolist()+[40]*3]
            for row in range(3):
                d_F_M += [[40]*3+cutpff_line+[40]*3]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        det_features_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(det_features_img, (435, 435))
        #cv2.putText(feature_img2, "det_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "det_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((img, feature_img2), axis=1)

    if len(cost_matrix_det) != 0 and len(cost_matrix_det[0]) != 0:
        max_f = cost_matrix_det.max()
        min_f = cost_matrix_det.min()
        cost_matrix_det = np.round((cost_matrix_det - min_f) / (max_f - min_f) * 255)
        d_F_M = []
        cutpff_line = [40]*len(cost_matrix_det)*10
        for c_m in cost_matrix_det:
            add = []
            for row in range(len(c_m)):
                add += [255-c_m[row]]*10
            for row in range(10):
                d_F_M += [[40]+add+[40]]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        cost_matrix_det_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(cost_matrix_det_img, (435, 435))
        #cv2.putText(feature_img2, "cost_matrix_det", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "cost_matrix_det", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    if len(track_features) != 0:
        max_f = track_features.max()
        min_f = track_features.min()
        track_features = np.round((track_features - min_f) / (max_f - min_f) * 255)
        track_features = track_features.astype(np.uint8)
        d_F_M = []
        cutpff_line = [40]*512
        for d_f in track_features:
            for row in range(45):
                d_F_M += [[40]*3+d_f.tolist()+[40]*3]
            for row in range(3):
                d_F_M += [[40]*3+cutpff_line+[40]*3]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        track_features_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(track_features_img, (435, 435))
        #cv2.putText(feature_img2, "track_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "track_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    if len(cost_matrix_track) != 0 and len(cost_matrix_track[0]) != 0:
        max_f = cost_matrix_track.max()
        min_f = cost_matrix_track.min()
        cost_matrix_track = np.round((cost_matrix_track - min_f) / (max_f - min_f) * 255)
        d_F_M = []
        cutpff_line = [40]*len(cost_matrix_track)*10
        for c_m in cost_matrix_track:
            add = []
            for row in range(len(c_m)):
                add += [255-c_m[row]]*10
            for row in range(10):
                d_F_M += [[40]+add+[40]]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        cost_matrix_track_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(cost_matrix_track_img, (435, 435))
        #cv2.putText(feature_img2, "cost_matrix_track", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "cost_matrix_track", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    if len(cost_matrix) != 0 and len(cost_matrix[0]) != 0:
        max_f = cost_matrix.max()
        min_f = cost_matrix.min()
        cost_matrix = np.round((cost_matrix - min_f) / (max_f - min_f) * 255)
        d_F_M = []
        cutpff_line = [40]*len(cost_matrix[0])*10
        for c_m in cost_matrix:
            add = []
            for row in range(len(c_m)):
                add += [255-c_m[row]]*10
            for row in range(10):
                d_F_M += [[40]+add+[40]]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        cost_matrix_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(cost_matrix_img, (435, 435))
        #cv2.putText(feature_img2, "cost_matrix", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "cost_matrix", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    dst_path = out_path + "/" + seq_num + "_" + num_zero[len(str(frame_id))-1] + str(frame_id) + '.png'
    cv2.imwrite(dst_path, feature_img)
    
    