# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import caffe
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors,generate_vertical_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform

DEBUG = False

class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        #self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._feat_stride = layer_params['feat_stride']
        self._anchors = generate_vertical_anchors(self._feat_stride) # CTPN Xiangyu zhu
        self._num_anchors = self._anchors.shape[0]
        

        if DEBUG:
            print 'anchors:'
            print self._anchors
            print 'anchor shapes:'
            print np.hstack((
                self._anchors[:, 2::4] - self._anchors[:, 0::4],
                self._anchors[:, 3::4] - self._anchors[:, 1::4],
            ))
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = layer_params.get('allowed_border', 0)

        height, width = bottom[0].data.shape[-2:]
        if DEBUG:
            print 'AnchorTargetLayer: height', height, 'width', width

        A = self._num_anchors
        # labels
        top[0].reshape(1, 1, A * height, width)
        # bbox_targets
        top[1].reshape(1, A * 4, height, width)
        # bbox_inside_weights
        top[2].reshape(1, A * 4, height, width)
        # bbox_outside_weights
        top[3].reshape(1, A * 4, height, width)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = bottom[0].data.shape[-2:]
        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[1].data
        # im_info
        im_info = bottom[2].data[0, :]
        
        # CTPN ZXY divide GT into vertical GT
        gt_boxes_space = _get_divided_gt_neg_box_from_gt_box(gt_boxes,im_info[1],fixed_width=self._feat_stride) #jyy for space
        gt_boxes = _divide_gt_box(gt_boxes,fixed_width=self._feat_stride)

        if DEBUG:
            print ''
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])
            print 'height, width: ({}, {})'.format(height, width)
            print 'rpn: gt_boxes.shape', gt_boxes.shape
            print 'rpn: gt_boxes', gt_boxes

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = (self._anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)
        
        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -self._allowed_border) &
            (all_anchors[:, 1] >= -self._allowed_border) &
            (all_anchors[:, 2] < im_info[1] + self._allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + self._allowed_border)    # height
        )[0]

        if DEBUG:
            print 'total_anchors', total_anchors
            print 'inds_inside', len(inds_inside)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        if DEBUG:
            print 'anchors.shape', anchors.shape

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        gt_boxes = gt_boxes.reshape(gt_boxes.shape[0], gt_boxes.shape[1])
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
            
        #jyy for space -----------------------------------------
        if gt_boxes_space.size > 0:
            overlaps_space = bbox_overlaps(   
                np.ascontiguousarray(anchors, dtype=np.float),
                np.ascontiguousarray(gt_boxes_space, dtype=np.float))
                
            argmax_overlaps_space = overlaps_space.argmax(axis=1)
            max_overlaps_space = overlaps_space[np.arange(len(inds_inside)), argmax_overlaps_space]
            gt_argmax_overlaps_space = overlaps_space.argmax(axis=0)
            gt_max_overlaps_space = overlaps_space[gt_argmax_overlaps_space,
                                       np.arange(overlaps_space.shape[1])]
            gt_argmax_overlaps_space = np.where(overlaps_space == gt_max_overlaps_space)[0]
    
            # fg label: for each neg gt, anchor with highest overlap
            labels[gt_argmax_overlaps_space] = 2
    
            # fg label: above threshold IOU
            labels[max_overlaps_space >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 2 # 0 jyy             
        #jyy for space -----------------------------------------    
        
        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1
            
        # subsample negative labels from space if we have too many
        num_bg_space = int(0.1 * cfg.TRAIN.RPN_BATCHSIZE) #jyy for space
        bg_inds_space = np.where(labels == 2)[0]
        #print(bg_inds_space, num_bg_space)
        if len(bg_inds_space) > num_bg_space:
            disable_inds = npr.choice(
                bg_inds_space, size=(len(bg_inds_space) - num_bg_space), replace=False)
            labels[disable_inds] = -1 
          
        # subsample negative labels if we have too many
        #num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)-  np.sum(labels == 2) #jyy for space
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
            #print "was %s inds, disabling %s, now %s inds" % (
                #len(bg_inds), len(disable_inds), np.sum(labels == 0))
                
        # jyy for space 
        bg_inds_space = np.where(labels == 2)[0]
        labels[bg_inds_space] = 0
        
        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        if DEBUG:
            self._sums += bbox_targets[labels == 1, :].sum(axis=0)
            self._squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels == 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print 'means:'
            print means
            print 'stdevs:'
            print stds

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        if DEBUG:
            print 'rpn: max max_overlap', np.max(max_overlaps)
            print 'rpn: num_positive', np.sum(labels == 1)
            print 'rpn: num_negative', np.sum(labels == 0)
            self._fg_sum += np.sum(labels == 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            print 'rpn: num_positive avg', self._fg_sum / self._count
            print 'rpn: num_negative avg', self._bg_sum / self._count

        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        top[1].reshape(*bbox_targets.shape)
        top[1].data[...] = bbox_targets

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width
        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width
        top[3].reshape(*bbox_outside_weights.shape)
        top[3].data[...] = bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _divide_gt_box(gt_box,fixed_width = 16):
    divided_gt_box = []
    for box in gt_box:
        start_x = box[0]
        end_x = box[2]
        height = box[3] - box[1]
        start_y = box[1]
    
        start_x = (( (int)(start_x / fixed_width)+ 1)*fixed_width)  
        end_x = (((int)(end_x / fixed_width))*fixed_width)
        if  start_x - box[0] > fixed_width/2:
          d_box = [ box[0], start_y, start_x-1 , box[3], box[4]]
          divided_gt_box.append(d_box)
        while(start_x <end_x):
            d_box = [ start_x, start_y, start_x + fixed_width-1, box[3], box[4]]
            divided_gt_box.append(d_box)
            start_x = start_x + fixed_width
        if box[2] - end_x > fixed_width/2:
          d_box = [ end_x, start_y, box[2] , box[3], box[4]]
          divided_gt_box.append(d_box)
    if len(divided_gt_box)==0:
        divided_gt_box = gt_box
    divided_gt_box = np.array(divided_gt_box,dtype = np.float32)
    return divided_gt_box

def _get_divided_gt_neg_box_from_gt_box(gt_box, img_width, fixed_width=16):
    divided_gt_neg_box = []
    # print 'gt_box',gt_box
    for box1 in gt_box:
        minOffset = 9999999999
        isSingleWord = True
        for box2 in gt_box:
            if box1.tolist() == box2.tolist():
                continue
            if (((min(box1[3], box2[3]) - max(box1[1], box2[1])) > 0.8 * (
                max(box1[3], box2[3]) - min(box1[1], box2[1]))) or (box1[1] < box2[1] and box1[3] > box2[3]) or (
                    box1[1] > box2[1] and box1[3] < box2[3])) and box2[0] > box1[2] + fixed_width and (box2[0] - box1[
                2]) < minOffset:  # (intersection height > 0.8 or inside ) and (space at least 16 wide) and space <minOffset to find nearest neighbor ; box 1 left box2 right
                minOffset = box2[0] - box1[2]
                neighbor_box = box2
            if min(box1[3], box2[3]) - max(box1[1], box2[1]) > 0:  # overlap in vertical
                isSingleWord = False

        if isSingleWord == True and box1[0] - fixed_width > 0 and box1[
            2] + fixed_width < img_width:  # append start and end
            d_box = [box1[0] - fixed_width, box1[1], box1[0] - 1, box1[3], 2]
            divided_gt_neg_box.append(d_box)
            d_box = [box1[2], box1[1], box1[2] + fixed_width - 1, box1[3], 2]
            divided_gt_neg_box.append(d_box)
            #print 'single word'
            #continue

        if minOffset == 9999999999 or minOffset <= 0:  # there is no neighbor in the same textline /  box2 overlap with box1
            continue

        box2 = neighbor_box
        start_x = box1[2]  # + fixed_width
        end_x = box2[0]  # - fixed_width

        #
        start_x = (int(start_x / fixed_width)+1 ) * fixed_width
        end_x = (int(end_x / fixed_width) -1) * fixed_width

        height = box1[3] - box1[1]  # (box1[3]-box1[1]+box2[3]-box2[1])/2
        start_y = box1[1]  # (box1[1]+box2[1])/2

        # start_x = ((int)(start_x / fixed_width)*fixed_width)
        overlap_with_gt = 0
        for box3 in gt_box:  # if box3 is between box2 and box1,continue

             #if min(box1[3],box2[3])-max(box1[1],box2[1])>0 and min(box1[2],box2[2])-max(box1[0],box2[0]) >0:
            if min(start_y + height, box3[3]) - max(start_y, box3[1]) > 0 and min(box3[2], end_x) - max(start_x,
                                                                                                        box3[0]) > 0:
                overlap_with_gt = 1
                break
        if overlap_with_gt == 1:
            continue

        # print 'box1', box1
        # print 'neighbor_box', neighbor_box
        while (start_x < end_x):
            d_box = [start_x, start_y, start_x + fixed_width - 1, start_y + height, 2]
            divided_gt_neg_box.append(d_box)
            start_x = start_x + fixed_width
    divided_gt_neg_box = np.array(divided_gt_neg_box, dtype=np.float32)
    # print 'divided_gt_neg_box', divided_gt_neg_box.shape
    # print 'divided_gt_neg_box', divided_gt_neg_box
    return divided_gt_neg_box  
def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    targets = bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
    if cfg.TRAIN.RPN_NORMALIZE_TARGETS:
        assert cfg.TRAIN.RPN_NORMALIZE_MEANS is not None
        assert cfg.TRAIN.RPN_NORMALIZE_STDS is not None
        targets -= cfg.TRAIN.RPN_NORMALIZE_MEANS
        targets /= cfg.TRAIN.RPN_NORMALIZE_STDS
    return targets
