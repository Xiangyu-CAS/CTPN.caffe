#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
import matplotlib

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt

from matplotlib.pyplot import plot,savefig

import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

from text_proposal_connector import TextProposalConnector
#from cpu_nms import cpu_nms
#from cfg import Config as cfg

connect_proposals = True

CLASSES = ('__background__',
           'text')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'jyz_zf': ('ZF',
                    'zf_faster_rcnn_iter_5000.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

#    im = im[:, :, (2, 1, 0)]
    

 #   fig, ax = plt.subplots(figsize=(12, 12))
#    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        cv2.rectangle(im,(int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]),int(bbox[3])),
                      (0,255,0),3)

def write_4points_result(path,image_name,dets,thresh):
    f=open(os.path.join(path, 'res_' + image_name.replace('.jpg', '.txt')), 'w')
    inds = np.where(dets[:, -1] >= thresh)[0]
    for i in inds:
        bbox = dets[i, :4]
        k = str(int(bbox[0])) + ',' + str(int(bbox[1]))+ ',' +str(int(bbox[2])) + ','+ str(int(bbox[1]))+ ',' +str(int(bbox[2]))+ ',' + str(int(bbox[3])) + ','+ str(int(bbox[0]))+ ',' +str(int(bbox[3]))
        #k = ','.join([str(int(num)) for num in bbox])
        f.write(k + '\r\n')
    f.close()

def write_result(path,image_name,dets,thresh):
    f=open(os.path.join(path, 'res_' + image_name.replace('.jpg', '.txt')), 'w')
    inds = np.where(dets[:, -1] >= thresh)[0]
    for i in inds:
        bbox = dets[i, :4]
        k = ','.join([str(int(num)) for num in bbox])
        f.write(k + '\r\n')
    f.close()
    
def filter_boxes(boxes):
    heights=boxes[:, 3]-boxes[:, 1]+1
    widths=boxes[:, 2]-boxes[:, 0]+1
    scores=boxes[:, -1]
    LINE_MIN_SCORE=0.7 
    TEXT_PROPOSALS_MIN_SCORE=0.7
    TEXT_PROPOSALS_NMS_THRESH=0.3
    MAX_HORIZONTAL_GAP=50
    TEXT_LINE_NMS_THRESH=0.3
    MIN_NUM_PROPOSALS=2
    MIN_RATIO=1.2
    MIN_V_OVERLAPS=0.7#0.7
    MIN_SIZE_SIM=0.7
    TEXT_PROPOSALS_WIDTH=16
    
    MIN_RATIO=0.5 #0.5
    LINE_MIN_SCORE=0.9
    MIN_NUM_PROPOSALS=1
    return np.where((widths/heights>MIN_RATIO) & (scores>LINE_MIN_SCORE) &
          (widths>(TEXT_PROPOSALS_WIDTH*MIN_NUM_PROPOSALS)))[0]
#    LINE_MIN_SCORE=0.8
#    MIN_RATIO=0.8
#    return np.where((widths/heights>MIN_RATIO) & (scores>LINE_MIN_SCORE))
          
def demo(net, img_dir, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im_file = os.path.join(img_dir,image_name)
    im = cv2.imread(im_file)
    
    count = 0;
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    #print scores
    # Visualize detections for each class
    CONF_THRESH = 0.7
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        #cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        #print 'scores',scores
        #cls_scores =  scores
        print 'scores',scores.shape
        print 'cls_boxes',cls_boxes.shape
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
#faster-rcnn nms
        keep = np.where(dets[:, -1] >= CONF_THRESH)[0]
        dets = dets[keep,:]
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        
        
        count  = count + 1
        if connect_proposals:
          text_proposal_connector = TextProposalConnector()
          print 'dets',dets.shape
          text_lines=text_proposal_connector.get_text_lines(dets[:,0:4], dets[:,-1], im.shape[:2])
          keep_inds=filter_boxes(text_lines)
          text_lines=text_lines[keep_inds]
          if text_lines.shape[0]!=0:
            keep_inds=nms(text_lines, NMS_THRESH)
            text_lines=text_lines[keep_inds]

          dets = text_lines

       
        # ctpn nms
#        text_proposals = dets
#        
#        keep_inds=cpu_nms(np.hstack((text_proposals, cls_scores)), cfg.TEXT_PROPOSALS_MIN_SCORE)
#        text_proposals, cls_scores=text_proposals[keep_inds], cls_scores[keep_inds]
#        text_proposal_connector = TextProposalConnector()
#        text_lines=text_proposal_connector.get_text_lines(dets, cls_scores, im.shape[:2])
#        print 'text_lines',text_lines
#        print 'dets',dets
#        if text_lines.shape[0]!=0:
#            keep_inds=nms(text_lines, cfg.TEXT_LINE_NMS_THRESH)
#            text_lines=text_lines[keep_inds]
        
        vis_detections(im, cls, dets, thresh=CONF_THRESH) # vertical anchor 
        #vis_detections(im, cls, text_lines, thresh=CONF_THRESH)
        #cv2.imwrite(os.path.join(os.path.join(cfg.DATA_DIR, 'result'), image_name) ,im)
        #write_result(os.path.join(cfg.DATA_DIR, 'result'), image_name, dets,CONF_THRESH)
        
        cv2.imwrite(os.path.join(os.path.join(cfg.DATA_DIR, 'result'), image_name) ,im)
        #write_4points_result(os.path.join(cfg.DATA_DIR, 'result'), image_name, dets,CONF_THRESH)
        write_result(os.path.join(cfg.DATA_DIR, 'result'), image_name, dets,CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()


    prototxt = '../models/text_voc/VGG16/ctpn/test.prototxt'
  
    caffemodel = '../output/faster_rcnn_end2end/text_voc/vgg16_ctpn_iter_70000.caffemodel'
    


    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)


    # im_dir = '/home/xiangyuzhu/data/ICDAR2013/images'
    im_dir = '../../data/test_images'
    im_names = os.listdir(im_dir)
    print 'processing img'
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_dir, im_name)

