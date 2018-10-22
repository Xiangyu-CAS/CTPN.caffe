# Detecting Text in Natural Image with Connectionist Text Proposal Network

This repository is a caffe implementation of CTPN, **including training code**  
## Installation
1. requirement: python2.7
## Train
```
./train_ctpn.sh
```
Train loss
```
I1016 08:24:03.314466  7483 solver.cpp:228] Iteration 69980, loss = 0.0946643
I1016 08:24:03.314492  7483 solver.cpp:244]     Train net output #0: rpn_cls_loss = 0.0137189 (* 1 = 0.0137189 loss)
I1016 08:24:03.314512  7483 solver.cpp:244]     Train net output #1: rpn_loss_bbox = 0.0463248 (* 1 = 0.0463248 loss)
I1016 08:24:03.314517  7483 sgd_solver.cpp:106] Iteration 69980, lr = 1e-06
```

## Test
```
cd ./tools
python do_test.py
```

## TODO
- [x] training code
- [x] test code
- [ ] evaluate on icdar2013 dataset
- [ ] quantization INT8
- [ ] low bit
- [ ] rewrite convolution for bit computation
