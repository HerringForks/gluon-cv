python train_mask_rcnn_hvd.py --train-datapath /opt/ml/input/data/train/data/mxnet/mscoco/ \
--val-datapath /opt/ml/input/data/train/data/mxnet/mscoco/ \
--num-workers 4 \
--horovod \
--amp \
--lr-decay-epoch 8,10 \
--epochs 4 \
--log-interval 100 \
--val-interval 1 \
--batch-size 8 \
--use-fpn \
--lr 0.02 \
--lr-warmup-factor 0.001 \
--lr-warmup 1600 \
--static-alloc \
--clip-gradient 1.5 \
--use-ext \
--seed 987
