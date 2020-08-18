# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for the specific language governing permissions and limitations under the License.

import argparse
import logging
import time
import math
import random
import os
import mxnet as mx
import gluoncv as gcv

gcv.utils.check_version('0.6.0')

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs

from mxnet import autograd, gluon, lr_scheduler
from mxnet.gluon.data.vision import transforms

import herring.mxnet as herring

# Training settings
parser = argparse.ArgumentParser(description='MXNet ImageNet Example',
                                 formatter_class= \
                                     argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                    default=4, help='Number of data workers, you can use larger '
                    'number to accelerate data loading, if you CPU and GPUs '
                    'are powerful.')
parser.add_argument('--rec-train', type=str, default='',
                    help='the training data in recordio format')
parser.add_argument('--rec-val', type=str, default='',
                    help='the validation data in recordio format')
parser.add_argument('--batch-size', type=int, default=128,
                    help='training batch size per device (default: 128)')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training (default: float32)')
parser.add_argument('--num-epochs', type=int, default=90,
                    help='number of training epochs (default: 90)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate for a single GPU (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer (default: 0.9)')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate (default: 0.0001)')
parser.add_argument('--warmup-lr', type=float, default=0.0,
                    help='starting warmup learning rate (default: 0.0)')
parser.add_argument('--warmup-epochs', type=int, default=10,
                    help='number of warmup epochs (default: 10)')
parser.add_argument('--last-gamma', action='store_true', default=False,
                    help='whether to init gamma of the last BN layer in \
                    each bottleneck to 0 (default: False)')
parser.add_argument('--no-wd', action='store_true',
                    help='whether to remove weight decay on bias, and \
                    beta/gamma for batchnorm layers.')
parser.add_argument('--model', type=str, default='resnet50_v1',
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--use-pretrained', action='store_true', default=False,
                    help='load pretrained model weights (default: False)')
parser.add_argument('--eval-frequency', type=int, default=0,
                    help='frequency of evaluating validation accuracy \
                    when training with gluon mode (default: 0)')
parser.add_argument('--log-interval', type=int, default=40,
                    help='number of batches to wait before logging \
                          (default: 40)')
parser.add_argument('--save-frequency', type=int, default=20,
                    help='frequency of model saving (default: 0)')
parser.add_argument('--save-dir', type=str, default='params',
                    help='directory of saved models')
parser.add_argument('--static-alloc', action='store_true',
                    help='Whether to use static memory allocation. Memory usage will increase.')
# data
parser.add_argument('--input-size', type=int, default=224,
                    help='size of the input image size. default is 224')
parser.add_argument('--crop-ratio', type=float, default=0.875,
                    help='Crop ratio during validation. default is 0.875')
# resume
parser.add_argument('--resume-epoch', type=int, default=0,
                    help='epoch to resume training from.')
parser.add_argument('--resume-params', type=str, default='',
                    help='path of parameters to load from.')
parser.add_argument('--resume-states', type=str, default='',
                    help='path of trainer state to load from.')

args = parser.parse_args()

# Herring: initialize Herring
# TODO(lnyuan): add init API when available: herring.init()
num_gpus = herring.size()
rank = herring.rank()
local_rank = herring.local_rank()

if rank == 0:
    logging.basicConfig(level=logging.INFO)
    logging.info(args)

num_classes = 1000
num_training_samples = 1281167
batch_size = args.batch_size
epoch_size = int(math.ceil(int(num_training_samples // num_gpus) / batch_size))

lr_sched = lr_scheduler.CosineScheduler(
    args.num_epochs * epoch_size,
    base_lr=(args.lr * num_gpus),
    warmup_steps=(args.warmup_epochs * epoch_size),
    warmup_begin_lr=args.warmup_lr
)


class SplitSampler(mx.gluon.data.sampler.Sampler):
    """ Split the dataset into `num_parts` parts and sample from the part with
    index `part_index`

    Parameters
    ----------
    length: int
      Number of examples in the dataset
    num_parts: int
      Partition the data into multiple parts
    part_index: int
      The index of the part to read from
    """

    def __init__(self, length, num_parts=1, part_index=0, random=True):
        # Compute the length of each partition
        self.part_len = length // num_parts
        # Compute the start index for this partition
        self.start = self.part_len * part_index
        # Compute the end index for this partition
        self.end = self.start + self.part_len
        self.random = random

    def __iter__(self):
        # Extract examples between `start` and `end`, shuffle and return them.
        indices = list(range(self.start, self.end))
        if self.random:
            random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.part_len


def batch_fn(batch, ctx):
    data = batch[0].as_in_context(ctx)
    label = batch[1].as_in_context(ctx)
    return data, label


def get_dataloader():
    jitter_param = 0.4
    lighting_param = 0.1

    crop_ratio = args.crop_ratio if args.crop_ratio > 0 else 0.875
    resize = int(math.ceil(args.input_size / crop_ratio))

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param,
                                     contrast=jitter_param,
                                     saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        normalize
    ])

    transform_val = transforms.Compose([
        transforms.Resize(resize, keep_ratio=True),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        normalize
    ])

    train_set = mx.gluon.data.vision.ImageRecordDataset(args.rec_train) \
        .transform_first(transform_train)
    val_set = mx.gluon.data.vision.ImageRecordDataset(args.rec_val) \
        .transform_first(transform_val)

    train_sampler = SplitSampler(length=len(train_set), num_parts=num_gpus,
                                 part_index=rank)
    val_sampler = SplitSampler(length=len(val_set), num_parts=num_gpus,
                               part_index=rank)

    train_data = gluon.data.DataLoader(train_set, batch_size=batch_size,
                                       last_batch='discard',
                                       num_workers=args.num_workers,
                                       sampler=train_sampler)
    val_data = gluon.data.DataLoader(val_set, batch_size=batch_size,
                                     num_workers=args.num_workers,
                                     sampler=val_sampler)
    return train_data, val_data


# Herring: pin GPU to local rank
context = mx.gpu(local_rank)

train_data, val_data = get_dataloader()

herring.attach_dataloader([train_data, val_data])

# Get model from GluonCV model zoo
# https://gluon-cv.mxnet.io/model_zoo/index.html
kwargs = {'ctx': context,
          'pretrained': args.use_pretrained,
          'classes': num_classes}

net = get_model(args.model, **kwargs)
net.cast(args.dtype)

if rank == 0:
    logging.info(net)

# Create initializer
initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in",
                             magnitude=2)


def train_gluon():
    if args.save_dir:
        save_dir = args.save_dir
        save_dir = os.path.expanduser(save_dir)
        makedirs(save_dir)
    else:
        save_dir = './'
        save_frequency = 0

    def evaluate(epoch):
        acc_top1 = mx.metric.Accuracy()
        acc_top5 = mx.metric.TopKAccuracy(5)
        for _, batch in enumerate(val_data):
            data, label = batch_fn(batch, context)
            output = net(data.astype(args.dtype, copy=False))
            acc_top1.update([label], [output])
            acc_top5.update([label], [output])

        top1_name, top1_acc = acc_top1.get()
        top5_name, top5_acc = acc_top5.get()

        res1 = worker_comm.gather(top1_acc, root=0)
        res2 = worker_comm.gather(top5_acc, root=0)
        if rank == 0:
            top1_acc = sum(res1) / len(res1)
            top5_acc = sum(res2) / len(res2)
            logging.info('Epoch[%d] Rank[%d]\tValidation-%s=%f \
                         \tValidation-%s=%f',
                         epoch, rank, top1_name, top1_acc, top5_name, top5_acc)

    # Hybridize and initialize model
    if args.static_alloc:
        net.hybridize(static_alloc=True, static_shape=True)
    else:
        net.hybridize()

    if args.resume_params is not '':
        net.load_parameters(args.resume_params, ctx=context)
    else:
        net.initialize(initializer, ctx=context)

    if args.no_wd:
        for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0

    # Fetch parameters
    params = net.collect_params()

    # Create optimizer
    optimizer = 'nag'
    optimizer_params = {'wd': args.wd,
                        'momentum': args.momentum,
                        'lr_scheduler': lr_sched}
    if args.dtype == 'float16':
        optimizer_params['multi_precision'] = True
    opt = mx.optimizer.create(optimizer, **optimizer_params)

    # Herring: create DistributedTrainer, a subclass of gluon.Trainer
    trainer = herring.DistributedTrainer(params, opt)
    if args.resume_states is not '':
        trainer.load_states(args.resume_states)

    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=True)
    train_metric = mx.metric.Accuracy()

    # Get the MPI worker communicator for distributed validation
    worker_comm = herring.get_worker_comm()

    # Train model
    total_time = 0
    for epoch in range(args.resume_epoch, args.num_epochs):
        tic = time.time()
        train_metric.reset()

        btic = time.time()
        for nbatch, batch in enumerate(train_data, start=1):
            data, label = batch_fn(batch, context)
            data, label = [data], [label]

            with autograd.record():
                outputs = [net(X.astype(args.dtype, copy=False)) for X in data]
                loss = [loss_fn(yhat, y.astype(args.dtype, copy=False))
                        for yhat, y in zip(outputs, label)]

            for l in loss:
                l.backward()
            trainer.step(batch_size)
            train_metric.update(label, outputs)

            if args.log_interval and nbatch % args.log_interval == 0:
                if rank == 0:
                    logging.info('Epoch[%d] Batch[%d] Loss[%.3f]', epoch,
                                 nbatch, loss[0].mean().asnumpy()[0])

                    train_metric_name, train_metric_score = train_metric.get()
                    logging.info('Epoch[%d] Rank[%d] Batch[%d]\t%s=%f\tlr=%f',
                                 epoch, rank, nbatch, train_metric_name,
                                 train_metric_score, trainer.learning_rate)
                btic = time.time()

        # Report metrics
        elapsed = time.time() - tic
        total_time += elapsed
        _, acc = train_metric.get()
        if rank == 0:
            logging.info('Epoch[%d] Rank[%d] Batch[%d]\tTime cost=%.2f \
                          \tTrain-metric=%f', epoch, rank, nbatch, elapsed, acc)
            epoch_speed = num_gpus * batch_size * nbatch / elapsed
            logging.info('Epoch[%d]\tSpeed: %.2f samples/sec', epoch,
                         epoch_speed)

        # Evaluate performance
        if args.eval_frequency and (epoch + 1) % args.eval_frequency == 0:
            evaluate(epoch)

        # Save model
        if args.save_frequency and (epoch + 1) % args.save_frequency == 0:
            net.save_parameters('%s/imagenet-%s-%d.params' % (save_dir,
                                                              args.model, epoch))
            trainer.save_states('%s/imagenet-%s-%d.states' % (save_dir,
                                                              args.model, epoch))

    logging.info('Total Training Time: %d Seconds', total_time)

    # Evaluate performance at the end of training
    logging.info('Start Distributed Evaluation')
    evaluate(args.num_epochs - 1)

    net.save_parameters('%s/imagenet-%s-%d.params' % (save_dir, args.model,
                                                      args.num_epochs - 1))
    trainer.save_states('%s/imagenet-%s-%d.states' % (save_dir, args.model,
                                                      args.num_epochs - 1))


if __name__ == '__main__':
    train_gluon()
