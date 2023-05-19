#!/usr/bin/python3
import math
import os
import sys
import io
import json
import yaml
import platform
import re
from enum import Enum
import time
from datetime import datetime
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboard import program
import torch.profiler
from torch.profiler import profile, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
import torchvision
from collections import namedtuple
from collections import OrderedDict
from typing import Callable, Optional
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import cv2

from pymlutil.torch_util import count_parameters, model_stats, model_weights
from pymlutil.jsonutil import ReadDict, WriteDict, str2bool
from pymlutil.s3 import s3store, Connect
from pymlutil.functions import Exponential, GaussianBasis
from pymlutil.metrics import DatasetResults
import pymlutil.version as pymlutil_version
from pymlutil.version import VersionString


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-d', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug', type=str2bool, default=False, help='Wait for debuggee attach')
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')
    parser.add_argument('-min', action='store_true', help='Minimum run with a few iterations to test execution')
    parser.add_argument('-minimum', type=str2bool, default=False, help='Minimum run with a few iterations to test execution')

    parser.add_argument('-credentails', type=str, default='creds.yaml', help='Credentials file.')
    parser.add_argument('-s3_name', type=str, default='store', help='S3 name in credentials')

    parser.add_argument('-resnet_len', type=int, choices=[18, 34, 50, 101, 152, 20, 32, 44, 56, 110], default=152, help='Run description')
    parser.add_argument('-useConv1', type=str2bool, default=True, help='If true, use initial convolution and max pool before ResNet blocks')

    parser.add_argument('-dataset', type=str, default='imagenet', choices=['cifar10', 'imagenet'], help='Dataset')
    parser.add_argument('-dataset_path', type=str, default='/data', help='Local dataset path')
    parser.add_argument('-obj_imagenet', type=str, default='data/imagenet', help='Local dataset path')
    parser.add_argument('-model', type=str, default='model')

    parser.add_argument('-batch_size', type=int, default=100, help='Training batch size') 

    parser.add_argument('-optimizer', type=str, default='adamw', choices=['sgd', 'rmsprop', 'adam', 'adamw'], help='Optimizer')
    parser.add_argument('-learning_rate', type=float, default=1e-4, help='Training learning rate')
    parser.add_argument('-learning_rate_decay', type=float, default=1e-4, help='Rate decay multiple')
    parser.add_argument('-rate_schedule', type=json.loads, default='[50, 100, 150, 200, 250, 300, 350, 400, 450, 500]', help='Training learning rate')
    #parser.add_argument('-rate_schedule', type=json.loads, default='[40, 60, 65]', help='Training learning rate')
    #parser.add_argument('-rate_schedule', type=json.loads, default='[10, 15, 17]', help='Training learning rate')
    
    parser.add_argument('-momentum', type=float, default=0.9, help='Learning Momentum')
    parser.add_argument('-weight_decay', type=float, default=0.0001)
    parser.add_argument('-epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('-start_epoch', type=int, default=0, help='Start epoch')

    parser.add_argument('-num_workers', type=int, default=0, help='Data loader workers')
    parser.add_argument('-model_type', type=str,  default='classification')
    parser.add_argument('-model_class', type=str,  default='ImgClassifyPrune')
    parser.add_argument('-model_src', type=str,  default="")
    parser.add_argument('-model_dest', type=str, default="ImgClassifyPrune_imagenet_20221115_094226_ipc001_test")
    parser.add_argument('-test_sparsity', type=int, default=10, help='test step multiple')
    parser.add_argument('-test_results', type=str, default='test_results.json')
    parser.add_argument('-cuda', type=bool, default=True)

    parser.add_argument('-height', type=int, default=224, help='Input image height')
    parser.add_argument('-width', type=int, default=224, help='Input image width')
    parser.add_argument('-channels', type=int, default=3, help='Input image color channels')
    parser.add_argument('-k_accuracy', type=float, default=1.0, help='Accuracy weighting factor')
    parser.add_argument('-k_structure', type=float, default=0.5, help='Structure minimization weighting factor')
    parser.add_argument('-k_prune_basis', type=float, default=1.0, help='prune base loss scaling')
    parser.add_argument('-k_prune_exp', type=float, default=50.0, help='prune basis exponential weighting factor')
    parser.add_argument('-k_prune_sigma', type=float, default=1.0, help='prune basis exponential weighting factor')
    parser.add_argument('-target_structure', type=float, default=0.00, help='Structure minimization weighting factor')
    parser.add_argument('-batch_norm', type=bool, default=True)
    parser.add_argument('-dropout', type=str2bool, default=False, help='Enable dropout')
    parser.add_argument('-dropout_rate', type=float, default=0.0, help='Dropout probability gain')
    parser.add_argument('-weight_gain', type=float, default=11.0, help='Convolution norm tanh weight gain')
    parser.add_argument('-sigmoid_scale', type=float, default=5.0, help='Sigmoid scale domain for convolution channels weights')
    parser.add_argument('-feature_threshold', type=float, default=0.5, help='tanh pruning threshold')
    parser.add_argument('-convMaskThreshold', type=float, default=0.1, help='convolution channel sigmoid level to prune convolution channels')

    parser.add_argument('-augment_rotation', type=float, default=0.0, help='Input augmentation rotation degrees')
    parser.add_argument('-augment_scale_min', type=float, default=1.0, help='Input augmentation scale')
    parser.add_argument('-augment_scale_max', type=float, default=1.0, help='Input augmentation scale')
    parser.add_argument('-augment_translate_x', type=float, default=0.0, help='Input augmentation translation')
    parser.add_argument('-augment_translate_y', type=float, default=0.0, help='Input augmentation translation')
    parser.add_argument('-augment_noise', type=float, default=0.0, help='Augment image noise')

    parser.add_argument('-ejector', type=FenceSitterEjectors, default=FenceSitterEjectors.prune_basis, choices=list(FenceSitterEjectors))
    parser.add_argument('-ejector_start', type=float, default=4, help='Ejector start epoch')
    parser.add_argument('-ejector_full', type=float, default=5, help='Ejector full epoch')
    parser.add_argument('-ejector_max', type=float, default=1.0, help='Ejector max value')
    parser.add_argument('-ejector_exp', type=float, default=3.0, help='Ejector exponent')
    parser.add_argument('-prune', type=str2bool, default=False)
    parser.add_argument('-train', type=str2bool, default=False)
    parser.add_argument('-test', type=str2bool, default=True)
    parser.add_argument('-search_structure', type=str2bool, default=False)
    parser.add_argument('-search_flops', type=str2bool, default=False)
    parser.add_argument('-profile', type=str2bool, default=False)
    parser.add_argument('-time_trial', type=str2bool, default=False)
    parser.add_argument('-onnx', type=str2bool, default=False)
    parser.add_argument('-write_vision_graph', type=str2bool, default=False)
    parser.add_argument('-job', action='store_true',help='Run as job')

    parser.add_argument('-test_name', type=str, default='default_test', help='Test name for test log' )
    parser.add_argument('-test_path', type=str, default='test/tests.yaml', help='S3 path to test log')
    parser.add_argument('-resultspath', type=str, default='results.yaml')
    parser.add_argument('-prevresultspath', type=str, default=None)
    parser.add_argument('-test_dir', type=str, default=None)
    parser.add_argument('-tensorboard_dir', type=str, default='/tb_logs/test/ImgClassifyPrune_cifar10_20221106_094226_ipc001', help='to launch the tensorboard server, in the console, enter: tensorboard --logdir ./tb --bind_all')
    #parser.add_argument('-tensorboard_dir', type=str, default=None, help='to launch the tensorboard server, in the console, enter: tensorboard --logdir ./tb --bind_all')
    parser.add_argument('-tb_dest', type=str, default='ImgClassifyPrune_cifar10_20221106_094226_ipc001')
    parser.add_argument('-config', type=str, default='config/build.yaml', help='Configuration file')
    parser.add_argument('-description', type=json.loads, default='{"description":"CRISP classification"}', help='Test description')
    parser.add_argument('-output_dir', type=str, default='./out', help='to launch the tensorboard server, in the console, enter: tensorboard --logdir ./tb --bind_all')


    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=4, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )


    args = parser.parse_args()

    if args.d:
        args.debug = args.d
    if args.min:
        args.minimum = args.min

    return args


class ConvBR(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 batch_norm=True, 
                 relu=True,
                 kernel_size=3, 
                 stride=1,
                 dilation=1, 
                 groups=1, 
                 bias=True, 
                 padding_mode='zeros',
                 weight_gain = 11.0,
                 convMaskThreshold=0.5,
                 dropout_rate=0.2,
                 residual = False,
                 dropout=False,
                 conv_transpose=False, # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
                 device=torch.device("cpu"),
                 max_pool = False,
                 pool_kernel_size=3,
                 pool_stride=2,
                 pool_padding=1,
                 ):
        super(ConvBR, self).__init__()
        self.in_channels = in_channels
        if out_channels < 1:
            raise ValueError("out_channels must be > 0")
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.relu = relu
        self.weight_gain = abs(weight_gain)
        self.convMaskThreshold = convMaskThreshold
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.dropout_rate = dropout_rate
        self.residual = residual
        self.use_dropout = dropout
        self.conv_transpose = conv_transpose
        self.device = device
        self.input_shape = None
        self.output_shape = None
        self.max_pool = max_pool
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

        if type(kernel_size) == int:
            padding = kernel_size // 2 # dynamic add padding based on the kernel_size
        else:
            padding = kernel_size//2

        if not self.conv_transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        else:
            self.conv = nn.ConvTranspose2d( in_channels=in_channels, 
                                            out_channels=out_channels, 
                                            kernel_size=kernel_size, 
                                            stride=stride, 
                                            groups=groups, 
                                            bias=bias, 
                                            dilation=dilation, 
                                            padding_mode=padding_mode)

        if self.batch_norm:
            self.batchnorm2d = nn.BatchNorm2d(out_channels)

        self.total_trainable_weights = model_weights(self)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = None

        if self.relu:
            self.activation = nn.ReLU()
        else:
            self.activation = None

        if self.max_pool:
                self.pool = nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride, padding=self.pool_padding)
        else:
            self.pool = None

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif self.batch_norm and self.batchnorm2d and isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        if self.input_shape is None:
            self.input_shape = x.shape

        if self.out_channels > 0:
            if self.use_dropout:
                x = self.dropout(x)
                
            x = self.conv(x)
  
            if self.batch_norm:
                x = self.batchnorm2d(x)

            if self.relaxation:
                x = self.relaxation(x)  

            if self.activation:
                x = self.activation(x)

            if hasattr(self, 'pool') and self.pool is not None:
                x = self.pool(x)

        else :
            print("Failed to prune zero size convolution")

        if self.output_shape is None:
            self.output_shape = x.shape

        return x


DefaultMaxDepth = 1
class Cell(nn.Module):
    def __init__(self,
                 in1_channels, 
                 in2_channels = 0,
                 batch_norm=False, 
                 relu=True,
                 kernel_size=3, 
                 padding=0, 
                 dilation=1, 
                 groups=1,
                 bias=True, 
                 padding_mode='zeros',
                 residual=True,
                 dropout=False,
                 device=torch.device("cpu"),
                 feature_threshold=0.5,
                 cell_convolution=DefaultMaxDepth,
                 weight_gain = 11.0,
                 convMaskThreshold=0.5,
                 convolutions=[{'out_channels':64, 'kernel_size': 3, 'stride': 1, 'dilation': 1, 'conv_transpose':False}],
                 dropout_rate = 0.2,
                 ):
                
        super(Cell, self).__init__()

        self.in1_channels = in1_channels
        self.in2_channels = in2_channels
        self.prev_relaxation = prev_relaxation
        self.relaxation = relaxation
        self.batch_norm = batch_norm
        self.relu = relu
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.residual = residual
        self.dropout = dropout
        self.device = device
        self.feature_threshold = feature_threshold
        self.cell_convolution = nn.Parameter(torch.tensor(cell_convolution, dtype=torch.float, device=self.device))
        self.weight_gain = weight_gain
        self.convMaskThreshold = convMaskThreshold
        self.convolutions = deepcopy(convolutions)
        self.dropout_rate = dropout_rate
        self.prevent_collapse = prevent_collapse

        self.cnn = torch.nn.ModuleList()

        # First convolution uses in1_channels+in2_channels is input chanels. 
        # Remaining convolutions uses out_channels as chanels

        src_channels = in_chanels = self.in1_channels+self.in2_channels

        totalStride = 1
        totalDilation = 1
        self.activation = None

        if self.residual and (in_chanels != self.convolutions[-1]['out_channels'] or totalStride != 1 or totalDilation != 1):
            for i, convdev in enumerate(convolutions):
                totalStride *= convdev['stride']
                totalDilation *= convdev['dilation']

            self.conv_residual = ConvBR(in_chanels, self.convolutions[-1]['out_channels'],
                prev_relaxation = prev_relaxation,
                relaxation = relaxation,
                batch_norm=self.batch_norm, 
                relu=False, 
                kernel_size=1, 
                stride=totalStride, 
                dilation=totalDilation, 
                groups=self.groups, 
                bias=self.bias, 
                padding_mode=self.padding_mode,
                weight_gain=self.weight_gain,
                convMaskThreshold=self.convMaskThreshold, 
                dropout_rate=self.dropout_rate,
                search_structure=self.search_structure,
                residual=True,
                dropout=self.dropout,
                k_prune_sigma=self.k_prune_sigma,
                device=self.device,
                prevent_collapse = True)

            self.residual_relaxation = self.conv_residual.relaxation

        else:
            self.conv_residual = None

        if self.residual and self.relu:
            self.activation = nn.ReLU()

        for i, convdev in enumerate(convolutions):
            conv_transpose = False
            if 'conv_transpose' in convdev and convdev['conv_transpose']:
                conv_transpose = True

            relu = self.relu
            if 'relu' in convdev:
                relu = convdev['relu']

            batch_norm = self.batch_norm
            if 'batch_norm' in convdev:
                batch_norm = convdev['batch_norm']

            max_pool = False
            if 'max_pool' in convdev:
                max_pool = convdev['max_pool']

            pool_kernel_size = 3
            if 'pool_kernel_size' in convdev:
                pool_kernel_size = convdev['pool_kernel_size']

            pool_stride = 2
            if 'pool_stride' in convdev:
                pool_stride = convdev['pool_stride']

            pool_padding = 1
            if 'pool_padding' in convdev:
                pool_padding = convdev['pool_padding']

            relu = relu
            if 'relu' in convdev:
                relu = convdev['relu']

            conv = ConvBR(src_channels, convdev['out_channels'], 
                batch_norm=batch_norm, 
                relu=relu, 
                kernel_size=convdev['kernel_size'], 
                stride=convdev['stride'], 
                dilation=convdev['dilation'],
                groups=self.groups, 
                bias=self.bias, 
                padding_mode=self.padding_mode,
                weight_gain=self.weight_gain,
                convMaskThreshold=self.convMaskThreshold, 
                dropout_rate=self.dropout_rate,
                residual=False, 
                dropout=self.dropout,
                conv_transpose=conv_transpose,
                k_prune_sigma=self.k_prune_sigma,
                device=self.device,
                search_flops = self.search_flops,
                max_pool = max_pool,
                pool_kernel_size = pool_kernel_size,
                pool_padding = pool_padding,
                )

            prev_relaxation = [conv.relaxation]
            relaxation = None
            
            self.cnn.append(conv)

            src_channels = convdev['out_channels']

        self._initialize_weights()
        self.total_trainable_weights = model_weights(self)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, in1, in2 = None, isTraining=False):
        if in1 is not None and in2 is not None:
            u = torch.cat((in1, in2), dim=1)
        elif in1 is not None:
            u = in1
        elif in2 is not None:
            u = in2
        else:
            return None

        # Resizing convolution
        if self.residual:
            if self.conv_residual and u is not None:
                residual = self.conv_residual(u)
            else:
                residual = u
        else:
            residual = None

        if self.cnn is not None:
            x = u
            for i, l in enumerate(self.cnn):
                x = self.cnn[i](x)

            if self.residual:
                y = x + residual

            else:
                y = x
        else:
            y = residual

        if self.activation:
            y = self.activation(y)

        return y

class FC(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 device=torch.device("cpu"),
                 ):
        super(FC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device


        self.fc = nn.Linear(in_channels, self.out_channels)
        self.total_trainable_weights = model_weights(self)

    def forward(self, x):
        y = self.fc(x)
        return y

# Resenet definitions from https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
class Resnet(Enum):
    layers_18 = 18
    layers_34 = 34
    layers_50 = 50
    layers_101 = 101
    layers_152 = 152
    cifar_20 = 20
    cifar_32 = 32
    cifar_44 = 44
    cifar_56 = 56
    cifar_110 = 110


def ResnetCells(size = Resnet.layers_50, useConv1 = True, conv1_kernel_size = 7, conv1_stride = 2):
    resnetCells = []
    
    sizes = {
        'layers_18': [2, 2, 2, 2], 
        'layers_34': [3, 4, 6, 3], 
        'layers_50': [3, 4, 6, 3], 
        'layers_101': [3, 4, 23, 3], 
        'layers_152': [3, 8, 36, 3],
        'cifar_20' : [7,6,6],
        'cifar_32' : [11,10,10],
        'cifar_44' : [15,14,14],
        'cifar_56' : [19,18,18],
        'cifar_110' : [37,36,36],
        }
    bottlenecks = {
        'layers_18': False, 
        'layers_34': False, 
        'layers_50': True, 
        'layers_101': True, 
        'layers_152': True,
        'cifar_20': False, 
        'cifar_32': False, 
        'cifar_44': False,
        'cifar_56': False, 
        'cifar_110': False
        }

    cifar_style = {
        'layers_18': False, 
        'layers_34': False, 
        'layers_50': False, 
        'layers_101': False, 
        'layers_152': False,
        'cifar_20': True, 
        'cifar_32': True, 
        'cifar_44': True,
        'cifar_56': True, 
        'cifar_110': True
        }

    resnet_cells = []
    block_sizes = sizes[size.name]
    bottleneck = bottlenecks[size.name]
    is_cifar_style = cifar_style[size.name]

    cell = []
    if is_cifar_style:
        network_channels = [32, 16, 8]
        cell.append({'out_channels':network_channels[0], 'kernel_size': 3, 'stride': 1, 'dilation': 1, 'search_structure':True})
        resnetCells.append({'residual':False, 'cell':cell})
    else:
        network_channels = [64, 128, 256, 512]
        if useConv1:
            cell.append({'out_channels':network_channels[0], 'kernel_size': conv1_kernel_size, 'stride': conv1_stride, 'dilation': 1, 'search_structure':True, 'residual':False, 'max_pool': True})
            resnetCells.append({'residual':False, 'cell':cell})


    for i, layer_size in enumerate(block_sizes):
        block_channels = network_channels[i]
        for j in range(layer_size):
            stride = 1
            # Downsample by setting stride to 2 on the first layer of each block
            if i != 0 and j == 0:
                stride = 2
            cell = []
            if bottleneck:
                # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
                # while original implementation places the stride at the first 1x1 convolution(self.conv1)
                # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
                # This variant is also known as ResNet V1.5 and improves accuracy according to
                # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
                cell.append({'out_channels':network_channels[i], 'kernel_size': 1, 'stride': 1, 'dilation': 1, 'search_structure':True, 'relu': True})
                cell.append({'out_channels':network_channels[i], 'kernel_size': 3, 'stride': stride, 'dilation': 1, 'search_structure':True, 'relu': True})
                cell.append({'out_channels':4*network_channels[i], 'kernel_size': 1, 'stride': 1, 'dilation': 1, 'search_structure':True, 'relu': False})
            else:
                cell.append({'out_channels':network_channels[i], 'kernel_size': 3, 'stride': stride, 'dilation': 1, 'search_structure':True, 'relu': True})
                cell.append({'out_channels':network_channels[i], 'kernel_size': 3, 'stride': 1, 'dilation': 1, 'search_structure':True, 'relu': False})
            resnetCells.append({'residual':True, 'cell':cell})
        
    return resnetCells

class Classify(nn.Module):
    def __init__(self, 
                convolutions, 
                device=torch.device("cpu"), 
                source_channels = 3, 
                out_channels = 10, 
                initial_channels=16, 
                batch_norm=True, 
                weight_gain=11, 
                convMaskThreshold=0.5, 
                dropout_rate=0.2, 
                search_structure = True, 
                sigmoid_scale=5.0, 
                feature_threshold=0.5, 
                search_flops = True,):
        super().__init__()
        self.device = device
        self.source_channels = source_channels
        self.out_channels = out_channels
        self.initial_channels = initial_channels
        self.weight_gain = weight_gain
        self.convMaskThreshold = convMaskThreshold
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.search_structure = search_structure
        self.sigmoid_scale = sigmoid_scale
        self.feature_threshold = feature_threshold
        self.search_flops = search_flops
                
        self.cells = torch.nn.ModuleList()
        in_channels = self.source_channels

        prev_relaxation = None
        residual_relaxation = None
        for i, cell_convolutions in enumerate(convolutions):

            convdfn = None

            cell = Cell(in1_channels=in_channels, 
                batch_norm=self.batch_norm,
                device=self.device,  
                weight_gain = self.weight_gain,
                convMaskThreshold = self.convMaskThreshold,
                residual=cell_convolutions['residual'],
                convolutions=cell_convolutions['cell'],  
                dropout_rate=self.dropout_rate, 
                search_structure=self.search_structure, 
                sigmoid_scale=self.sigmoid_scale, 
                feature_threshold=self.feature_threshold,
                search_flops = self.search_flops,
                prev_relaxation = [prev_relaxation],
                residual_relaxation = residual_relaxation,
                bias = False
                )

            if cell.conv_residual is not None:
                prev_relaxation = cell.conv_residual.relaxation
                residual_relaxation = cell.conv_residual.relaxation
            in_channels = cell_convolutions['cell'][-1]['out_channels']
            self.cells.append(cell)

        #self.maxpool = nn.MaxPool2d(2, 2) # Match Pytorch pretrained Resnet
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = FC(in_channels, self.out_channels, device=self.device)

        self.total_trainable_weights = model_weights(self)

        self.fc_weights = model_weights(self.fc)

    def forward(self, x, isTraining=False):
        #x = self.resnet_conv1(x)
        for i, cell in enumerate(self.cells):
            x = cell(x, isTraining=isTraining)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch dimension
        x = self.fc(x)
        return x

    def Cells(self):
        return self.cells


def InitWeights(model_state_dict, tv_state_dict, useConv1 = True):
    mkeys = list(model_state_dict.keys())
    tvkeys = tv_state_dict.keys()

    if useConv1:
        iCell = 0
    else:
        iCell = -1
    imkeys = 0

    iConv = 0
    pBlock = -1
    pResidual = -1
    pLayer = -1
    for i, tvkey in enumerate(tvkeys):
        tvkeyname = re.split('(\d*)\.', tvkey)
        tvkeyname = list(filter(None, tvkeyname))
        mkey = None

        if len(tvkeyname) == 6:
            layer = tvkeyname[1]
            residual = tvkeyname[2]
            block = int(tvkeyname[4])

            if block != pBlock:
                iConv += 1
            
            if residual != pResidual:
                iCell += 1
                iConv = 0

            if layer != pLayer:
                print('layer {}'.format(layer))

            pBlock = block
            pResidual = residual
            pLayer = layer

            blockname = 'cnn'
            if tvkeyname[3] == 'downsample':
                blockname = 'conv_residual'
                if tvkeyname[4] == '0':
                    mkey_operator = 'conv'
                else:
                    mkey_operator = 'batchnorm2d'

                mkey = 'cells.{}.{}.{}.{}'.format(iCell, blockname, mkey_operator,tvkeyname[-1]) 
            else:
                if tvkeyname[3] == 'bn':
                    mkey_operator = 'batchnorm2d'
                else:
                    mkey_operator = tvkeyname[3]

                mkey = 'cells.{}.{}.{}.{}.{}'.format(iCell, blockname, iConv,mkey_operator,tvkeyname[-1]) 

        elif tvkeyname[0] == 'conv' and tvkeyname[1]=='1':
            if useConv1:
                mkey = 'cells.{}.cnn.{}.{}.{}'.format(iCell, iConv,tvkeyname[0],tvkeyname[-1])

        elif tvkeyname[0] == 'bn' and tvkeyname[1]=='1':
            if useConv1:
                mkey = 'cells.{}.cnn.{}.batchnorm2d.{}'.format(iCell, iConv, tvkeyname[-1]) 

        elif tvkeyname[0] == 'fc':
            mkey = 'fc.{}.{}'.format(tvkeyname[0], tvkeyname[1])

        else:
            print('{}: {} skipping'.format(i, tvkey))

        if mkey is not None:
            if mkey in model_state_dict:
                if model_state_dict[mkey].shape == tv_state_dict[tvkey].shape:
                    print('{}: {} = {}'.format(i, mkey, tvkey))
                    model_state_dict[mkey] = tv_state_dict[tvkey].data.clone()
                else:
                    print('{}: {}={} != {}={} not in model'.format(i, mkey, model_state_dict[mkey].shape, tvkey, tv_state_dict[tvkey].shape))
            else:
                print('{}: {} == {} not in model'.format(i, mkey, tvkey))
        
    return model_state_dict


def load(s3, s3def, args, loaders, results):

    model = MakeClassifier(args, source_channels = loaders[0]['in_channels'], out_channels = loaders[0]['num_classes'])
    model_weights = model.state_dict()

    tv_weights = None
    transforms_vision = None

    if args.resnet_len == 18:
        tv_weights = torchvision.models.ResNet18_Weights(torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        model_vision = torchvision.models.resnet18(weights=tv_weights)

    elif args.resnet_len == 34:
        tv_weights = torchvision.models.ResNet34_Weights(torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        model_vision = torchvision.models.resnet34(weights=tv_weights)

    elif args.resnet_len == 50:
        tv_weights = torchvision.models.ResNet50_Weights(torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        model_vision = torchvision.models.resnet50(weights=tv_weights)

    elif args.resnet_len == 101:
        tv_weights = torchvision.models.ResNet101_Weights(torchvision.models.ResNet101_Weights.IMAGENET1K_V2)
        model_vision = torchvision.models.resnet101(weights=tv_weights)

    elif args.resnet_len == 152:
        tv_weights = torchvision.models.ResNet152_Weights(torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
        model_vision = torchvision.models.resnet152(weights=tv_weights)

    if tv_weights is not None:
        state_dict = tv_weights.get_state_dict(progress=True)
        model_weights = InitWeights(model_weights, state_dict, useConv1 = args.useConv1)
        model.load_state_dict(state_dict = model_weights, strict = False)
        transforms_vision = tv_weights.transforms()

    device = torch.device("cpu")
    if args.cuda:
        device = torch.device("cuda")
    model.to(device)
    model_vision.to(device)

    results['initial_parameters'] , results['initial_flops'] = ModelSize(args, model, loaders)
    print('load initial_parameters = {} initial_flops = {}'.format(results['initial_parameters'], results['initial_flops']))


    if(args.model_src and args.model_src != ''):
        modelObj = s3.GetObject(s3def['sets']['model']['bucket'], '{}/{}/{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_src ))

        if modelObj is not None:
            model = torch.load(io.BytesIO(modelObj))
            
            model_parameters, model_flops = ModelSize(args, model, loaders)
            results['load'][args.model_dest]= {'model_parameters':model_parameters, 'model_flops':model_flops}
            print('load model_parameters = {} model_flops = {}'.format(model_parameters, model_flops))
        else:
            print('Failed to load model_src {}/{}/{}/{}.pt  Exiting'.format(
                s3def['sets']['model']['bucket'],
                s3def['sets']['model']['prefix'],
                args.model_class,args.model_src))
            return model

    return model, results, model_vision, transforms_vision

def MakeClassifier(args, source_channels = 3, out_channels = 10):
    resnetCells = ResnetCells(Resnet(args.resnet_len), useConv1 = args.useConv1)

    device = torch.device("cpu")
    if args.cuda:
        device = torch.device("cuda")
    network = Classify(convolutions=resnetCells, 
                        device=device, 
                        weight_gain=args.weight_gain, 
                        dropout_rate=args.dropout_rate, 
                        search_structure=args.search_structure, 
                        sigmoid_scale=args.sigmoid_scale,
                        batch_norm = args.batch_norm,
                        feature_threshold = args.feature_threshold, 
                        source_channels = source_channels, 
                        out_channels = out_channels, 
                        )

    return network

# Classifier based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def main(args):

    if args.dataset == 'cifar10':
        loaders = CreateCifar10Loaders(args.dataset_path, batch_size = args.batch_size,  
                                       num_workers=args.num_workers, 
                                       cuda = args.cuda, 
                                       rotate=args.augment_rotation, 
                                       scale_min=args.augment_scale_min, 
                                       scale_max=args.augment_scale_max, 
                                       offset=args.augment_translate_x,
                                       augment_noise=args.augment_noise,
                                       width=args.width, height=args.height)

    elif args.dataset == 'imagenet':
        loaders = CreateImagenetLoaders(s3, s3def, 
                                        args.obj_imagenet, 
                                        args.dataset_path+'/imagenet', 
                                        crop_width=args.width, 
                                        crop_height=args.height, 
                                        batch_size=args.batch_size, 
                                        num_workers=args.num_workers,
                                        cuda = args.cuda,
                                        rotate=args.augment_rotation, 
                                        scale_min=args.augment_scale_min, 
                                        scale_max=args.augment_scale_max, 
                                        offset=args.augment_translate_x,
                                        augment_noise=args.augment_noise,
                                        augment=False,
                                        normalize=True
                                       )
    else:
        raise ValueError("Unupported dataset {}".format(args.dataset))

    model, transforms, results = MakeClassifier(args, source_channels = loaders['in_channels'], out_channels = loaders['num_classes'])

    model.zero_grad(set_to_none=True)
    torch.save(model, outname)
    return 0



if __name__ == '__main__':
    args = parse_arguments()

    if args.debug:
        print("Wait for debugger attach on {}:{}".format(args.debug_address, args.debug_port))
        import debugpy

        debugpy.listen(address=(args.debug_address, args.debug_port)) # Pause the program until a remote debugger is attached
        debugpy.wait_for_client() # Pause the program until a remote debugger is attached
        print("Debugger attached")

    result = main(args)
    sys.exit(result)

