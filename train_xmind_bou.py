import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from lib.networks import EMCADNetv3, EMCADNetv4, EMCADNetv5, EMCADNetv6, EMCADNetv7, EMCADNetv8, EMCADNetv9, EMCADNetv5_DW, EMCADNetv5_DW_Skip
# from networks.WaveFormerCompact import Model as ModelCompact

from trainer_xmind_bou import trainer_synapse
import warnings
warnings.filterwarnings('ignore')



parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/images/PublicDataset/Transunet_synaps/project_TransUNet/data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--test_path', type=str, default='/images/PublicDataset/Transunet_synaps/project_TransUNet/data/Synapse/test_vol_h5', help='root dir for data')
parser.add_argument('--dataset', type=str, default='Synapse', help='experiment_name')
parser.add_argument('--dstr_fast', type=bool, default=False, help='SynapseDatasetFast: will load all data into RAM')
parser.add_argument('--en_lnum', type=int, default=3, help='en_lnum: Laplacian layers (Pyramid) for the encoder')
parser.add_argument('--br_lnum', type=int, default=3, help='br_lnum: Laplacian layers (Pyramid) for the bridge')
parser.add_argument('--de_lnum', type=int, default=3, help='de_lnum: Laplacian layers (Pyramid) for the decoder')
parser.add_argument('--compact', type=bool, default=False, help='compact with 3 blocks insted of 4 blocks')
parser.add_argument('--continue_tr', type=bool, default=False, help='continue training from the last saved epoch')
parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer: [SGD, AdamW])')
parser.add_argument('--dice_loss_weight', type=float, default=0.6, help="You need to determine <x> (default=0.6): => [loss = (1-x)*ce_loss + x*dice_loss]")
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int, default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str,  default='./model_out',help='output dir')                   
parser.add_argument('--max_iterations', type=int, default=90000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=400, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=20, help='batch_size per gpu')
parser.add_argument('--num_workers', type=int, default=12, help='num_workers')
parser.add_argument('--eval_interval', type=int, default=20, help='eval_interval')
parser.add_argument('--model_name', type=str, default='mogav5', help='model_name')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--bridge_layers', type=int, default=1, help='number of bridge layers')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[1, 3, 5], help='multi-scale kernel sizes in MSDC block')
parser.add_argument('--scale_factors', type = str, default = "0.8,0.4", help = "Boundary enhancement downsample scale factors")
parser.add_argument('--expansion_factor', type=int, default=2, help='expansion factor in MSCB block')
parser.add_argument('--activation_mscb', type=str, default='relu6', help='activation used in MSCB: relu6 or relu')
parser.add_argument('--no_dw_parallel', action='store_true',  default=False, help='use this flag to disable depth-wise parallel convolutions')
parser.add_argument('--concatenation', action='store_true',  default=False, help='use this flag to concatenate feature maps in MSDC block')
parser.add_argument('--encoder', type=str, default='pvt_v2_b2', help='Name of encoder: pvt_v2_b2, pvt_v2_b0, resnet18, resnet34 ...')
parser.add_argument('--no_pretrain', action='store_true',  default=False, help='use this flag to turn off loading pretrained enocder weights')
parser.add_argument('--lgag_ks', type=int, default=3, help='Kernel size in LGAG')
parser.add_argument('--base_lr', type=float,  default=0.05, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--z_spacing', type=int, default=1, help='z_spacing')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--amp', action='store_true', help='AMP mode')
parser.add_argument('--fast_data', action='store_true', help='no AMP mode')

# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()

args.output_dir = args.output_dir + f'/{args.model_name}'
os.makedirs(args.output_dir, exist_ok=True)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset

    if args.batch_size != 24 and args.batch_size % 5 == 0:
        args.base_lr *= args.batch_size / 24

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.concatenation:
        aggregation = 'concat'
    else: 
        aggregation = 'add'
    
    if args.no_dw_parallel:
        dw_mode = 'series'
    else: 
        dw_mode = 'parallel'
    scale_factors = list(map(float, args.scale_factors.split(',')))

    if args.model_name =="mogav3":
        net = EMCADNetv3(
            num_classes=args.num_classes, 
            kernel_sizes=args.kernel_sizes, 
            expansion_factor=args.expansion_factor, 
            dw_parallel=not args.no_dw_parallel, 
            add=not args.concatenation, 
            lgag_ks=args.lgag_ks, 
            activation=args.activation_mscb, 
            encoder=args.encoder, 
            pretrain= not args.no_pretrain
        ).cuda(0)
    elif args.model_name =="mogav4":
        net = EMCADNetv4(
            num_classes=args.num_classes, 
            kernel_sizes=args.kernel_sizes, 
            expansion_factor=args.expansion_factor, 
            dw_parallel=not args.no_dw_parallel, 
            add=not args.concatenation, 
            lgag_ks=args.lgag_ks, 
            activation=args.activation_mscb, 
            encoder=args.encoder, 
            pretrain= not args.no_pretrain
        ).cuda(0)
    elif args.model_name =="mogav5":
        net = EMCADNetv5(
            num_classes=args.num_classes, 
            kernel_sizes=args.kernel_sizes, 
            expansion_factor=args.expansion_factor, 
            dw_parallel=not args.no_dw_parallel, 
            add=not args.concatenation, 
            lgag_ks=args.lgag_ks, 
            activation=args.activation_mscb, 
            encoder=args.encoder, 
            pretrain= not args.no_pretrain
        ).cuda(0)
    elif args.model_name =="mogav6":
        net = EMCADNetv6(
            num_classes=args.num_classes, 
            kernel_sizes=args.kernel_sizes, 
            expansion_factor=args.expansion_factor, 
            dw_parallel=not args.no_dw_parallel, 
            add=not args.concatenation, 
            lgag_ks=args.lgag_ks, 
            activation=args.activation_mscb, 
            encoder=args.encoder, 
            pretrain= not args.no_pretrain
        ).cuda(0)
    elif args.model_name =="mogav7":
        net = EMCADNetv7(
            num_classes=args.num_classes, 
            kernel_sizes=args.kernel_sizes, 
            expansion_factor=args.expansion_factor, 
            dw_parallel=not args.no_dw_parallel, 
            add=not args.concatenation, 
            lgag_ks=args.lgag_ks, 
            activation=args.activation_mscb, 
            encoder=args.encoder, 
            pretrain= not args.no_pretrain
        ).cuda(0)
    elif args.model_name =="mogav8":
        net = EMCADNetv8(
            num_classes=args.num_classes, 
            kernel_sizes=args.kernel_sizes, 
            expansion_factor=args.expansion_factor, 
            dw_parallel=not args.no_dw_parallel, 
            add=not args.concatenation, 
            lgag_ks=args.lgag_ks, 
            activation=args.activation_mscb, 
            encoder=args.encoder, 
            pretrain= not args.no_pretrain
        ).cuda(0)
    elif args.model_name =="mogav9":
        net = EMCADNetv9(
            num_classes=args.num_classes, 
            kernel_sizes=args.kernel_sizes, 
            expansion_factor=args.expansion_factor, 
            dw_parallel=not args.no_dw_parallel, 
            add=not args.concatenation, 
            lgag_ks=args.lgag_ks, 
            activation=args.activation_mscb, 
            encoder=args.encoder, 
            pretrain= not args.no_pretrain
        ).cuda(0)
    elif args.model_name =="mogav5dw":
        net = EMCADNetv5_DW(
            num_classes=args.num_classes, 
            kernel_sizes=args.kernel_sizes,
            scale_factors = scale_factors, 
            expansion_factor=args.expansion_factor, 
            dw_parallel=not args.no_dw_parallel, 
            add=not args.concatenation, 
            lgag_ks=args.lgag_ks, 
            activation=args.activation_mscb, 
            encoder=args.encoder, 
            pretrain= not args.no_pretrain
        ).cuda(0)
    elif args.model_name =="mogav5dwskip":
        net = EMCADNetv5_DW_Skip(
            num_classes=args.num_classes, 
            kernel_sizes=args.kernel_sizes,
            scale_factors = scale_factors, 
            expansion_factor=args.expansion_factor, 
            dw_parallel=not args.no_dw_parallel, 
            add=not args.concatenation, 
            lgag_ks=args.lgag_ks, 
            activation=args.activation_mscb, 
            encoder=args.encoder, 
            pretrain= not args.no_pretrain
        ).cuda(0)
        
    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, net, args.output_dir)