import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from options.train_options import TrainOptions
from train import do_train

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', type=str, help='Path to datasets')
parser.add_argument('--dataset', type=str, default='ade20k', help='which dataset to process')
parser.add_argument('--norm', type=str, default='norm', help='normalization mode of dist map')

if __name__ == '__main__':
  options = TrainOptions().parse()
  options.norm_mode = 'clade'
  do_train(options)

  # calculate_fid_given_paths
  
  # args = parser.parse_args()
  # make_dist_train_val_ade_datasets(args.path, args.norm)