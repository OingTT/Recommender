import argparse

def get_args():
  parser = argparse.ArgumentParser(description='WhatSubs')
  parser.add_argument('--debug', action='store_true', help='debug mode', default=False)
  parser.add_argument('--log', action='store_true', help='log mode', default=False)
  parser.add_argument('--device', action='store', help='disables CUDA training', default='gpu')
  parser.add_argument('--refresh', action='store_true', help='---', default=False)
  parser.add_argument('--train-ae', action='store_true', help='train auto encoder', default=False)
  parser.add_argument('--latent-dim', action='store', help='latent dimension', default=8, type=int)
  parser.add_argument('--batch-size', action='store', help='batch size', default=1024, type=int)
  parser.add_argument('--num-workers', action='store', help='number of workers', default=8, type=int)
  parser.add_argument('--val-rate', action='store', help='validation rate', default=0.3, type=float)
  parser.add_argument('--max-epoch', action='store', help='maximum epoch', default=100, type=int)
  parser.add_argument('--sample-rate', action='store', help='movie lens sampling rate', default=0, type=int)
  parser.add_argument('--log-dir', action='store', help='Directory path save log', default='./train_log', type=str)
  parser.add_argument('--movie-lens-dir', action='store', help='Directory path contain movie lens data', default='./ml-1m', type=str)
  parser.add_argument('--pretrained-model-dir', action='store', help='Directory path save pretrained models', default='./pretrained_model', type=str)

  return vars(parser.parse_args())