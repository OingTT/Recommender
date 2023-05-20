import argparse

def get_args():
  parser = argparse.ArgumentParser(description='WhatSubs')
  parser.add_argument('--debug', action='store_true', help='debug mode', default=False)
  parser.add_argument('--log', action='store_true', help='log mode', default=False)
  parser.add_argument('--device', action='store', help='disables CUDA training', default='gpu')
  parser.add_argument('--train-ae', action='store_true', help='train auto encoder', default=False)
  parser.add_argument('--latent-dim', action='store', help='latent dimension', default=32, type=int)
  parser.add_argument('--batch-size', action='store', help='batch size', default=1024, type=int)
  parser.add_argument('--num-workers', action='store', help='number of workers', default=4, type=int)
  parser.add_argument('--val-rate', action='store', help='validation rate', default=0.2, type=float)
  parser.add_argument('--max-epoch', action='store', help='maximum epoch', default=100, type=int)
  
  return vars(parser.parse_args())