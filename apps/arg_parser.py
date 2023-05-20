import argparse

def get_args():
  parser = argparse.ArgumentParser(description='WhatSubs')
  parser.add_argument('--debug', action='store_true', help='debug mode', default=False)
  parser.add_argument('--device', action='store', help='disables CUDA training', default='gpu')
  parser.add_argument('--train_ae', action='store_true', help='train auto encoder', default=False)
  parser.add_argument('--latent_dim', action='store', help='latent dimension', default=32, type=int)
  parser.add_argument('--batch_size', action='store', help='batch size', default=1024, type=int)
  parser.add_argument('--num_workers', action='store', help='number of workers', default=4, type=int)
  parser.add_argument('--val_rate', action='store', help='validation rate', default=0.2, type=float)
  parser.add_argument('--max_epoch', action='store', help='maximum epoch', default=100, type=int)
  
  return vars(parser.parse_args())