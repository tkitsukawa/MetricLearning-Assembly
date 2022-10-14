import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--save-model', action='store_true', default=True)

args = parser.parse_args()
