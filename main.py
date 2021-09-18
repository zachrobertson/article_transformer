import tensorflow as tf

from argparse import ArgumentParser

from utils import preprocessing

parser = ArgumentParser()
parser.add_argument(
    '--data',
    type=str,
    default='test.csv',
    help='Path to the .csv file from the /data directory'
)

if __name__ == "__main__":
    args = parser.parse_args()
    data = preprocessing.Preprocessing(args.data)
    print(data.tokenized_data)