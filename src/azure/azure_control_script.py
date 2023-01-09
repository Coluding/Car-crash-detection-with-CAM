import argparse
import os
import pickle

# output will be logged, separate output from previous log entries.
print('-'*100)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str,
                        dest='data_path',
                        default='data',
                        help='data folder mounting point')

    return parser.parse_args()


if __name__ == '__main__':

    # parse the parameters passed to the this script
    args = parse_args()

    # set data paths
    train_folder = os.path.join(args.data_path, 'train')
    val_folder = os.path.join(args.data_path, 'test')



    print(train_folder)

