import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type=int, default=1, help="1: use gpu 0, -1: use cpu.")
    parser.add_argument('--dataset_name', type=str, default='mnist', help="name of dataset: mnist 、...")
    parser.add_argument('--dataset_path', type=str, default='D:\\AI_datasets\\torchvision_datasets',
                        help="the path of dataset")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--epochs', type=int, default=20, help="rounds of training")
    parser.add_argument('--gamma', type=float, default=0.1, help="Percentage of authorized neurons (default: 0.1)")
    parser.add_argument('--lam', type=float, default=0, help="Locking value (default: 0)")
    parser.add_argument('--mask_size', type=float, default=0.5, help="Mask something data as clean data, others as noise (default: 0.5)")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")

    return parser.parse_known_args()[0]