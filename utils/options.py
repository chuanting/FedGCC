# _*_ coding: utf-8 _*_
import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Federated learning with compression for wireless traffic prediction')
    # File parameters
    parser.add_argument('-file', type=str, default='milano',
                        help='file path and name')
    parser.add_argument('-model', type=str, default='mlp',
                        help='model choice')
    parser.add_argument('-comp', type=str, default='topk',
                        help='compressor choice')
    parser.add_argument('-compressed', dest='compressed', action='store_true')
    parser.add_argument('-no-compressed', dest='compressed', action='store_false')
    parser.set_defaults(compressed=True)

    parser.add_argument('-alg', type=str, default='fedgcc', help='federated algorithm')
    parser.add_argument('-mu', type=float, default=0.1, help='coefficient for FedProx')
    parser.add_argument('-thv', type=float, default=0.5, help='threshold value')
    parser.add_argument('-tkv', type=int, default=4, help='top k selection')
    parser.add_argument('-strategy', type=str, default='kb', help='aggregation strategy')
    parser.add_argument('-ratio', type=float, default=0.1, help='compression ratio')

    # Sliding window parameters
    parser.add_argument('-close_size', type=int, default=6,
                        help='how many time slots before target are used to model closeness')
    parser.add_argument('-test_days', type=int, default=7,
                        help='how many days data are used to test model performance')
    parser.add_argument('-granularity', type=int, default=6, help='time granularity (how many slots per hour)')
    parser.add_argument('-val_days', type=int, default=0,
                        help='how many days data are used to valid model performance')

    # Federated learning parameters
    # parser.add_argument('-bs', type=int, default=88, help='number of base stations')
    parser.add_argument('-frac', type=float, default=0.1,
                        help='fraction of clients: C')
    parser.add_argument('-local_epoch', type=int, default=5,
                        help='the number of local epochs: E')
    parser.add_argument('-local_bs', type=int, default=20,
                        help='local batch size: B')
    parser.add_argument('-epsilon', type=float, default=1.0, help='stepsize')

    parser.add_argument('-hidden_dim', type=int, default=128,
                        help='hidden neurons of MLP layer')
    parser.add_argument('-num_layers', type=int, default=2,
                        help='number of layers of MLP')
    parser.add_argument('-out_dim', type=int, default=1,
                        help='how many time slots we would like to predict for the future')
    parser.add_argument('-lr', type=float, default=0.1,
                        help='learning rate of NN')
    parser.add_argument('-momentum', type=float, default=0.5, help='momentum')
    parser.add_argument('-clip', type=float, default=2, help='gradient clip')
    parser.add_argument('-gpu', action='store_true', default=True,
                        help='Use CUDA for training')

    # Centralized and Isolated Neural Network parameters
    parser.add_argument('-batch_size', type=int, default=100,
                        help='batch size of centralized training')
    parser.add_argument('-epochs', type=int, default=10,
                        help='epochs of centralized training')

    parser.add_argument('-directory', type=str, default='./results/',
                        help='directory to store result')
    parser.add_argument('-seed', type=int, default=2, help='random seeds')
    parser.add_argument('-exp', type=int, default=1, help='repeat times')

    args = parser.parse_args()
    return args
