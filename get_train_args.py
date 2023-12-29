import argparse
def get_args():    
    parser = argparse.ArgumentParser(description ='Model training params')
    parser.add_argument('data_directory',
                    type = str,
                    default = False,
                    help ='training dataset')

    parser.add_argument('--save_dir',
                    type=str,
                   default = "/home/workspace/saved_models")

    parser.add_argument('--arch',
                    type=str,
                    choices = ["vgg19", "alexnet"],
                   default = 'vgg19')
    parser.add_argument('--learning_rate',
                    type=float,
                   default = 0.01)

    parser.add_argument('--hidden_units',
                    type=int,
                   default = 512)

    parser.add_argument('--epochs',
                    type=int,
                   default = 20)

    parser.add_argument('--print_every',
                    type=int,
                   default = 5)

    parser.add_argument('--gpu',
                    action = "store_true",
                   help = "Using gpu for training")

    args = parser.parse_args()
    return args
