import argparse
import time

from config import Config
from data.loader import Loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print summary of saved model.')
    parser.add_argument('-m', '--model', dest='model', type=str, required=True)
    parser.add_argument('--line-length', dest='line_length', type=int, default=140)
    args = parser.parse_args()

    model = Loader.get_model(args.model)

    model.summary(line_length=args.line_length)

    if 'placing' in args.model:
        model.get_layer('grasp').summary(line_length=args.line_length)
        model.get_layer('place').summary(line_length=args.line_length)
        model.get_layer('merge').summary(line_length=args.line_length)

        print('Grasp Dropout', model.get_layer('grasp').get_layer('dropout_6').rate)
        print('Place Dropout', model.get_layer('place').get_layer('dropout_21').rate)
        print('Merge Dropout', model.get_layer('merge').get_layer('dropout_24').rate)

        print('Grasp Z L2', model.get_layer('grasp').get_layer('z_m').activity_regularizer.l2)
        print('Place Z L2', model.get_layer('place').get_layer('z_p').activity_regularizer.l2)
        print('Merge L2', model.get_layer('merge').get_layer('dense_1').kernel_regularizer.l2)
