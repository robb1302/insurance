from src.main_function.main_funcion import create_model
import argparse

# TODO: correct help text for all arguments
# argparser
parser = argparse.ArgumentParser(
    description='', formatter_class=argparse.RawTextHelpFormatter)


parser.add_argument(
    '--country', help='name for output document default: test', default='test')

args = parser.parse_args()
if __name__ == '__main__':
    create_model(country=args.country)
    print('hello')