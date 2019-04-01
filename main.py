import argparse
from os import path
from tqdm import tqdm

from parser.parser import PCFG

SCRIPT_PATH = path.dirname(path.abspath(__file__))
OUTPUT_FOLDER = path.join(SCRIPT_PATH, './data')


def parse_from_std(args):
    pcfg = PCFG()
    pcfg.fill_pcfg_from_file(args.train_path)
    print('Trained PCFG')

    for line in args.lines.split('\n'):
        parsable, tree = pcfg.probabilitic_CKY(line.split())
        if parsable:
            print(' '.join(str(tree).split()))
        else:
            print('NOT PARSED : {}'.format(line))


def parse_from_file(args):
    pcfg = PCFG()
    pcfg.fill_pcfg_from_file(args.train_path)
    print('Trained PCFG')

    input_file = open(args.test_path, 'r')

    if args.out_path is not None:
        output_file = open(path.join(OUTPUT_FOLDER, 'out'), 'w')
        for line in tqdm(input_file):
            parsable, tree = pcfg.probabilitic_CKY(line.split())
            if parsable:
                output_file.write(' '.join(str(tree).split()) + '\n')
            else:
                output_file.write('()\n')
                print('NOT PARSED : {}'.format(line))
        output_file.close()

    else:
        for line in input_file:
            parsable, tree = pcfg.probabilitic_CKY(line.split())
            if parsable:
                print(' '.join(str(tree).split()))
            else:
                print('NOT PARSED : {}'.format(line))

    input_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_path',
        type=str,
        default=path.join(SCRIPT_PATH, './data/train.tag'),
        help='Path to the training dataset')

    subparsers = parser.add_subparsers(help='sub-command help')

    parser_std = subparsers.add_parser('std', help='Parse from standard input')
    parser_std.add_argument('lines', type=str, help='Sentence to parse')

    parser_std.set_defaults(func=parse_from_std)

    # create the parser for the "b" command
    parser_file = subparsers.add_parser('file', help='Parse file')

    parser_file.add_argument(
        'test_path', type=str, help='Path to the test dataset')
    parser_file.add_argument(
        '--out_path',
        type=str,
        default=None,
        help='Path where the results are written.')

    parser_file.set_defaults(func=parse_from_file)

    args = parser.parse_args()
    args.func(args)
