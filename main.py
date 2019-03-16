import argparse
from os import path
from tqdm import tqdm

from parser.parser import PCFG

SCRIPT_PATH = path.dirname(path.abspath(__file__))
OUTPUT_FOLDER = path.join(SCRIPT_PATH, './data')

def main(args):
    pcfg = PCFG()
    pcfg.fill_pcfg_from_file(args.train_path)
    print('Trained PCFG')

    input_file = open(args.test_path, 'r')
    output_file = open(path.join(OUTPUT_FOLDER, 'out'), 'w')

    for line in tqdm(input_file):
        parsable, tree = pcfg.probabilitic_CKY(line.split())
        if parsable:
            output_file.write(' '.join(str(tree).split()))
        else:
            output_file.write('()')
            print('NOT PARSED : {}'.format(line))

    input_file.close()
    output_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_path',
        type=str,
        default=path.join(SCRIPT_PATH, './data/train.tag'),
        help='Path to the training dataset')
    parser.add_argument(
        '--test_path',
        type=str,
        default=path.join(SCRIPT_PATH, './data/train.no_tag'),
        help='Path to the test dataset')

    args = parser.parse_args()

    main(args)
