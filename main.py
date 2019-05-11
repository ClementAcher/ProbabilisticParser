from os import path

from parser.parser import PCFG

import click
from tqdm import tqdm

SCRIPT_PATH = path.dirname(path.abspath(__file__))


@click.group()
def main():
    pass


@main.command()
@click.argument('lines_to_parse', type=str)
@click.option(
    '--training_set_path',
    type=click.Path(),
    default=path.join(SCRIPT_PATH, './data/train.tag'),
    help='Path to the training dataset.')
@click.option('--pprint', is_flag=True, help='Pretty print the result.')
def parse_from_std(lines_to_parse, training_set_path, pprint):
    pcfg = PCFG()
    pcfg.fill_pcfg_from_file(training_set_path)
    print('Trained PCFG')

    parsed_tree_generator = generate_parsed_tree(pcfg,
                                                 lines_to_parse.split('\n'))
    for parsable, tree in parsed_tree_generator:
        if parsable:
            if pprint:
                tree.pretty_print()
            else:
                print(get_tree_sequoia_format(tree))
        else:
            print('NOT PARSED : {}'.format(tree))


@main.command()
@click.argument('input_file', type=click.File())
@click.option(
    '--training_set_path',
    type=click.Path(),
    default=path.join(SCRIPT_PATH, './data/train.tag'),
    help='Path to the training dataset.')
@click.option(
    '--output_file',
    type=click.File(mode='w'),
    default='results.txt',
    help='Where to put the results.')
def parse_from_file(input_file, output_file, training_set_path):
    pcfg = PCFG()
    pcfg.fill_pcfg_from_file(training_set_path)
    print('Trained PCFG')

    parsed_tree_generator = generate_parsed_tree(pcfg, input_file)
    for parsable, tree in tqdm(parsed_tree_generator):
        if parsable:
            output_file.write(get_tree_sequoia_format(tree) + '\n')
        else:
            output_file.write('()\n')
            print('NOT PARSED : {}'.format(tree))

    input_file.close()
    output_file.close()


def get_tree_sequoia_format(tree):
    return (' '.join(str(tree).split()))


def generate_parsed_tree(pcfg, lines_to_parse):
    for line in lines_to_parse:
        parsable, tree = pcfg.probabilitic_CKY(line.split())
        yield parsable, tree


if __name__ == '__main__':
    main()
