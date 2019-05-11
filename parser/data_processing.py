from os.path import join
import re

import click
import numpy as np

# Size of each dataset
TRAIN_DATASET_SIZE = 0.8
DEV_DATASET_SIZE = 0.1
TEST_DATASET_SIZE = 0.1

DATASET_NAMES = ['train', 'dev', 'test']

# regex to extract string in the most inner parenthesis
INNER_PARENTHESIS = re.compile(r'\(([^()]+)\)')


def tree_to_sentence(line):
    """Get the sequence of token from a treebank sample"""
    matchs = INNER_PARENTHESIS.findall(line)
    regex_cleaner = re.compile(r'(-)\w+')
    term_tags = []
    for match in matchs:
        non_term, term = match.split()
        non_term = regex_cleaner.sub('', non_term)
        term_tags.append(term)
    return ' '.join(term_tags) + '\n'


@click.command()
@click.argument('sequoia_corpus_path', type=click.Path())
@click.option('--output_folder', type=click.Path())
def split_dataset(sequoia_corpus_path, output_folder):
    with open(sequoia_corpus_path, 'r') as f:
        dataset = f.readlines()

    train_idx = int(TRAIN_DATASET_SIZE * len(dataset))
    dev_idx = train_idx + int(DEV_DATASET_SIZE * len(dataset))

    datasets = np.split(dataset, (train_idx, dev_idx))

    for split_ds, name in zip(datasets, DATASET_NAMES):
        with open(join(output_folder, '{}.tag'.format(name)), 'w') as f:
            f.writelines(split_ds)
        with open(join(output_folder, '{}.no_tag'.format(name)), 'w') as f:
            f.writelines([tree_to_sentence(line) for line in split_ds])

    return True


if __name__ == '__main__':
    split_dataset()
