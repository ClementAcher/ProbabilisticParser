from tqdm import tqdm
import re

from .parser import PCFG, UNARY_JOIN_CHAR
from .oov import OOVHandler
from .data_processing import INNER_PARENTHESIS


def tree_to_tag_terminal(line):
    """
    Get the (POS, terminal) tuples from a tree.
    """
    matchs = INNER_PARENTHESIS.findall(line)
    tag_terminals = []
    regex_cleaner = re.compile(r'(-)\w+')

    for match in matchs:
        pos, word = match.split()
        pos = regex_cleaner.sub('', pos)
        tag_terminals.append((pos, word))

    return tag_terminals


def evaluate_oov_strategy(train_path,
                          test_path,
                          polyglot_path='./data/polyglot-fr.pkl'):
    """
    Evaluate the OOV strategy : trains a PCFG on a dataset and a OOVHandler,
    then checks if words from an unknown dataset are attributed the right
    POS.
    """
    pcfg = PCFG()
    pcfg.fill_pcfg_from_file(train_path, polyglot_path)

    with open(test_path, 'r') as test_file:
        trees = test_file.readlines()

    tag_counter, failures = 0, []

    for tree in tqdm(trees):
        for tag, word in tree_to_tag_terminal(tree):
            tag_counter += 1
            close_word = pcfg.oov_handler.find_close_word(word)
            infered_POS = [
                str(tag).split(UNARY_JOIN_CHAR)[-1]
                for tag in pcfg.rev_lexicon_proba[close_word].keys()
            ]
            if tag not in infered_POS:
                failures.append((tag, word))

    accuracy = (tag_counter - len(failures)) / tag_counter

    return accuracy, failures
