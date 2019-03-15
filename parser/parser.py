import numpy as np
from nltk.tree import Tree

from collections import defaultdict, namedtuple
import re
from itertools import product


def clean_tag(tree):
    regex_cleaner = re.compile(r'(-)\w+')
    tree.set_label(regex_cleaner.sub('', tree.label()))
    for child in tree:
        if isinstance(child, Tree):
            clean_tag(child)


Rule = namedtuple('Rule', ['lhs', 'rhs'])


class Parser:
    def __init__(self, filepath):
        with open(filepath, 'r') as f:
            self.lines = f.readlines()
        self.rules = self._extract_rules(self.lines)

    def _extract_rules(self, lines):
        rules = []
        for word in lines:
            tree = Tree.fromstring(word)[0]
            clean_tag(tree)
            tree.chomsky_normal_form()
            tree.collapse_unary(collapseRoot=True, collapsePOS=True)
            rules.extend(tree.productions())
        return rules

    def get_rules(self):
        return self.rules


class PCFG:
    def __init__(self):
        self.grammar_proba = defaultdict(float)
        self.lexicon_proba = defaultdict(float)

        self.terminals = set()

        self.rev_grammar_proba = defaultdict(dict)
        self.rev_lexicon_proba = defaultdict(dict)

    def count_to_proba(self, count_dict, total_count_dict):
        proba_dict = defaultdict(float)
        for rule, count in count_dict.items():
            proba_dict[rule] = np.log(count / total_count_dict[rule.lhs])
        return proba_dict

    def fill_pcfg_from_file(self, filepath):
        rules = Parser(filepath).get_rules()

        lexicon_count = defaultdict(int)
        grammar_count = defaultdict(int)
        lhs_lexicon_count = defaultdict(int)
        lhs_grammar_count = defaultdict(int)

        for rule in rules:
            if rule.is_lexical():
                lexicon_count[Rule(rule.lhs(), rule.rhs()[0])] += 1
                lhs_lexicon_count[rule.lhs()] += 1
                self.terminals.add(rule.rhs()[0])
            else:
                grammar_count[Rule(rule.lhs(), rule.rhs())] += 1
                lhs_grammar_count[rule.lhs()] += 1

        self.grammar_proba = self.count_to_proba(grammar_count,
                                                 lhs_grammar_count)
        self.lexicon_proba = self.count_to_proba(lexicon_count,
                                                 lhs_lexicon_count)


    def reverse_dict(self):
        for rule, proba in self.grammar_proba.items():
            self.rev_grammar_proba[rule.rhs][rule.lhs] = proba

        for rule, proba in self.lexicon_proba.items():
            self.rev_lexicon_proba[rule.rhs][rule.lhs] = proba

    def probabilitic_CKY(self, words):
        proba_table = defaultdict(lambda: -np.inf)
        table = defaultdict(list)
        back = dict()

        for j in range(1, len(words) + 1):
            for A, proba in self.rev_lexicon_proba[words[j-1]].items():
                proba_table[(j - 1, j, A)] = proba
                table[(j - 1, j)].append(A)

            for i in range(j - 2, -1, -1):
                for k in range(i + 1, j):
                    current_non_term = product(table[(i, k)], table[(k, j)])
                    for B, C in current_non_term:
                        for A, proba in self.rev_grammar_proba[(B, C)].items():
                            cumulated_proba = proba + proba_table[(i, k, B)] + proba_table[(k, j, C)]
                            if (proba_table[i, j, A] < cumulated_proba):
                                proba_table[(i, j, A)] = cumulated_proba
                                table[(i, j)].append(A)
                                back[(i, j, A)] = (k, B, C)

        parsable = False
        maxi = -np.inf
        for n1, n2, A in proba_table.keys():
            if (n1, n2, A) == (0, len(words), A):
                if proba_table[0, len(words), A] > maxi:
                    h_prob = proba_table[0, len(words), A]
                    top_node = A
                    parsable = True

        if parsable:
            tree = self._build_tree(words, back, 0, len(words), top_node)
            tree.set_label('SENT')
            tree.un_chomsky_normal_form()
        else:
            tree = None

        return parsable, tree

    def _build_tree(self, words, back, i, j, node):
        tree = Tree(node.symbol(), children=[])
        if (i, j) == (j-1, j):
            tree.append(words[j-1])
            return tree
        else:
            if (i, j, node) in back.keys():
                k, b, c = back[i, j, node]
                tree.append(self._build_tree(words, back, i, k, b))
                tree.append(self._build_tree(words, back, k, j, c))
                return tree
            else:
                return tree
