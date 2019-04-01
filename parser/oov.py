import pickle
import re
from collections import defaultdict, namedtuple
from operator import itemgetter

import numpy as np

EmbeddingContainer = namedtuple('EmbeddingContainer',
                                ('words', 'embeddings', 'word_id', 'id_word'))


class OOVHandler:

    UNK_TOKEN = '<UNK>'
    NPP_TOKEN = '<NPP>'
    DIGITS = re.compile("[0-9]", re.UNICODE)

    def __init__(self, vocabulary, polyglot_data_path):
        self.vocabulary = vocabulary

        # Polyglot data
        with open(polyglot_data_path, 'rb') as pg_file:
            polyglot_words, embeddings = pickle.load(
                pg_file, encoding='iso-8859-1')

        word_id = {w: i for (i, w) in enumerate(polyglot_words)}
        id_word = dict(enumerate(polyglot_words))
        self.all_embeddings = EmbeddingContainer(polyglot_words, embeddings,
                                                 word_id, id_word)

        vocabulary_in_lexicon = list(set(polyglot_words) & set(vocabulary))
        vocabulary_idx = [word_id[w] for w in vocabulary_in_lexicon]
        vocabulary_word_id = {
            w: i
            for (i, w) in enumerate(vocabulary_in_lexicon)
        }
        vocabulary_id_word = dict(enumerate(vocabulary_in_lexicon))
        self.vocabulary_embeddings = EmbeddingContainer(
            vocabulary_in_lexicon, embeddings[vocabulary_idx],
            vocabulary_word_id, vocabulary_id_word)

    def find_close_word(self, word, max_dist=2):
        """
        Returns a word that is in the available vocabulary.

        Strategy :
        1. If the word is already in the available vocabulary,
        returns the word.
        2. If no, check if the word is in polyglot. If so, that means that it
        is not a typo. Get a similar word using the embeddings that is in the
        available vocabulary.
        3. The word has probably a typo : look for a word that has a close editing
        distance.
        4. If no word is found, return UNK token
        """
        if word in self.vocabulary:
            return word

        # Check if the word is in in the polyglot list of words
        word_in_embeddings, word = self._is_in_embeddings(word)
        if word_in_embeddings:
            word_idx = self.all_embeddings.word_id[word]
            idx_in_voc = self._l2_nearest_in_voc(word_idx)
            return idx_in_voc

        # Check edit distance
        dist_dict = defaultdict(list)
        for available_word in self.vocabulary:
            distance = levenshtein_distance(word, available_word, max_dist)
            if distance <= max_dist:
                dist_dict[distance].append(available_word)

        for dist in range(max_dist):
            if dist_dict[dist]:
                return dist_dict[dist][0]

        if word[0].isupper():
            return self.NPP_TOKEN

        return self.UNK_TOKEN

    def _is_in_embeddings(self, word):
        if word in self.all_embeddings.words:
            return True, word

        if not word in self.all_embeddings.words:
            word = self.DIGITS.sub("#", word)

        if word in self.all_embeddings.words:
            return True, word
        if word.lower() in self.all_embeddings.words:
            return True, word.lower()
        if word.upper() in self.all_embeddings.words:
            return True, word.upper()
        if word.title() in self.all_embeddings.words:
            return True, word.title()

        return False, word

    def _l2_nearest_in_voc(self, word_index, k=5):
        """
        Return the nearest word available in the vocabulary
        """
        word_embedding = self.all_embeddings.embeddings[word_index]
        distances = ((self.vocabulary_embeddings.embeddings - word_embedding)
                     **2).sum(axis=1)
        sorted_distances = sorted(enumerate(distances), key=itemgetter(1))
        return self.vocabulary_embeddings.id_word[sorted_distances[0][0]]

def levenshtein_distance(s1, s2, early_stopping=None):
    """
    Compute the Levenshtein distance between two strings

    Args:
    - s1, s2 (str)
    - early_stopping (int) : if not None, compare the lenght of s1 and s2. This
      difference is a lower bound on the distance. If it is strictly greater
      than early_stopping, return early_stopping + 1

    Returns:
    - distance (int)
    """

    if early_stopping and (np.abs(len(s1) - len(s2)) > early_stopping):
        return early_stopping + 1

    d = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)
    cost = 0

    for i in range(len(s1) + 1):
        d[i, 0] = i
    for j in range(len(s2) + 1):
        d[0, j] = j

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + cost)
    return d[-1, -1]

