from .context import oov

class TestLevenshtein:
    """
    Test the Levenshtein distance function
    """

    def test_simple_case(self):
        """
        Simple case
        """
        assert oov.levenshtein_distance('chat', 'chqt') == 1

    def test_same_string(self):
        """
        Same string
        """
        assert oov.levenshtein_distance('pomme', 'pomme') == 0

    def test_different_lenght(self):
        """
        Different lenght
        """
        assert oov.levenshtein_distance('pistou', 'pistouche') == 3

    def test_symmetry(self):
        """
        Test function symmetry
        """
        s1, s2 = 'peluche', 'trois'
        assert oov.levenshtein_distance(s1, s2) == oov.levenshtein_distance(s2, s1)
