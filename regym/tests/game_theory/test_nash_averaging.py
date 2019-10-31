import pytest
import unittest
import numpy as np

from regym.game_theory import compute_nash_averaging, compute_nash_average


class NashAveragingTest(unittest.TestCase):

    def test_for_none_game_raises_valueerror(self):
        with pytest.raises(ValueError) as _:
            _ = compute_nash_averaging(None)

    def test_for_empty_game_raises_valueerror(self):
        with pytest.raises(ValueError) as _:
            _ = compute_nash_averaging([])

    def test_for_empty_numpy_array_game_raises_valueerror(self):
        with pytest.raises(ValueError) as _:
            _ = compute_nash_averaging(np.array(None))

    def test_for_non_integer_or_float_list_raises_valueerror(self):
        with pytest.raises(ValueError) as _:
            _ = compute_nash_averaging([['a', 'b']])

    def test_for_non_antisymmetric_matrix_raises_valueerror(self):
        random_winrate_matrix = [[0.5, 0.2], [0.8,0.5]]
        with pytest.raises(ValueError) as _:
            _ = compute_nash_averaging(random_winrate_matrix)

    def test_redundancy_invariance(self):
        payoff_matrix = np.array([[0, 1, -1, -1], [-1, 0, 1, 1], [1, -1, 0, 0], [1, -1, 0, 0]])
        ne, rating = compute_nash_average(4.6 * payoff_matrix, steps=2**10)
        np.testing.assert_array_almost_equal(ne, np.array([1 / 3, 1 / 3, 1 / 6, 1 / 6]))
        np.testing.assert_array_almost_equal(rating, np.zeros(4), decimal=4)

    def test_continuity_balduzzi_eg2(self):
        C = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
        T = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])

        for e in [0.25, 0.75]:
            payoff = C + e * T
            ne, rating = compute_nash_average(payoff, steps=2**10)
            if e <= 0.5:
                np.testing.assert_array_almost_equal(ne,
                                                     np.array([(1 + e) / 3, (1 - 2 * e) / 3,
                                                               (1 + e) / 3]))
                np.testing.assert_array_almost_equal(rating, np.zeros(3))
            else:
                np.testing.assert_array_almost_equal(ne, np.array([1, 0, 0]))
                np.testing.assert_array_almost_equal(rating, np.array([0, -1 - e, 1 - 2 * e]))
