import pytest
import unittest
import numpy as np
from regym.game_theory import solve_zero_sum_game


class ZeroSumGameSolver(unittest.TestCase):

    def test_for_none_game_raises_valueerror(self):
        with pytest.raises(ValueError) as _:
            _ = solve_zero_sum_game(None)

    def test_for_empty_game_raises_valueerror(self):
        with pytest.raises(ValueError) as _:
            _ = solve_zero_sum_game([])

    def test_for_empty_numpy_array_game_raises_valueerror(self):
        with pytest.raises(ValueError) as _:
            _ = solve_zero_sum_game(np.array(None))

    def test_for_non_integer_or_float_list_raises_valueerror(self):
        with pytest.raises(ValueError) as _:
            _ = solve_zero_sum_game([['a', 'b']])

    # Symmetric game
    def test_rock_paper_scissors(self):
        rock_paper_scissors = [[0.0, -1.0, 1.0],
                               [1.0, 0.0, -1.0],
                               [-1.0, 1.0, 0.0]]
        self._test_game(game=rock_paper_scissors,
                        expected_supports=([1/3, 1/3, 1/3], [1/3, 1/3, 1/3]),
                        expected_minimax_values=(0, 0))

    def test_game_different_action_spaces(self):
        different_action_spaces = [[0.0, 0.0],
                                   [1.0, 1.0],
                                   [2.0, 2.0]]
        self._test_game(game=different_action_spaces,
                        expected_supports=([0, 0, 1], [1/2, 1/2]), # Player 2 is indifferent between both actions
                        expected_minimax_values=(2, -2))

    def test_game_different_action_spaces_2(self):
        different_action_spaces = [[0.0, 1.0, 2.0],
                                   [0.0, 1.0, 2.0]]
        self._test_game(game=different_action_spaces,
                        expected_supports=([1/2, 1/2], [1, 0, 0]), # Player 1 is indifferent between both actions
                        expected_minimax_values=(0, 0))

    def _test_game(self, game, expected_supports, expected_minimax_values):
        support_p1, support_p2, \
        minimax_val_p1, minimax_val_p2 = solve_zero_sum_game(game)

        supports = (support_p1, support_p2)
        minimax_vals = (minimax_val_p1, minimax_val_p2)

        # Supports should range over the same number of actions
        for actual_s, expected_s in zip(supports, expected_supports):
            self.assertEqual(len(actual_s), len(expected_s))

        # Supports should distribute probability mass as expected
        for actual_s, expected_s in zip(supports, expected_supports):
            for actual_action_support, expected_action_support in zip(actual_s, expected_s):
                self.assertAlmostEqual(actual_action_support, expected_action_support)

        # Minimax values should match
        for actual_minimax_val, expected_minimax_val in zip(minimax_vals, expected_minimax_values):
            self.assertAlmostEqual(actual_minimax_val, expected_minimax_val)
