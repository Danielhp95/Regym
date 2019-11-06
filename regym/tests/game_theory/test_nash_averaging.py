import pytest
import numpy as np

from regym.game_theory import compute_nash_averaging, compute_nash_average


def test_for_none_game_raises_valueerror():
    with pytest.raises(ValueError) as _:
        _ = compute_nash_averaging(None)


def test_for_empty_game_raises_valueerror():
    with pytest.raises(ValueError) as _:
        _ = compute_nash_averaging([])


def test_for_empty_numpy_array_game_raises_valueerror():
    with pytest.raises(ValueError) as _:
        _ = compute_nash_averaging(np.array(None))


def test_for_non_integer_or_float_list_raises_valueerror():
    with pytest.raises(ValueError) as _:
        _ = compute_nash_averaging([['a', 'b']])


def test_for_non_antisymmetric_matrix_raises_valueerror():
    random_winrate_matrix = [[0.5, 0.2], [0.8, 0.5]]
    with pytest.raises(ValueError) as _:
        _ = compute_nash_averaging(random_winrate_matrix)


def test_monoaction_game():
    payoff_matrix = np.array([[0]])
    expected_nash_average     = np.array([0.])
    expected_nash_equilibrium = np.array([1.])

    actual_nash_equilibrium, \
        actual_nash_averaging = compute_nash_average(payoff_matrix, steps=2**10)

    np.testing.assert_array_almost_equal(actual_nash_equilibrium,
                                         expected_nash_equilibrium)
    np.testing.assert_array_almost_equal(actual_nash_averaging,
                                         expected_nash_average)


def test_game_with_only_zero_payoffs():
    '''
    Rationale: Nash Averagings are often used in winrate matrices
               These matrices may feature 50% winrates between two policies
               if that's the case. Then the log-odds transformation will
               make the 0.5 winrate into a 0. This can break certain division operations
    '''
    payoff_matrix = np.array([[0., 0.],
                              [0., 0.]])
    expected_nash_average     = np.array([0., 0.])
    expected_nash_equilibrium = np.array([1/2, 1/2])

    actual_nash_equilibrium, \
        actual_nash_averaging = compute_nash_average(payoff_matrix, steps=2**10)

    np.testing.assert_array_almost_equal(actual_nash_equilibrium,
                                         expected_nash_equilibrium)
    np.testing.assert_array_almost_equal(actual_nash_averaging,
                                         expected_nash_average)


def test_redundancy_invariance():
    payoff_matrix = np.array([[0, 1, -1, -1], [-1, 0, 1, 1], [1, -1, 0, 0], [1, -1, 0, 0]])
    ne, rating = compute_nash_average(4.6 * payoff_matrix, steps=2**10)
    np.testing.assert_array_almost_equal(ne, np.array([1 / 3, 1 / 3, 1 / 6, 1 / 6]))
    np.testing.assert_array_almost_equal(rating, np.zeros(4), decimal=4)


def test_continuity_balduzzi_eg2():
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
