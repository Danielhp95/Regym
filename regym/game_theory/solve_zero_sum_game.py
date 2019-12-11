from typing import List, Tuple
import numpy as np
import cvxopt
tol = 1e-07
cvxopt.solvers.options["maxtol"] = tol
cvxopt.solvers.options["feastol"] = tol
cvxopt.solvers.options["show_progress"] = False


def solve_zero_sum_game(matrix: np.array) \
        -> Tuple[List[float], List[float], float, float]:
    '''
    Computes one (not all!) Nash Equilibrium for the input :param matrix:
    2-player 0-sum game and its corresponding minimax value for both players.

    If :param matrix: is antisymmetrical, computation for column player is optimized.
    This is because if the :param matrix: is antisymmetrical, the underlying game
    is symmetric, and the support for actions and minimax value for the row player
    is the same as the column player's.

    :param matrix: Payoff matrix for row player of a zero-sum game
    :returns: (support over row actions, support over column actions,
               minimax value for player1, minimax value for player 2)
    '''
    if not isinstance(matrix, np.ndarray): matrix = np.array(matrix)
    check_parameter_validity(matrix)

    solution_player1 = solve_for_row_player(matrix)
    solution_player2 = solution_player1 if is_matrix_antisymetrical(matrix) else solve_for_row_player(-1 * matrix.T)
    return (np.array(solution_player1[0]), np.array(solution_player2[0]),
            solution_player1[1], solution_player2[1])


def solve_for_row_player(matrix: np.array) -> Tuple[cvxopt.base.matrix, float]:
    r'''
    Solving the :param matrix: game for the row player corresponds to finding
    a mixed strategy (a probability distribution over the actions [rows])
    such that the value obtained by playing such mixed strategy is maximized

    This problem can be specified as a linear program with the following variables:
        - |A_{row_player| variables corresponding to the support for each action (s).
          where A_{row_player} is the action space for the row player.
        - 1 variable corresponding to the minimax value (V)

    Formulation:

        maximize V
        such that:
              (1)  \sum_{a \in A_{row_player} U_{row} (a, a_j) * s_a >= V   \forall a_j \in A_{column_player}
              (2)  s_a >= 0  \forall a \in A_{row_player} (Prababilities must be positive)
              (3)  \sum_{a \in A_{row_player} s_a = 1 (Probabilities sum to one)

    For more detail, please refer to Sec 4.1 of Shoham & Leyton-Brown, 2009:
    Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations
    http://www.masfoundations.org/mas.pdf

    :param matrix: Payoff matrix for row player of a zero-sum game
    :returns: (support for row player, minimax value for row player)
    '''
    c, g_mat, h, a_mat, b = generate_solver_compliant_matrices(matrix)
    solution = cvxopt.solvers.lp(c, g_mat, h, a_mat, b)
    # Last element of 'solution['x'] is the minimax value
    return (solution['x'][:-1], solution['x'][-1])


def generate_solver_compliant_matrices(matrix: np.array) \
        -> Tuple[cvxopt.base.matrix, cvxopt.base.matrix, cvxopt.base.matrix,
                 cvxopt.base.matrix, cvxopt.base.matrix]:
    '''
    From http://cvxopt.org/userguide/coneprog.html#linear-programming,
    CVXOPT uses the formulation:
       minimize: c^t x
          s.t.   Gx <= h
                 Ax = b

    Here:
     - x is the vector the variables
     - c is the vector of objective coefficients
     - G is the matrix of LEQ (and GEQ) constraint coefficients
     - h is the vector or right-hand side values of the LEQ/GEQ constraints
     - A is the matrix of equality constraint coefficients
     - b is the vector of right-hand side values of the equality constraints

    This function builds these sparse matrices from :param matrix:
    :returns: (c, G, h, A, b) matrices
    '''
    h, w = matrix.shape
    # One support for each row action and one for the minimax value
    num_variables = h+1
    # One for maximizing over each column player pure strategy and one for each
    # support for each row action being positive (inequalities (1) and (2))
    num_leq_constraints = (w + h)
    # One for the sum of supports for row actions summing up to 1  (equality (3))
    num_eq_constraints = 1

    c = compute_objective_coefficients(num_variables)

    g_mat = compute_leq_constraint_coefficients(num_variables, num_leq_constraints, matrix)
    h = cvxopt.matrix([0.0] * num_leq_constraints) # Right hand side of inequality constraints

    a_mat = compute_eq_constraint_coefficients(num_eq_constraints, num_variables)
    b = cvxopt.matrix([0.0] * num_eq_constraints); b[0, 0] = 1.0 # Right hand side of equality constraint Probability distribution should sum to one
    return c, g_mat, h, a_mat, b


def compute_objective_coefficients(num_variables: int) -> cvxopt.base.matrix:
    '''
    The objective coefficients (c) is a vector of length :param num_variables:
        - c[:-1] = [0, ..., 0]. Because the variables representing the support
                                are not part of the minimization
        - c[-1] = -1            Because we want to minimize the negative
                                minimax value (aka. maximize minimax value)

    :return: objective coefficient vector (1 x num_variables)
    '''
    c = cvxopt.matrix([0.0] * num_variables)
    c[-1] = -1.0
    return c


def compute_eq_constraint_coefficients(num_eq_constraints: int, num_variables: int) -> cvxopt.base.matrix:
    '''
    The equality constraint coefficients is a matrix of dimensions:
    (:param num_eq_constraints:, :param num_variables:). Corresponding to
    equality (3) from the equation in comments above.  In this case
    there is only 1 equality constraint (all support variables must sum up to one)

    :returns: equality constraint coefficients matrix (1 x num_variables)
    '''

    a_mat = cvxopt.spmatrix([], [], [], (num_eq_constraints, num_variables))
    for i in range(a_mat.size[1] - 1): a_mat[i] = 1.0
    return a_mat


def compute_leq_constraint_coefficients(num_variables: int, num_leq_constraints: int, payoff_matrix: np.array) -> cvxopt.base.matrix:
    '''
    The inequality constraint coefficient is a matrix of dimensions:
    (:param num_leq_constraints: :param num_variables). Corresponding
    to inequalities (1) and (2) from the equation in comments above.

    :returns: inequality constraint coefficients matrix (num_leq_constraints x num_variables)
    '''
    g_mat = cvxopt.spmatrix([], [], [], (num_leq_constraints, num_variables))

    # Inequalities from (1)
    for i in range(payoff_matrix.shape[1]): # For each column
        for j in range(payoff_matrix.shape[0]):
            g_mat[i, j] = -1 * payoff_matrix[j, i]
        g_mat[i, -1] = 1.0

    # inequalities form (2)
    j = 0
    for i in range(payoff_matrix.shape[1], g_mat.size[0]):
        g_mat[i, j] = -1.0
        j += 1
    return g_mat


def is_matrix_antisymetrical(m: np.array) -> bool:
    '''
    Checks if the following properties hold for matrix :param m::
        (1) m is square
        (2) m = -m^T

    :returns: whether (1) and (2) hold
    '''
    return m.shape[0] == m.shape[1] and np.allclose(m, -1 * m.T, rtol=1e-03, atol=1e-03)


def check_parameter_validity(matrix):
    if (matrix.ndim and matrix.size) == 0: raise ValueError('Game matrix should not be empty')
    if matrix is None: raise ValueError('Input matrix was None. It should be a 2D matrix of Floats')
    if isinstance(matrix, np.ndarray):
        if matrix.dtype.kind not in np.typecodes['AllInteger'] and \
           matrix.dtype.kind not in np.typecodes['AllFloat']: raise ValueError('Input matrix should contain floats or integers')
