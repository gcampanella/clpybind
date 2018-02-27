import numpy as np
import unittest

import clpybind


class TestClp(unittest.TestCase):
    def test_solve(self):
        x = clpybind.Matrix(np.array([
            [3, 2, 5],
            [2, 1, 1],
            [1, 1, 3],
            [5, 2, 4]
        ], dtype=np.double))
        col_lb = np.zeros(3, dtype=np.double)
        col_ub = np.repeat(np.infty, 3)
        obj = np.array([20, 10, 15], dtype=np.double)
        row_lb = np.repeat(-np.infty, 4)
        row_ub = np.array([55, 26, 30, 57], dtype=np.double)
        solver = clpybind.Simplex(x, col_lb, col_ub, obj, row_lb, row_ub)
        solver.log_level = clpybind.Simplex.LogLevel.Off
        solver.optimization_direction = clpybind.Simplex.OptimizationDirection.Maximize
        solver.initial_solve()
        np.testing.assert_equal(solver.status,
                                clpybind.Simplex.ProblemStatus.Optimal)
        np.testing.assert_almost_equal(solver.objective_value, 268.0)
        np.testing.assert_almost_equal(solver.solution, [1.8, 20.8, 1.6])
        np.testing.assert_almost_equal(solver.reduced_costs, np.zeros(3))
        np.testing.assert_almost_equal(solver.shadow_prices, [1, 6, 0, 1])
