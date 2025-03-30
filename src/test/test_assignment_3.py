#!/usr/bin/env python3
"""
test_assignment_3.py

Example test file for assignment_3.py
"""

import unittest
from src.main.assignment_3 import (
    gaussian_elimination_solve,
    lu_factorization,
    is_diagonally_dominant,
    is_positive_definite
)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))



class TestAssignment3(unittest.TestCase):
    def test_gaussian_elimination(self):
        # System:
        #   2x +  1y +  1z = 4
        #   1x +  1y + -1z = 1
        #   1x + -1y +  1z = 2
        aug_matrix = [
            [2, -1, 1, 6],
            [1, 3, 1, 0],
            [-1, 5, 4, -3]
        ]
        sol = gaussian_elimination_solve(aug_matrix)
        # We found solution (x, y, z) = (1.5, 0.25, 0.75)
        self.assertAlmostEqual(sol[0], 2.0, places=7)
        self.assertAlmostEqual(sol[1], -1.0, places=7)
        self.assertAlmostEqual(sol[2], 1., places=7)

    def test_lu_factorization(self):
        A = [
            [1,  1,  0, 3],
            [2, 1,  -1, 1],
            [3, -1,  -1, 2],
            [-1, 2, 3, -1]
        ]
        L, U, detA = lu_factorization(A)
        # Because the matrix is singular, determinant should be 0.
        self.assertAlmostEqual(detA, 39.0, places=7)
        # Check shapes of L and U
        self.assertEqual(len(L), 4)
        self.assertEqual(len(U), 4)

    def test_diagonally_dominant(self):
        A_dd = [
            [9, 0, 5, 2, 1],
            [3, 9, 1, 2, 1],
            [0, 1, 7, 2, 3],
            [4, 2, 3, 12, 2],
            [3, 2, 4, 0, 8]
        ]
        self.assertFalse(is_diagonally_dominant(A_dd))

    def test_positive_definite(self):
        A_pd = [
            [2, 2, 1],
            [2, 3, 0],
            [1, 0, 2]
        ]
        self.assertTrue(is_positive_definite(A_pd))

if __name__ == '__main__':
    unittest.main()
