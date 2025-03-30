#!/usr/bin/env python3
"""
assignment_3.py

Solves the following tasks:

1. Solve a given system of linear equations (via Gaussian Elimination).
2. Perform an LU factorization of a given matrix (without scipy).
3. Check if a matrix is (strictly) diagonally dominant.
4. Check if a matrix is positive definite.
"""

from copy import deepcopy

def gaussian_elimination_solve(aug_matrix):
    """
    Solve a system of linear equations via Gaussian elimination
    and back-substitution. The input aug_matrix is a list of lists
    containing the augmented matrix of size n x (n+1).

    Returns: a list containing the solution [x1, x2, ..., xn].
    """
    # Make a deep copy if needed, or operate in-place:
    mat = [row[:] for row in aug_matrix]  # safer copy
    n = len(mat)

    # Forward Elimination
    for i in range(n):
        # Simple partial pivot (optional for stability):
        max_row = i
        max_val = abs(mat[i][i])
        for r in range(i+1, n):
            if abs(mat[r][i]) > max_val:
                max_val = abs(mat[r][i])
                max_row = r
        if max_row != i:
            mat[i], mat[max_row] = mat[max_row], mat[i]

        # Eliminate below row i
        pivot = mat[i][i]
        if abs(pivot) < 1e-14:
            raise ValueError("Matrix is singular or nearly singular.")

        for r in range(i+1, n):
            factor = mat[r][i] / pivot
            for c in range(i, n+1):
                mat[r][c] -= factor * mat[i][c]

    # Back-substitution
    x = [0]*n
    for i in reversed(range(n)):
        sum_ax = 0
        for j in range(i+1, n):
            sum_ax += mat[i][j] * x[j]
        x[i] = (mat[i][n] - sum_ax) / mat[i][i]

    return x


def lu_factorization(A):
    """
    Perform an LU factorization of matrix A (square) without using scipy.
    We assume A is n x n, given as a list of lists.

    Returns: (L, U, detA)
      where L is lower-triangular (unit diagonal),
            U is upper-triangular,
            detA is the determinant of A (product of diag(U)).
    If the matrix is singular, the product of diag(U) will be zero.
    """
    n = len(A)
    # Initialize L as identity, U as zero
    L = [[0]*n for _ in range(n)]
    U = [[0]*n for _ in range(n)]

    # Doolittle method (no pivot for simplicity).
    for i in range(n):
        L[i][i] = 1.0

    for i in range(n):
        # Compute U[i][j] for j >= i
        for j in range(i, n):
            s = 0.0
            for k in range(i):
                s += L[i][k] * U[k][j]
            U[i][j] = A[i][j] - s

        # Compute L[j][i] for j > i
        for j in range(i+1, n):
            s = 0.0
            for k in range(i):
                s += L[j][k] * U[k][i]
            if abs(U[i][i]) < 1e-14:
                L[j][i] = 0.0
            else:
                L[j][i] = (A[j][i] - s) / U[i][i]

    # Determinant is product of diagonal of U
    detA = 1.0
    for i in range(n):
        detA *= U[i][i]

    return L, U, detA


def is_diagonally_dominant(A):
    """
    Check if the square matrix A is (strictly) diagonally dominant.
    Returns True if for every i, |A[i][i]| > sum(|A[i][j]| for j != i).
    Otherwise returns False.
    """
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        off_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag <= off_sum:
            return False
    return True


def is_positive_definite(A):
    """
    Check if the square matrix A is positive definite using
    the principal minors criterion:
      - All leading principal minors must be > 0.
    Returns True or False.
    """
    n = len(A)

    def det_submatrix(k):
        # Determinant of top-left k x k submatrix
        sub = [row[:k] for row in A[:k]]
        return determinant(sub)

    def determinant(M):
        # Compute determinant via mini Gaussian elimination
        mat = deepcopy(M)
        size = len(mat)
        det_val = 1.0
        for i in range(size):
            pivot = mat[i][i]
            if abs(pivot) < 1e-14:
                return 0.0
            det_val *= pivot
            for r in range(i+1, size):
                factor = mat[r][i] / pivot
                for c in range(i, size):
                    mat[r][c] -= factor * mat[i][c]
        return det_val

    for k in range(1, n+1):
        if det_submatrix(k) <= 0:
            return False
    return True


def main():
    """
    Main driver to demonstrate each part of the assignment.
    """

    # 1) Solve via Gaussian Elimination
    aug_matrix = [
        [2, -1, 1, 6],
        [1, 3, 1, 0],
        [-1, 5, 4, -3]
    ]

    # aug_matrix = [
    #     [2, 3, -1, 5],
    #     [4, -2, 1, 1],
    #     [-2, 1, 2, 3]
    # ]

    solution = gaussian_elimination_solve(aug_matrix)
    print("1) Solution to the system [Gaussian Elimination]:")
    for i, val in enumerate(solution):
        # Print the float directly (no rounding or formatting)
        print("   x{} =".format(i+1), val)
    print()

    # 2) LU Factorization
    A_lu = [
        [1,  1,  0, 3],
        [2,  1, -1, 1],
        [3, -1, -1, 2],
        [-1, 2,  3, -1]
    ]
    L, U, detA = lu_factorization(A_lu)
    print("2) LU Factorization Results:")
    print("   Determinant of A:", detA)
    print("   L matrix:")
    for row in L:
        print("     ", [int(round(x)) for x in row])
    print("   U matrix:")
    for row in U:
        print("     ", [int(round(x)) for x in row])
    print()

    # 3) Check if a matrix is diagonally dominant
    A_dd = [
        [9, 0, 5, 2, 1],
        [3, 9, 1, 2, 1],
        [0, 1, 7, 2, 3],
        [4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8]
    ]
    dd = is_diagonally_dominant(A_dd)
    print("3) Is the matrix diagonally dominant?", dd)
    print()

    # 4) Check if the matrix is positive definite
    A_pd = [
        [2, 2, 1],
        [2, 3, 0],
        [1, 0, 2]
    ]
    pd = is_positive_definite(A_pd)
    print("4) Is the matrix positive definite?", pd)
    print()


if __name__ == "__main__":
    main()
