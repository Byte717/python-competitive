import math
from functools import lru_cache

@lru_cache(maxsize=None)
def quicksort(arr):
    if len(arr) <= 1: return arr
    mid_index = len(arr) // 2
    pivot = arr[mid_index]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


def isSorted(arr):
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))


def binary_search(arr, target):
    if not isSorted(arr): arr = quicksort(arr)
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target: return mid
        left, right = (mid + 1, right) if arr[mid] < target else (left, mid - 1)
    return -1


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    return abs(a * b) // gcd(a, b)


def prime_factors(n):
    factors = []
    divisor = 2
    while n > 1:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1
    return factors


@lru_cache(maxsize=None)
def binomial(n, k, p):
    # return nCk mod p using recursion
    if k == 0 or k == n: return 1
    return (binomial(n - 1, k - 1, p) + binomial(n - 1, k, p)) % p


@lru_cache(maxsize=None)
def exp(x, n, m=1):  # x^n mod m
    x %= m
    res = 1
    while n > 0:
        if n % 2 == 1: res = (res * x) % m
        x = (x * x) % m
        n //= 2
    return res


@lru_cache(maxsize=None)
def factorial(n):
    if n == 1: return 1
    return n * factorial(n - 1)


@lru_cache(maxsize=None)
def matrix_determinant(matrix):
    if len(matrix) == 1: return matrix[0][0]
    det = 0
    for i in range(len(matrix)):
        submatrix = [row[:i] + row[i + 1:] for row in matrix[1:]]
        det += matrix[0][i] * (-1) ** i * matrix_determinant(submatrix)
    return det


def matMul(matrix1, matrix2):
    return [[sum(a * b for a, b in zip(row1, col)) for col in zip(*matrix2)] for row1 in matrix1]


def matrix_inverse(matrix):
    n = len(matrix)
    aug_matrix = [row + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        max_row = max(range(col, n), key=lambda i: abs(aug_matrix[i][col]))
        aug_matrix[col], aug_matrix[max_row] = aug_matrix[max_row], aug_matrix[col]
        aug_matrix[col] = [aug_matrix[col][i] / aug_matrix[col][col] for i in range(2 * n)]
        for row in range(n):
            if row != col:
                factor = aug_matrix[row][col]
                aug_matrix[row] = [aug_matrix[row][i] - factor * aug_matrix[col][i] for i in range(2 * n)]
    return [[aug_matrix[i][j + n] for j in range(n)] for i in range(n)]


def mod_inverse(a, m):
    m0, x0, x1 = m, 0, 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += m0
    return x1



def chinese_remainder_theorem(a_list, m_list):
    M = math.prod(m_list)
    x = 0
    for i in range(len(a_list)):
        Mi = M // m_list[i]
        Mi_inverse = mod_inverse(Mi, m_list[i])
        x += a_list[i] * Mi * Mi_inverse
    return x % M

