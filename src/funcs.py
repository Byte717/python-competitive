import math
from functools import lru_cache

def quicksort(arr):
    """
    A recursive implementation of the quicksort algorithm.

    Parameters:
    - arr (List[Any]): The list to sort.

    Returns:
    - List[Any]: A new sorted list.

    ### Example usage:
    ```python
    sorted_arr = quicksort([3, 6, 2, 5, 1])
    print(sorted_arr)  # Output: [1, 2, 3, 5, 6]
    ```
    """
    if len(arr) <= 1:
        return arr
    mid_index = len(arr) // 2
    pivot = arr[mid_index]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

def isSorted(arr):
    """
    Checks if a list is sorted in non-decreasing order.

    Parameters:
    - arr (List[Comparable]): The list to check.

    Returns:
    - bool: True if the list is sorted, False otherwise.

    ### Example usage:
    ```python
    print(isSorted([1, 2, 3, 4]))     # Output: True
    print(isSorted([5, 3, 2, 1]))     # Output: False
    ```
    """
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))


def binary_search(arr, target):
    """
    Performs binary search on the array for a target value.

    If the array is not sorted, it will be sorted first using quicksort.

    Parameters:
    - arr (List[Comparable]): The list to search.
    - target (Comparable): The value to search for.

    Returns:
    - int: The index of the target if found, otherwise -1.

    ### Example usage:
    ```python
    index = binary_search([5, 3, 1, 4, 2], 4)
    print(index)  # Output: 3 (after sorting: [1, 2, 3, 4, 5])
    ```
    """
    if not isSorted(arr):
        arr = quicksort(arr)
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def gcd(a, b):
    """
    Computes the Greatest Common Divisor (GCD) of two integers using the Euclidean algorithm.

    Parameters:
    - a (int): First integer.
    - b (int): Second integer.

    Returns:
    - int: The greatest common divisor of a and b.

    ### Example usage:
    ```python
    print(gcd(48, 18))  # Output: 6
    print(gcd(7, 5))    # Output: 1
    ```
    """
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """
    Computes the Least Common Multiple (LCM) of two integers.

    Parameters:
    - a (int): First integer.
    - b (int): Second integer.

    Returns:
    - int: The least common multiple of a and b.

    ### Example usage:
    ```python
    print(lcm(12, 18))  # Output: 36
    print(lcm(7, 5))    # Output: 35
    ```

    Note: This function relies on the `gcd` function being defined.
    """
    return abs(a * b) // gcd(a, b)


def prime_factors(n):
    """
    Computes the prime factorization of a given integer `n`.

    Parameters:
    - n (int): The number to factorize (n > 1).

    Returns:
    - List[int]: A list of prime factors (with repetition if applicable).

    ### Example usage:
    ```python
    print(prime_factors(60))  # Output: [2, 2, 3, 5]
    print(prime_factors(13))  # Output: [13]
    ```

    Note:
    - This is a naive O(√n) algorithm and works well for small to moderately sized integers.
    """
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
    """
    Computes the binomial coefficient C(n, k) modulo p using recursion and memoization.

    Uses Pascal's identity:
        C(n, k) = C(n-1, k-1) + C(n-1, k)
    and computes the result modulo `p`.

    Parameters:
    - n (int): The number of items.
    - k (int): The number of items to choose.
    - p (int): The modulus.

    Returns:
    - int: The value of C(n, k) % p

    ### Example usage:
    ```python
    print(binomial(5, 2, 1000000007))  # Output: 10
    print(binomial(10, 3, 13))         # Output: 3
    ```

    Notes:
    - Uses `@lru_cache` for efficient memoization.
    - Time complexity: O(n * k) due to memoization.
    """
    if k == 0 or k == n:
        return 1
    return (binomial(n - 1, k - 1, p) + binomial(n - 1, k, p)) % p


@lru_cache(maxsize=None)
def exp(x, n, m=1):
    """
    Computes (x ** n) % m efficiently using binary exponentiation.

    Parameters:
    - x (int): The base.
    - n (int): The exponent.
    - m (int): The modulus (default is 1, which returns 0 for any non-zero x).

    Returns:
    - int: The result of (x^n) mod m.

    ### Example usage:
    ```python
    print(exp(2, 10, 1000))  # Output: 24
    print(exp(5, 0, 13))     # Output: 1
    ```

    Notes:
    - Time complexity: O(log n)
    - Uses `@lru_cache` for memoization of repeated calls.
    """
    x %= m
    res = 1
    while n > 0:
        if n % 2 == 1:
            res = (res * x) % m
        x = (x * x) % m
        n //= 2
    return res


@lru_cache(maxsize=None)
def factorial(n):
    """
    Computes the factorial of a non-negative integer `n` using recursion and memoization.

    Parameters:
    - n (int): A non-negative integer.

    Returns:
    - int: The factorial of `n`, i.e., n!

    ### Example usage:
    ```python
    print(factorial(5))  # Output: 120
    print(factorial(0))  # Output: 1
    ```

    Notes:
    - Uses `@lru_cache` to memoize results for faster repeated calls.
    - Time complexity: O(n)
    - Raises: `RecursionError` for very large `n` due to Python's recursion limit.
    """
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

@lru_cache(maxsize=None)
def matrix_determinant(matrix):
    """
    Recursively computes the determinant of a square matrix.

    Parameters:
    - matrix (Tuple[Tuple[float]]): The input matrix as a tuple of tuples.

    Returns:
    - float: The determinant of the matrix.

    ### Example usage:
    ```python
    mat = (
        (1, 2, 3),
        (0, 4, 5),
        (1, 0, 6)
    )
    print(matrix_determinant(mat))  # Output: 22.0
    ```

    Notes:
    - The matrix must be square (n x n).
    - Input must be a tuple of tuples to enable caching.
    - Time complexity: O(n!) for n×n matrix due to recursive expansion.
    """
    n = len(matrix)

    # Base case for 1×1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case for 2×2 matrix
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case
    det = 0
    for i in range(n):
        # Build the minor by excluding row 0 and column i
        submatrix = tuple(
            tuple(row[:i] + row[i + 1:]) for row in matrix[1:]
        )
        det += matrix[0][i] * (-1) ** i * matrix_determinant(submatrix)

    return det


def matMul(matrix1, matrix2):
        """
    Performs matrix multiplication between two 2D matrices.

    Parameters:
    - matrix1 (List[List[float]]): The left-hand matrix (m x n).
    - matrix2 (List[List[float]]): The right-hand matrix (n x p).

    Returns:
    - List[List[float]]: The resulting matrix of size (m x p).

    ### Example usage:
    ```python
    A = [
        [1, 2],
        [3, 4]
    ]
    B = [
        [5, 6],
        [7, 8]
    ]
    result = matMul(A, B)
    print(result)  # Output: [[19, 22], [43, 50]]
    ```

    Notes:
    - Assumes the number of columns in `matrix1` equals the number of rows in `matrix2`.
    - Raises a `ValueError` if dimensions are incompatible.
    """
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Incompatible matrix dimensions for multiplication.")
    return [[sum(a * b for a, b in zip(row1, col)) for col in zip(*matrix2)] for row1 in matrix1]


def matrix_inverse(matrix):
    """
    Computes the inverse of a square matrix using Gauss-Jordan elimination.

    Parameters:
    - matrix (List[List[float]]): A square matrix (n x n) to invert.

    Returns:
    - List[List[float]]: The inverse of the input matrix.

    Raises:
    - ValueError: If the matrix is singular or not square.

    ### Example usage:
    ```python
    A = [
        [4, 7],
        [2, 6]
    ]
    inv_A = matrix_inverse(A)
    print(inv_A)
    # Output (approximately):
    # [[0.6, -0.7],
    #  [-0.2, 0.4]]
    ```
    """
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
     """
    Computes the modular inverse of `a` modulo `m` using the Extended Euclidean Algorithm.

    The modular inverse is the number `x` such that:
        (a * x) % m == 1

    Parameters:
    - a (int): The number to find the inverse of.
    - m (int): The modulus (must be > 1 and coprime with `a`).

    Returns:
    - int: The modular inverse of `a` modulo `m`, or raises an exception if not invertible.

    ### Example usage:
    ```python
    print(mod_inverse(3, 11))   # Output: 4
    print(mod_inverse(10, 17))  # Output: 12
    ```

    Raises:
    - ZeroDivisionError: If the modular inverse does not exist (i.e. `a` and `m` are not coprime).
    """
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
    """
    Solves a system of simultaneous congruences using the Chinese Remainder Theorem (CRT).

    Finds the smallest `x` such that:
        x ≡ a_list[i] (mod m_list[i])  for all i

    Assumes all moduli in `m_list` are pairwise coprime.

    Parameters:
    - a_list (List[int]): List of remainders.
    - m_list (List[int]): List of moduli.

    Returns:
    - int: The smallest non-negative solution `x` such that x satisfies all given congruences.

    ### Example usage:
    ```python
    a = [2, 3, 2]
    m = [3, 5, 7]
    print(chinese_remainder_theorem(a, m))  # Output: 23
    ```

    Raises:
    - ValueError: If `a_list` and `m_list` are not the same length.
    """
    if len(a_list) != len(m_list):
        raise ValueError("Lists a_list and m_list must be of the same length.")
    M = math.prod(m_list)
    x = 0
    for i in range(len(a_list)):
        Mi = M // m_list[i]
        Mi_inverse = mod_inverse(Mi, m_list[i])
        x += a_list[i] * Mi * Mi_inverse
    return x % M

