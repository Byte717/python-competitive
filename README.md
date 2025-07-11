# pyAlgorithm

**pyAlgorithm** is a lightweight and growing Python library that provides a clean, well-documented collection of data structures and algorithms commonly used in computer science, competitive programming, and technical interviews.

---

## ✅ Features

This library includes thoroughly tested and modular implementations of:

### 📦 Data Structures
- Arrays
- Linked Lists
- Stack / Queue / Deque
- Segment Tree
- Binary Search Tree
- AVL Tree
- Red-Black Tree
- Suffix Tree / Suffix Array
- Trie
- Graphs (Adjacency List)
- Disjoint Set Union (DSU)
- Skip List
- Binary Indexed Tree (Fenwick Tree)
- BitSet
- Heap / Priority Queue
- Hash Table

### 📐 Algorithms & Math
- GCD / LCM
- Prime Factorization
- Sieve of Eratosthenes
- Modular Arithmetic (Inverse, Power)
- Binomial Coefficient (mod p)
- Chinese Remainder Theorem
- Matrix operations (Multiply, Inverse, Determinant)
- Quicksort
- Binary Search
- isSorted

---

## 📘 Usage

Every class and function includes **Markdown-style docstrings** for hover help and in-editor documentation. No need to navigate to separate files.

### Example:

```python
from pyAlgorithm import SegmentTree, mod_inverse

tree = SegmentTree(8)
tree.set(3, 10)
tree.set(4, 7)
print(tree.query(3, 4))  # Output: 17

print(mod_inverse(3, 11))  # Output: 4
```
---

## 🚀 Coming Soon

- Graph algorithms (Dijkstra, Floyd-Warshall, Kruskal)
- Network flow (Ford-Fulkerson, Edmonds-Karp)
- Visualizers for trees and graphs
- PyPI package release and playground CLI

---

## 📂 Project Structure

```
pyAlgorithm/
├── src/
│   ├── __init__.py       # Exposes everything via pyAlgorithm
│   ├── DataStructs.py    # All data structures
│   └── Funcs.py          # All utility functions and algorithms
├── README.md
```

---

## 🤝 Contributing

Pull requests are welcome! Add new algorithms, clean up code, or expand documentation. Star the repo if you find it helpful!

---

## 📜 License

MIT License — free to use, modify, and distribute.
