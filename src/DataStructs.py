from functools import lru_cache
from collections import deque
import random
import math


class Array:
    def __init__(self, size):
        self.size = size
        self.array = [None] * size

    def get(self, index):
        return self.array[index]

    def set(self, index, value):
        self.array[index] = value

    def size(self):
        return self.size


class ListNode:
    def __init__(self, value):
        self.value = value
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def insert(self, value):
        new_node = ListNode(value)
        new_node.next = self.head
        self.head = new_node

    def search(self, value):
        current = self.head
        while current:
            if current.value == value:
                return True
            current = current.next
        return False

    def delete(self, value):
        current = self.head
        previous = None
        while current:
            if current.value == value:
                if previous:
                    previous.next = current.next
                else:
                    self.head = current.next
                return
            previous = current
            current = current.next


class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def peek(self):
        if not self.is_empty():
            return self.items[-1]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)


class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)

    def peek(self):
        if not self.is_empty():
            return self.items[0]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)


class Deque:
    def __init__(self):
        self.deque = deque()

    def append(self, item):
        self.deque.append(item)

    def appendleft(self, item):
        self.deque.appendleft(item)

    def pop(self):
        return self.deque.pop() if not self.is_empty() else None

    def popleft(self):
        return self.deque.popleft() if not self.is_empty() else None

    def peek(self):
        return self.deque[-1] if not self.is_empty() else None

    def peekleft(self):
        return self.deque[0] if not self.is_empty() else None

    def is_empty(self):
        return len(self.deque) == 0

    def size(self):
        return len(self.deque)


class priority_queue:
    def __init__(self, initial=None, max: bool = False) -> None:
        self.q = initial if initial is not None else []
        self.max = max

    def top(self):
        return self.q[0]

    def push(self, x) -> None:
        lo, hi = 0, len(self.q)
        while lo < hi:
            hi = (lo + hi) // 2 if self.cmp(self.q[(lo + hi) // 2], x, eq=True) else hi
            lo = ((lo + hi) // 2) + 1 if not self.cmp(self.q[(lo + hi) // 2], x) else lo
        lo += 1 if lo < len(self.q) and self.q[lo] < x else 0
        self.q.insert(lo - (1 if self.max else 0), x)

    def cmp(self, x, y, eq=False):
        return (x <= y if eq else x < y) if self.max else (x >= y if eq else x > y)

    def pop(self):
        return self.q.pop(0)

    def size(self):
        return len(self.q)

    def __str__(self) -> str:
        return str(self.q)


class HashTable:
    def __init__(self):
        self.size = 10
        self.table = [[] for _ in range(self.size)]

    def _hash(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        hash_key = self._hash(key)
        for pair in self.table[hash_key]:
            if pair[0] == key:
                pair[1] = value
                return
        self.table[hash_key].append([key, value])

    def get(self, key):
        hash_key = self._hash(key)
        for pair in self.table[hash_key]:
            if pair[0] == key:
                return pair[1]
        raise KeyError(f'Key {key} not found')

    def delete(self, key):
        hash_key = self._hash(key)
        for i, pair in enumerate(self.table[hash_key]):
            if pair[0] == key:
                del self.table[hash_key][i]
                return
        raise KeyError(f'Key {key} not found')


class TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        self.root = self._insert(self.root, key)

    def _insert(self, node, key):
        if node is None:
            return TreeNode(key)
        if key < node.key:
            node.left = self._insert(node.left, key)
        else:
            node.right = self._insert(node.right, key)
        return node

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, node, key):
        if node is None or node.key == key:
            return node
        if key < node.key:
            return self._search(node.left, key)
        else:
            return self._search(node.right, key)

    def delete(self, key):
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        if node is None:
            return node
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            else:
                successor = self._find_min(node.right)
                node.key = successor.key
                node.right = self._delete(node.right, successor.key)
        return node

    def _find_min(self, node):
        current = node
        while current.left:
            current = current.left
        return current


class AVLTreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1


class AVLTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        self.root = self._insert(self.root, key)

    def _insert(self, node, key):
        if node is None:
            return AVLTreeNode(key)
        if key < node.key:
            node.left = self._insert(node.left, key)
        else:
            node.right = self._insert(node.right, key)

        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        balance = self._get_balance(node)

        if balance > 1 and key < node.left.key:
            return self._rotate_right(node)
        if balance < -1 and key > node.right.key:
            return self._rotate_left(node)
        if balance > 1 and key > node.left.key:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        if balance < -1 and key < node.right.key:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    def _get_height(self, node):
        if not node:
            return 0
        return node.height

    def _get_balance(self, node):
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _rotate_left(self, z):
        y = z.right
        T2 = y.left

        y.left = z
        z.right = T2

        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))

        return y

    def _rotate_right(self, z):
        y = z.left
        T3 = y.right

        y.right = z
        z.left = T3

        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))

        return y

    def delete(self, key):
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        if node is None:
            return node
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            else:
                successor = self._find_min(node.right)
                node.key = successor.key
                node.right = self._delete(node.right, successor.key)

        if node is None:
            return node

        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        balance = self._get_balance(node)

        if balance > 1 and self._get_balance(node.left) >= 0:
            return self._rotate_right(node)
        if balance < -1 and self._get_balance(node.right) <= 0:
            return self._rotate_left(node)
        if balance > 1 and self._get_balance(node.left) < 0:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        if balance < -1 and self._get_balance(node.right) > 0:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    def _find_min(self, node):
        current = node
        while current.left:
            current = current.left
        return current


class RedBlackTreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None
        self.color = "RED"


class RedBlackTree:
    def __init__(self):
        self.nil = RedBlackTreeNode(None)
        self.nil.color = "BLACK"
        self.root = self.nil

    def insert(self, key):
        new_node = RedBlackTreeNode(key)
        new_node.left = self.nil
        new_node.right = self.nil

        parent = None
        current = self.root

        while current != self.nil:
            parent = current
            if new_node.key < current.key:
                current = current.left
            else:
                current = current.right

        new_node.parent = parent

        if parent == None:
            self.root = new_node
        elif new_node.key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node

        new_node.color = "RED"
        self._insert_fixup(new_node)

    def _insert_fixup(self, node):
        while node.parent.color == "RED":
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle.color == "RED":
                    node.parent.color = "BLACK"
                    uncle.color = "BLACK"
                    node.parent.parent.color = "RED"
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        node = node.parent
                        self._left_rotate(node)
                    node.parent.color = "BLACK"
                    node.parent.parent.color = "RED"
                    self._right_rotate(node.parent.parent)
            else:
                uncle = node.parent.parent.left
                if uncle.color == "RED":
                    node.parent.color = "BLACK"
                    uncle.color = "BLACK"
                    node.parent.parent.color = "RED"
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self._right_rotate(node)
                    node.parent.color = "BLACK"
                    node.parent.parent.color = "RED"
                    self._left_rotate(node.parent.parent)

        self.root.color = "BLACK"

    def _left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.nil:
            y.left.parent = x
        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.nil:
            y.right.parent = x
        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def delete(self, key):
        node_to_delete = self._search(self.root, key)
        if node_to_delete == None:
            return

        if node_to_delete.left == self.nil or node_to_delete.right == self.nil:
            y = node_to_delete
        else:
            y = self._find_successor(node_to_delete)

        if y.left != self.nil:
            x = y.left
        else:
            x = y.right

        x.parent = y.parent

        if y.parent == None:
            self.root = x
        elif y == y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x

        if y != node_to_delete:
            node_to_delete.key = y.key

        if y.color == "BLACK":
            self._delete_fixup(x)

    def _delete_fixup(self, node):
        while node != self.root and node.color == "BLACK":
            if node == node.parent.left:
                sibling = node.parent.right
                if sibling.color == "RED":
                    sibling.color = "BLACK"
                    node.parent.color = "RED"
                    self._left_rotate(node.parent)
                    sibling = node.parent.right
                if sibling.left.color == "BLACK" and sibling.right.color == "BLACK":
                    sibling.color = "RED"
                    node = node.parent
                else:
                    if sibling.right.color == "BLACK":
                        sibling.left.color = "BLACK"
                        sibling.color = "RED"
                        self._right_rotate(sibling)
                        sibling = node.parent.right
                    sibling.color = node.parent.color
                    node.parent.color = "BLACK"
                    sibling.right.color = "BLACK"
                    self._left_rotate(node.parent)
                    node = self.root
            else:
                sibling = node.parent.left
                if sibling.color == "RED":
                    sibling.color = "BLACK"
                    node.parent.color = "RED"
                    self._right_rotate(node.parent)
                    sibling = node.parent.left
                if sibling.right.color == "BLACK" and sibling.left.color == "BLACK":
                    sibling.color = "RED"
                    node = node.parent
                else:
                    if sibling.left.color == "BLACK":
                        sibling.right.color = "BLACK"
                        sibling.color = "RED"
                        self._left_rotate(sibling)
                        sibling = node.parent.left
                    sibling.color = node.parent.color
                    node.parent.color = "BLACK"
                    sibling.left.color = "BLACK"
                    self._right_rotate(node.parent)
                    node = self.root

        node.color = "BLACK"

    def _search(self, node, key):
        if node == self.nil or key == node.key:
            return node
        if key < node.key:
            return self._search(node.left, key)
        else:
            return self._search(node.right, key)

    def _find_successor(self, node):
        if node.right != self.nil:
            return self._find_min(node.right)
        parent = node.parent
        while parent != self.nil and node == parent.right:
            node = parent
            parent = parent.parent
        return parent

    def _find_min(self, node):
        while node.left != self.nil:
            node = node.left
        return node


class Graph:
    def __init__(self):
        self.adj_list = {}

    def add_vertex(self, v):
        if v not in self.adj_list:
            self.adj_list[v] = []

    def add_edge(self, v1, v2):
        if v1 in self.adj_list and v2 in self.adj_list:
            self.adj_list[v1].append(v2)
            self.adj_list[v2].append(v1)

    def remove_edge(self, v1, v2):
        if v1 in self.adj_list and v2 in self.adj_list:
            self.adj_list[v1].remove(v2)
            self.adj_list[v2].remove(v1)

    def remove_vertex(self, v):
        if v in self.adj_list:
            for vertex in self.adj_list[v]:
                self.adj_list[vertex].remove(v)
            del self.adj_list[v]

    def get_adjacency_list(self):
        return self.adj_list


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_end_of_word = True

    def search(self, word):
        current = self.root
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]
        return current.is_end_of_word

    def starts_with(self, prefix):
        current = self.root
        for char in prefix:
            if char not in current.children:
                return False
            current = current.children[char]
        return True


class Heap:
    def __init__(self, initial=None, key=lambda x: x):
        self.key = key
        if initial:
            self.heap = [(key(item), item) for item in initial]
            self._heapify()
        else:
            self.heap = []

    def _heapify(self):
        n = len(self.heap)
        for i in range(n // 2 - 1, -1, -1):
            self._heapify_down(i)

    def push(self, item):
        self.heap.append((self.key(item), item))
        self._heapify_up(len(self.heap) - 1)

    def pop(self):
        if len(self.heap) > 1:
            self._swap(0, len(self.heap) - 1)
        item = self.heap.pop()[1]
        self._heapify_down(0)
        return item

    def peek(self):
        return self.heap[0][1] if self.heap else None

    def size(self):
        return len(self.heap)

    def _heapify_up(self, index):
        parent = (index - 1) // 2
        while index > 0 and self.heap[parent][0] > self.heap[index][0]:
            self._swap(parent, index)
            index = parent
            parent = (index - 1) // 2

    def _heapify_down(self, index):
        left = 2 * index + 1
        right = 2 * index + 2
        smallest = index

        if left < len(self.heap) and self.heap[left][0] < self.heap[smallest][0]:
            smallest = left
        if right < len(self.heap) and self.heap[right][0] < self.heap[smallest][0]:
            smallest = right

        if smallest != index:
            self._swap(index, smallest)
            self._heapify_down(smallest)

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

class SegmentTree:
    DEF = 0
    def __init__(self,n:int) -> None:
        self.n = n
        self.SegTree = [self.DEF*2*n]

    def set(self,idx: int,val: int) -> None:
        idx += self.n
        self.SegTree[idx] = val
        while idx > 1:
            self.SegTree[idx//2] = self.SegTree[idx] + self.SegTree[idx^1]
            idx //=2

    def query(self,start:int,end:int) -> int:
        ret:int = self.DEF
        start += self.n; end += self.n
        while start <= end:
            if start % 2 == 1:
                ret += self.SegTree[start]
                start += 1
            if end % 2== 1:
                end -= 1
                ret += self.SegTree[end]

            start //= 2
            end //= 2
        return ret

    @staticmethod
    def LOG2(n:int):
        log:int = 0; while (1 << (log+1) <= n): log+=1; return log

class BIT:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)

    def add(self, index, value):
        while index <= self.size:
            self.tree[index] += value
            index += index & -index

    def sum(self, index):
        sum = 0
        while index > 0:
            sum += self.tree[index]
            index -= index & -index
        return sum

    def range_sum(self, left, right):
        return self.sum(right) - self.sum(left - 1)


class DSU:
    parent:list = []; size : list = []
    def __init__(self,n:int) -> None:
        self.parent = [i for i in range(n)]
        self.size = [1*n]

    def get(self, x:int) -> int:
        while x != self.parent[x]: x = self.parent[x]; return x

    def same_set(self, a: int,b: int) -> bool: return self.get(a) == self.get(b)

    def size(self,x:int) -> int: return self.size[x]

    def link(self,a:int,b:int) -> bool:
        a = self.get(a); b = self.get(b)
        if(a == b):return False
        if(self.size[a] < self.size[b]):
            a, b = b, a
        self.size[a] += self.size[b]
        self.parent[b] = a
        return True

class SuffixArray:
    def __init__(self, text):
        self.text = text
        self.suffix_array = self._build_suffix_array(text)

    def _build_suffix_array(self, text):
        suffixes = [(text[i:], i) for i in range(len(text))]
        suffixes.sort(key=lambda x: x[0])
        return [suffix[1] for suffix in suffixes]

    def get_suffix_array(self):
        return self.suffix_array


class SuffixTreeNode:
    def __init__(self):
        self.children = {}
        self.suffix_link = None
        self.start = None
        self.end = None


class SuffixTree:
    def __init__(self, text):
        self.text = text
        self.root = SuffixTreeNode()
        self.suffix_link = None
        self._build_suffix_tree()

    def _build_suffix_tree(self):
        n = len(self.text)
        for i in range(n):
            self._extend_suffix(i)

    def _extend_suffix(self, i):
        current = self.root
        j = i
        while j < len(self.text):
            if self.text[j] not in current.children:
                current.children[self.text[j]] = SuffixTreeNode()
                current.children[self.text[j]].start = j
            current = current.children[self.text[j]]
            j += 1
        current.end = i

    def search(self, pattern):
        current = self.root
        for char in pattern:
            if char not in current.children:
                return False
            current = current.children[char]
        return True


class SkipListNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.forward = []


class SkipList:
    def __init__(self):
        self.head = SkipListNode(float('-inf'), None)
        self.max_level = 1

    def _random_level(self):
        level = 1
        while random.random() < 0.5 and level < self.max_level + 1:
            level += 1
        return level

    def insert(self, key, value):
        update = [None] * (self.max_level + 1)
        current = self.head

        for i in range(self.max_level, 0, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        level = self._random_level()
        if level > self.max_level:
            for i in range(self.max_level + 1, level + 1):
                update.append(self.head)
            self.max_level = level

        new_node = SkipListNode(key, value)
        for i in range(1, level + 1):
            new_node.forward.append(update[i].forward[i])
            update[i].forward[i] = new_node

    def search(self, key):
        current = self.head
        for i in range(self.max_level, 0, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        current = current.forward[1]
        if current and current.key == key:
            return current.value
        return None

    def delete(self, key):
        update = [None] * (self.max_level + 1)
        current = self.head

        for i in range(self.max_level, 0, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[1]
        if current and current.key == key:
            for i in range(1, self.max_level + 1):
                if update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i]
            while self.max_level > 1 and self.head.forward[self.max_level] is None:
                self.max_level -= 1


class BitSet:
    def __init__(self, size):
        self.size = size
        self.arr = [0] * ((size + 31) // 32)

    def set(self, pos):
        if pos < 0 or pos >= self.size:
            raise IndexError("Index out of range")
        word_index = pos // 32
        bit_index = pos % 32
        self.arr[word_index] |= (1 << bit_index)

    def reset(self, pos):
        if pos < 0 or pos >= self.size:
            raise IndexError("Index out of range")
        word_index = pos // 32
        bit_index = pos % 32
        self.arr[word_index] &= ~(1 << bit_index)

    def test(self, pos):
        if pos < 0 or pos >= self.size:
            raise IndexError("Index out of range")
        word_index = pos // 32
        bit_index = pos % 32
        return (self.arr[word_index] & (1 << bit_index)) != 0

    def flip(self, pos):
        if pos < 0 or pos >= self.size:
            raise IndexError("Index out of range")
        word_index = pos // 32
        bit_index = pos % 32
        self.arr[word_index] ^= (1 << bit_index)

    def any(self):
        for word in self.arr:
            if word != 0:
                return True
        return False

    def none(self):
        for word in self.arr:
            if word != 0:
                return False
        return True

    def all(self):
        for word in self.arr:
            if word != 0xFFFFFFFF:
                return False
        return True

    def count(self):
        count = 0
        for word in self.arr:
            count += bin(word).count('1')
        return count

    def size(self):
        return self.size

    def __getitem__(self, pos):
        return self.test(pos)

    def __setitem__(self, pos, value):
        if value:
            self.set(pos)
        else:
            self.reset(pos)

    def __repr__(self):
        return ''.join(['1' if self.test(i) else '0' for i in range(self.size)])


class SieveOfEratosthenes:
    def __init__(self, n):
        self.n = n
        self.is_prime = [True] * (n + 1)
        self.primes = self._generate_primes()

    def _generate_primes(self):
        p = 2
        while (p * p <= self.n):
            if (self.is_prime[p] == True):
                for i in range(p * p, self.n + 1, p):
                    self.is_prime[i] = False
            p += 1
        return [p for p in range(2, self.n + 1) if self.is_prime[p]]

    def setN(self, newN):
        self.n = newN
        self.is_prime = [True] * (newN + 1)
        self.primes = self._generate_primes()

    def get_primes(self):
        return self.primes

    def getPrimeArr(self):
        return self.is_prime



class SparseTableRMQ:
    def __init__(self, array):
        self.n = len(array)
        self.k = math.floor(math.log2(self.n)) + 1
        self.st = [[0] * self.k for _ in range(self.n)]
        for i in range(self.n): self.st[i][0] = array[i]

        j = 1
        while (1 << j) <= self.n:
            i = 0
            while (i + (1 << j) - 1) < self.n:
                self.st[i][j] = min(self.st[i][j - 1], self.st[i + (1 << (j - 1))][j - 1])
                i += 1
            j += 1

    def query(self, l, r):
        j = math.floor(math.log2(r - l + 1))
        return min(self.st[l][j], self.st[r - (1 << j) + 1][j])


