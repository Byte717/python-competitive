from functools import lru_cache
from collections import deque
import random
import math
import heapq


class Array:
    """
    A fixed-size array implementation that provides index-based access and assignment.

    Internally uses a Python list of fixed length initialized with `None`.

    ### Example usage:
    ```python
    arr = Array(5)
    arr.set(0, "a")
    arr.set(1, "b")
    print(arr.get(1))       # Output: b
    print(arr.length())     # Output: 5
    ```

    Attributes:
    - size (int): The fixed size of the array.
    - array (List[Any]): The underlying list storing the array elements.
    """

    def __init__(self, size):
        """
        Initializes a new fixed-size array.

        Parameters:
        - size (int): The number of elements the array can hold.
        """
        self.sz = size
        self.array = [None] * size

    def get(self, index):
        """
        Retrieves the value at the specified index.

        Parameters:
        - index (int): The index of the element to retrieve.

        Returns:
        - Any: The value at the specified index.

        Raises:
        - IndexError: If the index is out of bounds.
        """
        return self.array[index]

    def set(self, index, value):
        """
        Sets the value at the specified index.

        Parameters:
        - index (int): The index to assign the value to.
        - value (Any): The value to store.

        Raises:
        - IndexError: If the index is out of bounds.
        """
        self.array[index] = value

    def size(self):
        """
        Returns the fixed size of the array.

        Returns:
        - int: The number of elements the array can hold.
        """
        return self.sz

class ListNode:
    """
    A node in a singly linked list.

    Attributes:
    - value (Any): The value stored in the node.
    - next (ListNode): The reference to the next node in the list.
    """

    def __init__(self, value):
        """
        Initializes a list node with a value.

        Parameters:
        - value (Any): The value to store in the node.
        """
        self.value = value
        self.next = None


class LinkedList:
    """
    A simple singly linked list implementation.

    Supports insertion at the head, search, and deletion by value.

    ### Example usage:
    ```python
    ll = LinkedList()
    ll.insert(3)
    ll.insert(2)
    ll.insert(1)
    print(ll.search(2))  # True
    ll.delete(2)
    print(ll.search(2))  # False
    ```

    Attributes:
    - head (ListNode): The head (first node) of the linked list.
    """

    def __init__(self):
        """
        Initializes an empty linked list.
        """
        self.head = None

    def insert(self, value):
        """
        Inserts a new node with the given value at the head of the list.

        Parameters:
        - value (Any): The value to insert.
        """
        new_node = ListNode(value)
        new_node.next = self.head
        self.head = new_node

    def search(self, value):
        """
        Searches for a value in the linked list.

        Parameters:
        - value (Any): The value to search for.

        Returns:
        - bool: True if the value is found, False otherwise.
        """
        current = self.head
        while current:
            if current.value == value:
                return True
            current = current.next
        return False

    def delete(self, value):
        """
        Deletes the first node containing the given value.

        Parameters:
        - value (Any): The value to delete.

        Notes:
        - If the value does not exist, nothing happens.
        """
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
    """
    A simple LIFO (Last-In-First-Out) stack implementation using a Python list.

    Supports push, pop, peek, size, and empty check operations.

    ### Example usage:
    ```python
    s = Stack()
    s.push(10)
    s.push(20)
    print(s.peek())     # Output: 20
    print(s.pop())      # Output: 20
    print(s.size())     # Output: 1
    ```

    Attributes:
    - items (List[Any]): Internal list storing stack elements.
    """

    def __init__(self):
        """
        Initializes an empty stack.
        """
        self.items = []

    def push(self, item):
        """
        Pushes an item onto the top of the stack.

        Parameters:
        - item (Any): The item to add.
        """
        self.items.append(item)

    def pop(self):
        """
        Removes and returns the item at the top of the stack.

        Returns:
        - Any: The top item, or None if the stack is empty.
        """
        if not self.is_empty():
            return self.items.pop()

    def peek(self):
        """
        Returns the item at the top of the stack without removing it.

        Returns:
        - Any: The top item, or None if the stack is empty.
        """
        if not self.is_empty():
            return self.items[-1]

    def is_empty(self):
        """
        Checks whether the stack is empty.

        Returns:
        - bool: True if the stack is empty, False otherwise.
        """
        return len(self.items) == 0

    def size(self):
        """
        Returns the number of items in the stack.

        Returns:
        - int: The current size of the stack.
        """
        return len(self.items)

class Queue:
    """
    A simple FIFO (First-In-First-Out) queue implementation using a Python list.

    Enqueue operations add items to the end, and dequeue operations remove items from the front.

    ### Example usage:
    ```python
    q = Queue()
    q.enqueue(1)
    q.enqueue(2)
    print(q.peek())     # Output: 1
    print(q.dequeue())  # Output: 1
    print(q.size())     # Output: 1
    ```

    Attributes:
    - items (List[Any]): The list holding the queue's items.
    """

    def __init__(self):
        """
        Initializes an empty queue.
        """
        self.items = []

    def enqueue(self, item):
        """
        Adds an item to the end of the queue.

        Parameters:
        - item (Any): The item to add.
        """
        self.items.append(item)

    def dequeue(self):
        """
        Removes and returns the item at the front of the queue.

        Returns:
        - Any: The item at the front of the queue, or None if the queue is empty.
        """
        if not self.is_empty():
            return self.items.pop(0)

    def peek(self):
        """
        Returns the item at the front of the queue without removing it.

        Returns:
        - Any: The item at the front, or None if the queue is empty.
        """
        if not self.is_empty():
            return self.items[0]

    def is_empty(self):
        """
        Checks if the queue is empty.

        Returns:
        - bool: True if the queue has no items, False otherwise.
        """
        return len(self.items) == 0

    def size(self):
        """
        Returns the number of items in the queue.

        Returns:
        - int: The size of the queue.
        """
        return len(self.items)


class Deque:
    """
    A double-ended queue (deque) wrapper using Python's `collections.deque`.

    Supports appending and popping from both ends in O(1) time.

    ### Example usage:
    ```python
    dq = Deque()
    dq.append(10)
    dq.appendleft(5)
    print(dq.peek())      # Output: 10
    print(dq.peekleft())  # Output: 5
    dq.pop()              # Removes 10
    dq.popleft()          # Removes 5
    ```

    Attributes:
    - deque (collections.deque): The underlying deque object.
    """

    def __init__(self):
        """
        Initializes an empty deque.
        """
        self.deque = deque()

    def append(self, item):
        """
        Appends an item to the right end.

        Parameters:
        - item (Any): The item to add.
        """
        self.deque.append(item)

    def appendleft(self, item):
        """
        Appends an item to the left end.

        Parameters:
        - item (Any): The item to add.
        """
        self.deque.appendleft(item)

    def pop(self):
        """
        Removes and returns the item from the right end.

        Returns:
        - Any: The removed item, or None if the deque is empty.
        """
        return self.deque.pop() if not self.is_empty() else None

    def popleft(self):
        """
        Removes and returns the item from the left end.

        Returns:
        - Any: The removed item, or None if the deque is empty.
        """
        return self.deque.popleft() if not self.is_empty() else None

    def peek(self):
        """
        Returns the item at the right end without removing it.

        Returns:
        - Any: The item at the right, or None if empty.
        """
        return self.deque[-1] if not self.is_empty() else None

    def peekleft(self):
        """
        Returns the item at the left end without removing it.

        Returns:
        - Any: The item at the left, or None if empty.
        """
        return self.deque[0] if not self.is_empty() else None

    def is_empty(self):
        """
        Checks whether the deque is empty.

        Returns:
        - bool: True if empty, False otherwise.
        """
        return len(self.deque) == 0

    def size(self):
        """
        Returns the number of elements in the deque.

        Returns:
        - int: The size of the deque.
        """
        return len(self.deque)

'''
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
'''

class PriorityQueue:
    """
    A priority queue implementation using Python's `heapq` module.

    Supports both min-heap and max-heap behavior with efficient O(log n) push/pop operations.

    ### Example usage:
    ```python
    pq = PriorityQueue(max=False)  # Min-heap
    pq.push(4)
    pq.push(2)
    pq.push(7)
    print(pq.top())     # Output: 2
    print(pq.pop())     # Output: 2
    print(pq.size())    # Output: 2

    max_pq = PriorityQueue(max=True)
    max_pq.push(4)
    max_pq.push(2)
    max_pq.push(7)
    print(max_pq.pop()) # Output: 7
    ```

    Attributes:
    - max (bool): Determines whether the priority queue acts as a max-heap or min-heap.
    - heap (List[Tuple]): The internal heap list.
    """

    def __init__(self, initial=None, max: bool = False):
        """
        Initializes the priority queue.

        Parameters:
        - initial (List[Any], optional): Optional list of initial values.
        - max (bool): Whether the queue should behave as a max-heap (default: False).
        """
        self.max = max
        if initial is None:
            self.heap = []
        else:
            self.heap = [self._wrap(item) for item in initial]
            heapq.heapify(self.heap)

    def _wrap(self, item):
        """
        Wraps the item for max-heap behavior if needed.

        Parameters:
        - item (Any): The value to wrap.

        Returns:
        - Any or tuple: The item itself or negated value for max-heap.
        """
        return -item if self.max else item

    def _unwrap(self, item):
        """
        Unwraps the internal representation to the original value.

        Parameters:
        - item (Any): The internal heap representation.

        Returns:
        - Any: The original value.
        """
        return -item if self.max else item

    def push(self, item):
        """
        Adds an item to the priority queue.

        Parameters:
        - item (Any): The item to insert.
        """
        heapq.heappush(self.heap, self._wrap(item))

    def pop(self):
        """
        Removes and returns the item with the highest priority.

        Returns:
        - Any: The item with the highest or lowest value depending on heap type.
        """
        return self._unwrap(heapq.heappop(self.heap))

    def top(self):
        """
        Returns the item with the highest priority without removing it.

        Returns:
        - Any: The item at the top of the priority queue.
        """
        return self._unwrap(self.heap[0]) if self.heap else None

    def size(self):
        """
        Returns the number of elements in the priority queue.

        Returns:
        - int: The size of the queue.
        """
        return len(self.heap)

    def __str__(self):
        """
        Returns a string representation of the priority queue.

        Returns:
        - str: The string form of the queue's elements in heap order.
        """
        return str([self._unwrap(item) for item in self.heap])


class HashTable:
    """
    A basic hash table implementation using separate chaining for collision resolution.

    Keys are hashed into one of `size` buckets. Each bucket is a list of key-value pairs.

    ### Example usage:
    ```python
    ht = HashTable()
    ht.insert("name", "Alice")
    print(ht.get("name"))   # Output: Alice
    ht.insert("name", "Bob")
    print(ht.get("name"))   # Output: Bob
    ht.delete("name")
    # ht.get("name") would raise KeyError
    ```

    Attributes:
    - size (int): Number of buckets in the hash table.
    - table (List[List[List[Any]]]): The list of buckets, each containing key-value pairs.
    """

    def __init__(self):
        """
        Initializes the hash table with a default of 10 buckets.
        """
        self.size = 10
        self.table = [[] for _ in range(self.size)]

    def _hash(self, key):
        """
        Computes the hash index for a given key.

        Parameters:
        - key (Any): The key to hash.

        Returns:
        - int: The bucket index corresponding to the key.
        """
        return hash(key) % self.size

    def insert(self, key, value):
        """
        Inserts a key-value pair into the hash table. Updates the value if key already exists.

        Parameters:
        - key (Any): The key to insert.
        - value (Any): The value to associate with the key.
        """
        hash_key = self._hash(key)
        for pair in self.table[hash_key]:
            if pair[0] == key:
                pair[1] = value
                return
        self.table[hash_key].append([key, value])

    def get(self, key):
        """
        Retrieves the value associated with the given key.

        Parameters:
        - key (Any): The key to look up.

        Returns:
        - Any: The associated value.

        Raises:
        - KeyError: If the key is not found in the table.
        """
        hash_key = self._hash(key)
        for pair in self.table[hash_key]:
            if pair[0] == key:
                return pair[1]
        raise KeyError(f'Key {key} not found')

    def delete(self, key):
        """
        Removes a key-value pair from the hash table.

        Parameters:
        - key (Any): The key to delete.

        Raises:
        - KeyError: If the key is not found.
        """
        hash_key = self._hash(key)
        for i, pair in enumerate(self.table[hash_key]):
            if pair[0] == key:
                del self.table[hash_key][i]
                return
        raise KeyError(f'Key {key} not found')



class TreeNode:
    """
    A node in a Binary Search Tree.

    Attributes:
    - key (Any): The key stored in the node.
    - left (TreeNode): Reference to the left child.
    - right (TreeNode): Reference to the right child.
    """

    def __init__(self, key):
        """
        Initializes a TreeNode with a given key.

        Parameters:
        - key (Any): The key to store in the node.
        """
        self.key = key
        self.left = None
        self.right = None


class BinarySearchTree:
    """
    A standard Binary Search Tree (BST) implementation.

    Supports:
    - Insertion
    - Deletion
    - Search

    Each operation runs in O(h) time, where h is the height of the tree.

    ### Example usage:
    ```python
    bst = BinarySearchTree()
    bst.insert(10)
    bst.insert(5)
    bst.insert(15)

    node = bst.search(5)
    print(node.key if node else "Not found")  # Output: 5

    bst.delete(10)
    ```

    Attributes:
    - root (TreeNode): The root node of the tree.
    """

    def __init__(self):
        """
        Initializes an empty Binary Search Tree.
        """
        self.root = None

    def insert(self, key):
        """
        Inserts a key into the BST.

        Parameters:
        - key (Any): The key to insert.
        """
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
        """
        Searches for a key in the BST.

        Parameters:
        - key (Any): The key to search for.

        Returns:
        - TreeNode: The node containing the key, or None if not found.
        """
        return self._search(self.root, key)

    def _search(self, node, key):
        if node is None or node.key == key:
            return node
        if key < node.key:
            return self._search(node.left, key)
        else:
            return self._search(node.right, key)

    def delete(self, key):
        """
        Deletes a key from the BST, if it exists.

        Parameters:
        - key (Any): The key to delete.
        """
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        if node is None:
            return node
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            # Node with one or no child
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            # Node with two children
            successor = self._find_min(node.right)
            node.key = successor.key
            node.right = self._delete(node.right, successor.key)
        return node

    def _find_min(self, node):
        """
        Finds the node with the minimum key in the given subtree.

        Parameters:
        - node (TreeNode): The subtree root.

        Returns:
        - TreeNode: Node with the smallest key.
        """
        current = node
        while current.left:
            current = current.left
        return current

class AVLTreeNode:
    """
    A node in an AVL Tree.

    Attributes:
    - key (Any): The key or value stored in the node.
    - left (AVLTreeNode): Left child.
    - right (AVLTreeNode): Right child.
    - height (int): Height of the node for balancing purposes.
    """

    def __init__(self, key):
        """
        Initializes a new AVL tree node.

        Parameters:
        - key (Any): The key to be stored in the node.
        """
        self.key = key
        self.left = None
        self.right = None
        self.height = 1


class AVLTree:
    """
    AVL Tree implementation that supports balanced insertion and deletion.

    AVL Trees automatically maintain height-balance after insertions and deletions,
    ensuring O(log n) time complexity for search, insert, and delete.

    ### Example usage:
    ```python
    tree = AVLTree()
    tree.insert(10)
    tree.insert(20)
    tree.insert(5)
    tree.delete(10)
    ```

    Attributes:
    - root (AVLTreeNode): The root of the AVL tree.
    """

    def __init__(self):
        """
        Initializes an empty AVL Tree.
        """
        self.root = None

    def insert(self, key):
        """
        Inserts a key into the AVL Tree.

        Parameters:
        - key (Any): The key to insert.
        """
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

        # Left Left Case
        if balance > 1 and key < node.left.key:
            return self._rotate_right(node)

        # Right Right Case
        if balance < -1 and key > node.right.key:
            return self._rotate_left(node)

        # Left Right Case
        if balance > 1 and key > node.left.key:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)

        # Right Left Case
        if balance < -1 and key < node.right.key:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    def delete(self, key):
        """
        Deletes a key from the AVL Tree.

        Parameters:
        - key (Any): The key to delete.
        """
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

        # Balancing cases
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

    def _rotate_left(self, z):
        """
        Performs a left rotation on the given node.

        Parameters:
        - z (AVLTreeNode): The root of the subtree to rotate.

        Returns:
        - AVLTreeNode: New root after rotation.
        """
        y = z.right
        T2 = y.left

        y.left = z
        z.right = T2

        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))

        return y

    def _rotate_right(self, z):
        """
        Performs a right rotation on the given node.

        Parameters:
        - z (AVLTreeNode): The root of the subtree to rotate.

        Returns:
        - AVLTreeNode: New root after rotation.
        """
        y = z.left
        T3 = y.right

        y.right = z
        z.left = T3

        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))

        return y

    def _get_height(self, node):
        """
        Returns the height of the given node.

        Parameters:
        - node (AVLTreeNode): The node whose height to return.

        Returns:
        - int: The height, or 0 if node is None.
        """
        return node.height if node else 0

    def _get_balance(self, node):
        """
        Computes the balance factor of a node.

        Parameters:
        - node (AVLTreeNode): The node to check.

        Returns:
        - int: Balance factor (left height - right height).
        """
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _find_min(self, node):
        """
        Finds the node with the minimum key in the subtree rooted at `node`.

        Parameters:
        - node (AVLTreeNode): The root of the subtree.

        Returns:
        - AVLTreeNode: The node with the smallest key.
        """
        current = node
        while current.left:
            current = current.left
        return current


class RedBlackTreeNode:
    """
    A node in a Red-Black Tree.

    Attributes:
    - key (Any): The key or value stored in the node.
    - left (RedBlackTreeNode): Pointer to the left child.
    - right (RedBlackTreeNode): Pointer to the right child.
    - parent (RedBlackTreeNode): Pointer to the parent node.
    - color (str): "RED" or "BLACK" indicating the node's color.
    """

    def __init__(self, key):
        """
        Initializes a red node with the given key.
        """
        self.key = key
        self.left = None
        self.right = None
        self.parent = None
        self.color = "RED"


class RedBlackTree:
    """
    A Red-Black Tree implementation supporting insertions, deletions, and search operations.

    This self-balancing binary search tree guarantees O(log n) time complexity for all operations.

    ### Example usage:
    ```python
    tree = RedBlackTree()
    tree.insert(10)
    tree.insert(20)
    tree.insert(15)
    tree.delete(10)
    ```

    Attributes:
    - root (RedBlackTreeNode): The root node of the tree.
    - nil (RedBlackTreeNode): Sentinel node used to represent null leaves (colored black).
    """

    def __init__(self):
        """
        Initializes an empty Red-Black Tree with a sentinel `nil` node.
        """
        self.nil = RedBlackTreeNode(None)
        self.nil.color = "BLACK"
        self.root = self.nil

    def insert(self, key):
        """
        Inserts a key into the Red-Black Tree and rebalances it if necessary.

        Parameters:
        - key (Any): The key to insert.
        """
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

        if parent is None:
            self.root = new_node
        elif new_node.key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node

        new_node.color = "RED"
        self._insert_fixup(new_node)

    def _insert_fixup(self, node):
        """
        Fixes Red-Black Tree violations after insertion.
        """
        while node.parent and node.parent.color == "RED":
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
        """
        Performs a left rotation on node `x`.
        """
        y = x.right
        x.right = y.left
        if y.left != self.nil:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _right_rotate(self, x):
        """
        Performs a right rotation on node `x`.
        """
        y = x.left
        x.left = y.right
        if y.right != self.nil:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def delete(self, key):
        """
        Deletes a node with the given key from the tree, if it exists.

        Parameters:
        - key (Any): The key of the node to delete.
        """
        node_to_delete = self._search(self.root, key)
        if node_to_delete == self.nil:
            return

        if node_to_delete.left == self.nil or node_to_delete.right == self.nil:
            y = node_to_delete
        else:
            y = self._find_successor(node_to_delete)

        x = y.left if y.left != self.nil else y.right
        x.parent = y.parent

        if y.parent is None:
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
        """
        Fixes Red-Black Tree violations after deletion.
        """
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
        """
        Searches for a node with the given key in the subtree rooted at `node`.

        Parameters:
        - node (RedBlackTreeNode): Subtree root.
        - key (Any): The key to search for.

        Returns:
        - RedBlackTreeNode: The node with the given key, or `self.nil` if not found.
        """
        if node == self.nil or key == node.key:
            return node
        if key < node.key:
            return self._search(node.left, key)
        else:
            return self._search(node.right, key)

    def _find_successor(self, node):
        """
        Finds the in-order successor of the given node.

        Parameters:
        - node (RedBlackTreeNode): The node whose successor is to be found.

        Returns:
        - RedBlackTreeNode: The successor node.
        """
        if node.right != self.nil:
            return self._find_min(node.right)
        parent = node.parent
        while parent != self.nil and node == parent.right:
            node = parent
            parent = parent.parent
        return parent

    def _find_min(self, node):
        """
        Finds the node with the minimum key in the subtree.

        Parameters:
        - node (RedBlackTreeNode): Subtree root.

        Returns:
        - RedBlackTreeNode: The node with the minimum key.
        """
        while node.left != self.nil:
            node = node.left
        return node
class Graph:
    """
    An undirected graph implementation using an adjacency list.

    Supports basic operations such as adding/removing vertices and edges, and retrieving the adjacency list.

    ### Example usage:
    ```python
    g = Graph()
    g.add_vertex("A")
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    print(g.get_adjacency_list())  
    # Output: {'A': ['B', 'C'], 'B': ['A'], 'C': ['A']}

    g.remove_edge("A", "B")
    print(g.get_adjacency_list())
    # Output: {'A': ['C'], 'B': [], 'C': ['A']}

    g.remove_vertex("C")
    print(g.get_adjacency_list())
    # Output: {'A': [], 'B': []}
    ```

    Attributes:
    - adj_list (dict): A dictionary mapping each vertex to a list of its neighbors.
    """

    def __init__(self):
        """
        Initializes an empty graph.
        """
        self.adj_list = {}

    def add_vertex(self, v):
        """
        Adds a vertex to the graph if it doesn't already exist.

        Parameters:
        - v (Any): The vertex to add.
        """
        if v not in self.adj_list:
            self.adj_list[v] = []

    def add_edge(self, v1, v2):
        """
        Adds an undirected edge between vertices v1 and v2. 
        If either vertex does not exist, it is added automatically.

        Parameters:
        - v1 (Any): One end of the edge.
        - v2 (Any): The other end of the edge.
        """
        if v1 not in self.adj_list:
            self.add_vertex(v1)
        if v2 not in self.adj_list:
            self.add_vertex(v2)

        self.adj_list[v1].append(v2)
        self.adj_list[v2].append(v1)

    def remove_edge(self, v1, v2):
        """
        Removes the undirected edge between v1 and v2, if it exists.

        Parameters:
        - v1 (Any): One end of the edge.
        - v2 (Any): The other end of the edge.
        """
        if v1 in self.adj_list and v2 in self.adj_list:
            if v2 in self.adj_list[v1]:
                self.adj_list[v1].remove(v2)
            if v1 in self.adj_list[v2]:
                self.adj_list[v2].remove(v1)

    def remove_vertex(self, v):
        """
        Removes a vertex and all its connected edges from the graph.

        Parameters:
        - v (Any): The vertex to remove.
        """
        if v in self.adj_list:
            for neighbor in self.adj_list[v]:
                self.adj_list[neighbor].remove(v)
            del self.adj_list[v]

    def get_adjacency_list(self):
        """
        Returns the graph's adjacency list.

        Returns:
        - dict: The adjacency list representing the graph.
        """
        return self.adj_list


class TrieNode:
    """
    A node in the Trie structure.

    Attributes:
    - children (dict): Mapping from characters to child TrieNodes.
    - is_end_of_word (bool): True if the node marks the end of a complete word.
    """

    def __init__(self):
        """
        Initializes a TrieNode with empty children and end-of-word flag set to False.
        """
        self.children = {}
        self.is_end_of_word = False


class Trie:
    """
    A Trie (prefix tree) implementation for storing and querying strings efficiently.

    Supports:
    - Inserting words
    - Searching for complete words
    - Checking if any word starts with a given prefix

    ### Example usage:
    ```python
    trie = Trie()
    trie.insert("apple")
    print(trie.search("apple"))     # True
    print(trie.search("app"))       # False
    print(trie.starts_with("app"))  # True
    trie.insert("app")
    print(trie.search("app"))       # True
    ```

    Attributes:
    - root (TrieNode): The root node of the Trie.
    """

    def __init__(self):
        """
        Initializes an empty Trie with a root TrieNode.
        """
        self.root = TrieNode()

    def insert(self, word):
        """
        Inserts a word into the Trie.

        Parameters:
        - word (str): The word to insert.
        """
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_end_of_word = True

    def search(self, word):
        """
        Searches for a full word in the Trie.

        Parameters:
        - word (str): The word to search.

        Returns:
        - bool: True if the word exists in the Trie, False otherwise.
        """
        current = self.root
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]
        return current.is_end_of_word

    def starts_with(self, prefix):
        """
        Checks if any word in the Trie starts with the given prefix.

        Parameters:
        - prefix (str): The prefix to check.

        Returns:
        - bool: True if any word starts with the prefix, False otherwise.
        """
        current = self.root
        for char in prefix:
            if char not in current.children:
                return False
            current = current.children[char]
        return True


class Heap:
    """
    A customizable min-heap implementation using a binary heap.

    Internally stores tuples of (key, item) for comparison, allowing custom sorting behavior.

    ### Example usage:
    ```python
    h = Heap([4, 1, 3, 2])
    h.push(0)
    print(h.pop())    # Output: 0
    print(h.peek())   # Output: 1
    print(h.size())   # Output: 4

    # Custom key (max-heap behavior)
    max_heap = Heap([4, 1, 3, 2], key=lambda x: -x)
    print(max_heap.pop())  # Output: 4
    ```

    Attributes:
    - heap (List[Tuple[key, item]]): Internal list representing the heap.
    - key (Callable): A function that extracts the key for ordering.
    """

    def __init__(self, initial=None, key=lambda x: x):
        """
        Initializes the heap with an optional list and a custom key function.

        Parameters:
        - initial (List[Any], optional): Initial elements to heapify.
        - key (Callable): Function to extract comparison key from items.
        """
        self.key = key
        if initial:
            self.heap = [(key(item), item) for item in initial]
            self._heapify()
        else:
            self.heap = []

    def _heapify(self):
        """
        Converts the initial list into a valid heap structure.
        """
        n = len(self.heap)
        for i in range(n // 2 - 1, -1, -1):
            self._heapify_down(i)

    def push(self, item):
        """
        Inserts an item into the heap.

        Parameters:
        - item (Any): The item to insert.
        """
        self.heap.append((self.key(item), item))
        self._heapify_up(len(self.heap) - 1)

    def pop(self):
        """
        Removes and returns the smallest item from the heap.

        Returns:
        - Any: The item with the smallest key.

        Raises:
        - IndexError: If the heap is empty.
        """
        if len(self.heap) > 1:
            self._swap(0, len(self.heap) - 1)
        if not self.heap:
            raise IndexError("pop from empty heap")
        item = self.heap.pop()[1]
        self._heapify_down(0)
        return item

    def peek(self):
        """
        Returns the smallest item without removing it.

        Returns:
        - Any: The item with the smallest key, or None if empty.
        """
        return self.heap[0][1] if self.heap else None

    def size(self):
        """
        Returns the number of elements in the heap.

        Returns:
        - int: The size of the heap.
        """
        return len(self.heap)

    def _heapify_up(self, index):
        """
        Maintains the heap property by moving the item at index up the tree.
        """
        parent = (index - 1) // 2
        while index > 0 and self.heap[parent][0] > self.heap[index][0]:
            self._swap(parent, index)
            index = parent
            parent = (index - 1) // 2

    def _heapify_down(self, index):
        """
        Maintains the heap property by moving the item at index down the tree.
        """
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
        """
        Swaps two elements in the internal heap list.

        Parameters:
        - i (int): First index.
        - j (int): Second index.
        """
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        
'''
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
'''

class SegmentTree:
    """
    A class implementing a segment tree for efficient range sum queries and point updates.

    Supports:
    - Building a tree of size `n`
    - Setting a value at a specific index
    - Querying the sum over a range [start, end]
    
    All operations are O(log n).

    ### Example usage:
    ```python
    st = SegmentTree(8)
    st.set(3, 5)
    st.set(5, 2)
    print(st.query(3, 5))  # Output: 7
    ```

    Attributes:
    - DEF (int): Default value used in segment tree operations (e.g., 0 for sum).
    - n (int): Size of the array.
    - SegTree (List[int]): Internal array storing the segment tree.
    """
    
    DEF = 0  # Default value for sum queries

    def __init__(self, n: int) -> None:
        """
        Initializes a segment tree for an array of size `n`.

        Parameters:
        - n (int): The number of elements in the array.
        """
        self.n = n
        self.SegTree = [self.DEF] * (2 * n)

    def set(self, idx: int, val: int) -> None:
        """
        Sets the value at index `idx` and updates the segment tree.

        Parameters:
        - idx (int): The index to update (0-based).
        - val (int): The new value to set.
        """
        idx += self.n
        self.SegTree[idx] = val
        while idx > 1:
            self.SegTree[idx // 2] = self.SegTree[idx] + self.SegTree[idx ^ 1]
            idx //= 2

    def query(self, start: int, end: int) -> int:
        """
        Returns the sum of the values in the range [start, end].

        Parameters:
        - start (int): The starting index of the query range (0-based).
        - end (int): The ending index of the query range (0-based).

        Returns:
        - int: The sum of elements in the specified range.
        """
        result = self.DEF
        start += self.n
        end += self.n
        while start <= end:
            if start % 2 == 1:
                result += self.SegTree[start]
                start += 1
            if end % 2 == 0:
                result += self.SegTree[end]
                end -= 1
            start //= 2
            end //= 2
        return result

    @staticmethod
    def LOG2(n: int) -> int:
        """
        Computes the floor of the base-2 logarithm of `n`.

        Parameters:
        - n (int): The number to compute log2 for.

        Returns:
        - int: The largest integer `l` such that 2^l <= n.
        """
        log = 0
        while (1 << (log + 1)) <= n:
            log += 1
        return log

'''
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
'''


class BIT:
    """
    A class that implements a Binary Indexed Tree (Fenwick Tree) for efficient
    prefix sums and updates on a list of numbers.

    Supports point updates and prefix/range sum queries in O(log n) time.

    ### Example usage:
    ```python
    bit = BIT(10)
    bit.add(3, 5)
    bit.add(5, 2)
    print(bit.sum(5))        # Output: 7 (5 at index 3 + 2 at index 5)
    print(bit.range_sum(3, 5))  # Output: 7
    ```

    Attributes:
    - size (int): The size of the array (1-based indexing).
    - tree (List[int]): Internal array representing the Binary Indexed Tree.
    """

    def __init__(self, size):
        """
        Initializes the BIT with a given size.

        Parameters:
        - size (int): The number of elements the BIT will manage (1-based indexing).
        """
        self.size = size
        self.tree = [0] * (size + 1)

    def add(self, index, value):
        """
        Adds `value` to the element at `index`.

        Parameters:
        - index (int): The 1-based index to update.
        - value (int): The value to add.
        """
        while index <= self.size:
            self.tree[index] += value
            index += index & -index

    def sum(self, index):
        """
        Computes the prefix sum from index 1 to `index`.

        Parameters:
        - index (int): The 1-based index up to which the sum is computed.

        Returns:
        - int: The prefix sum.
        """
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= index & -index
        return result

    def range_sum(self, left, right):
        """
        Computes the sum of the elements in the range [left, right].

        Parameters:
        - left (int): The starting 1-based index.
        - right (int): The ending 1-based index.

        Returns:
        - int: The sum of values in the range.
        """
        return self.sum(right) - self.sum(left - 1)

'''
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
'''

class DSU:
    """
    A class representing a Disjoint Set Union (Union-Find) data structure.

    Supports efficient union and find operations with union by size optimization.

    ### Example usage:
    ```python
    dsu = DSU(5)
    dsu.link(0, 1)
    dsu.link(1, 2)
    print(dsu.same_set(0, 2))  # True
    print(dsu.same_set(0, 3))  # False
    print(dsu.get(2))          # Representative of the set containing 2
    ```

    Attributes:
    - parent (List[int]): Points to the parent of each node.
    - size (List[int]): Stores the size of each component's root.
    """

    def __init__(self, n: int) -> None:
        """
        Initializes the DSU for `n` elements.

        Parameters:
        - n (int): The number of elements (from 0 to n-1).
        """
        self.parent = [i for i in range(n)]
        self.size = [1 for _ in range(n)]

    def get(self, x: int) -> int:
        """
        Finds the representative (root) of the set containing `x`.

        Parameters:
        - x (int): The element to find.

        Returns:
        - int: The representative of the set.
        """
        while x != self.parent[x]:
            x = self.parent[x]
        return x

    def same_set(self, a: int, b: int) -> bool:
        """
        Checks whether elements `a` and `b` are in the same set.

        Parameters:
        - a (int), b (int): Elements to compare.

        Returns:
        - bool: True if in the same set, False otherwise.
        """
        return self.get(a) == self.get(b)

    def component_size(self, x: int) -> int:
        """
        Returns the size of the set containing element `x`.

        Parameters:
        - x (int): The element to query.

        Returns:
        - int: The size of the component.
        """
        return self.size[self.get(x)]

    def link(self, a: int, b: int) -> bool:
        """
        Merges the sets containing `a` and `b`.

        Parameters:
        - a (int), b (int): The elements to union.

        Returns:
        - bool: True if a merge happened, False if already in the same set.
        """
        a = self.get(a)
        b = self.get(b)
        if a == b:
            return False
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.size[a] += self.size[b]
        self.parent[b] = a
        return True

class SuffixArray:
    """
    A class that constructs and stores the suffix array for a given string.

    The suffix array is a sorted list of all suffix starting indices in the string,
    which enables efficient substring searching and other text processing tasks.

    ### Example usage:
    ```python
    sa = SuffixArray("banana")
    print(sa.get_suffix_array())  # Output: [5, 3, 1, 0, 4, 2]
    ```

    Attributes:
    - text (str): The original input string.
    - suffix_array (List[int]): The sorted list of suffix starting indices.
    """

    def __init__(self, text):
        """
        Initializes the SuffixArray with a given string and builds the suffix array.

        Parameters:
        - text (str): The input string to construct the suffix array from.
        """
        self.text = text
        self.suffix_array = self._build_suffix_array(text)

    def _build_suffix_array(self, text):
        """
        Constructs the suffix array by sorting all suffixes of the input text.

        Parameters:
        - text (str): The string from which to generate suffixes.

        Returns:
        - List[int]: A list of starting indices of sorted suffixes.
        """
        suffixes = [(text[i:], i) for i in range(len(text))]
        suffixes.sort(key=lambda x: x[0])
        return [suffix[1] for suffix in suffixes]

    def get_suffix_array(self):
        """
        Returns the suffix array.

        Returns:
        - List[int]: The sorted list of suffix starting indices.
        """
        return self.suffix_array

class SuffixTreeNode:
    """
    Represents a node in the suffix tree.

    Attributes:
    - children (dict): Mapping from characters to child nodes.
    - suffix_link (SuffixTreeNode): Optional link used in advanced suffix tree construction (not used here).
    - start (int): The start index of the substring represented by this node.
    - end (int): The end index of the substring represented by this node.
    """

    def __init__(self):
        """
        Initializes an empty suffix tree node.
        """
        self.children = {}
        self.suffix_link = None
        self.start = None
        self.end = None


class SuffixTree:
    """
    A simplified implementation of a suffix tree for substring search.

    Note: This is not a full Ukkonen's algorithm and has O(n²) construction time.
    Suitable for educational or small-scale string search problems.

    ### Example usage:
    ```python
    st = SuffixTree("banana")
    print(st.search("ana"))   # True
    print(st.search("nana"))  # True
    print(st.search("apple")) # False
    ```

    Attributes:
    - text (str): The original string used to build the suffix tree.
    - root (SuffixTreeNode): The root node of the suffix tree.
    """

    def __init__(self, text):
        """
        Initializes and constructs the suffix tree for the given text.

        Parameters:
        - text (str): The input string to build the suffix tree from.
        """
        self.text = text
        self.root = SuffixTreeNode()
        self.suffix_link = None
        self._build_suffix_tree()

    def _build_suffix_tree(self):
        """
        Builds the suffix tree by inserting all suffixes of the text one by one.
        """
        n = len(self.text)
        for i in range(n):
            self._extend_suffix(i)

    def _extend_suffix(self, i):
        """
        Inserts the suffix starting at index `i` into the tree.

        Parameters:
        - i (int): The starting index of the suffix to insert.
        """
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
        """
        Searches for a substring pattern in the suffix tree.

        Parameters:
        - pattern (str): The substring to search for.

        Returns:
        - bool: True if the pattern exists in the text, False otherwise.
        """
        current = self.root
        for char in pattern:
            if char not in current.children:
                return False
            current = current.children[char]
        return True


class SkipListNode:
    """
    Represents a node in a skip list.

    Attributes:
    - key: The key associated with the node.
    - value: The value stored in the node.
    - forward (List[SkipListNode]): Pointers to nodes at each level.
    """

    def __init__(self, key, value):
        """
        Initializes a skip list node with a key and value.
        
        Parameters:
        - key: The key for ordering in the skip list.
        - value: The value to store.
        """
        self.key = key
        self.value = value
        self.forward = []


class SkipList:
    """
    A probabilistic data structure that allows fast search, insertion, and deletion in O(log n) time on average.

    The skip list maintains multiple forward pointers at each level, allowing for fast traversal by skipping over nodes.

    ### Example usage:
    ```python
    sl = SkipList()
    sl.insert(10, "a")
    sl.insert(20, "b")
    sl.insert(15, "c")

    print(sl.search(15))  # "c"
    print(sl.search(99))  # None

    sl.delete(15)
    print(sl.search(15))  # None
    ```
    """

    def __init__(self):
        """
        Initializes an empty skip list with a head node having negative infinity as key.
        """
        self.head = SkipListNode(float('-inf'), None)
        self.max_level = 1

    def _random_level(self):
        """
        Generates a random level for a new node based on a geometric distribution.
        
        Returns:
        - int: A randomly determined level for the new node.
        """
        level = 1
        while random.random() < 0.5 and level < self.max_level + 1:
            level += 1
        return level

    def insert(self, key, value):
        """
        Inserts a key-value pair into the skip list. If the key already exists, its value is updated.

        Parameters:
        - key: The key to insert.
        - value: The value to associate with the key.
        """
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
            if len(new_node.forward) < i + 1:
                new_node.forward.extend([None] * (i + 1 - len(new_node.forward)))
            if len(update[i].forward) < i + 1:
                update[i].forward.extend([None] * (i + 1 - len(update[i].forward)))
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

    def search(self, key):
        """
        Searches for a given key in the skip list.

        Parameters:
        - key: The key to search for.

        Returns:
        - The value associated with the key, or None if not found.
        """
        current = self.head
        for i in range(self.max_level, 0, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        current = current.forward[1] if len(current.forward) > 1 else None
        if current and current.key == key:
            return current.value
        return None

    def delete(self, key):
        """
        Deletes a key-value pair from the skip list, if it exists.

        Parameters:
        - key: The key to delete.
        """
        update = [None] * (self.max_level + 1)
        current = self.head

        for i in range(self.max_level, 0, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[1] if len(current.forward) > 1 else None
        if current and current.key == key:
            for i in range(1, self.max_level + 1):
                if len(update[i].forward) <= i or update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i] if len(current.forward) > i else None

            while self.max_level > 1 and (len(self.head.forward) <= self.max_level or self.head.forward[self.max_level] is None):
                self.max_level -= 1

class BitSet:
    """
    A fixed-size bitset implementation using an integer array as a bitfield.

    Supports setting, clearing, flipping, and querying bits efficiently.
    
    ### Example usage:
    ```python
    bs = BitSet(10)
    bs.set(3)
    bs.set(5)
    print(bs.test(3))  # True
    print(bs[5])       # True (via __getitem__)
    bs[5] = False
    print(bs[5])       # False
    print(bs.count())  # 1
    print(bs)          # 0001000000
    ```

    Attributes:
    - size (int): The number of bits in the BitSet.
    - arr (List[int]): Internal representation as a list of 32-bit integers.
    """

    def __init__(self, size):
        """
        Initializes a BitSet of the given size.

        Parameters:
        - size (int): The number of bits the BitSet should manage.
        """
        self.size = size
        self.arr = [0] * ((size + 31) // 32)

    def set(self, pos):
        """
        Sets the bit at position `pos` to 1.

        Raises:
        - IndexError: If `pos` is out of bounds.
        """
        if pos < 0 or pos >= self.size:
            raise IndexError("Index out of range")
        word_index = pos // 32
        bit_index = pos % 32
        self.arr[word_index] |= (1 << bit_index)

    def reset(self, pos):
        """
        Resets the bit at position `pos` to 0.

        Raises:
        - IndexError: If `pos` is out of bounds.
        """
        if pos < 0 or pos >= self.size:
            raise IndexError("Index out of range")
        word_index = pos // 32
        bit_index = pos % 32
        self.arr[word_index] &= ~(1 << bit_index)

    def test(self, pos):
        """
        Checks whether the bit at position `pos` is set (1).

        Returns:
        - bool: True if set, False otherwise.

        Raises:
        - IndexError: If `pos` is out of bounds.
        """
        if pos < 0 or pos >= self.size:
            raise IndexError("Index out of range")
        word_index = pos // 32
        bit_index = pos % 32
        return (self.arr[word_index] & (1 << bit_index)) != 0

    def flip(self, pos):
        """
        Flips the bit at position `pos` (0 → 1, 1 → 0).

        Raises:
        - IndexError: If `pos` is out of bounds.
        """
        if pos < 0 or pos >= self.size:
            raise IndexError("Index out of range")
        word_index = pos // 32
        bit_index = pos % 32
        self.arr[word_index] ^= (1 << bit_index)

    def any(self):
        """
        Checks if any bit is set in the BitSet.

        Returns:
        - bool: True if at least one bit is set, False otherwise.
        """
        for word in self.arr:
            if word != 0:
                return True
        return False

    def none(self):
        """
        Checks if all bits are unset in the BitSet.

        Returns:
        - bool: True if all bits are 0, False otherwise.
        """
        for word in self.arr:
            if word != 0:
                return False
        return True

    def all(self):
        """
        Checks if all bits are set in the BitSet.

        Returns:
        - bool: True if all bits are 1, False otherwise.
        """
        for word in self.arr:
            if word != 0xFFFFFFFF:
                return False
        return True

    def count(self):
        """
        Counts the number of bits that are set to 1.

        Returns:
        - int: The number of set bits.
        """
        count = 0
        for word in self.arr:
            count += bin(word).count('1')
        return count

    def size(self):
        """
        Returns the total number of bits in the BitSet.

        Returns:
        - int: The size of the BitSet.
        """
        return self.size

    def __getitem__(self, pos):
        """
        Gets the value of the bit at position `pos`.

        Returns:
        - bool: True if the bit is set, False otherwise.
        """
        return self.test(pos)

    def __setitem__(self, pos, value):
        """
        Sets or resets the bit at position `pos` based on `value`.

        Parameters:
        - value (bool): If True, set the bit; if False, reset it.
        """
        if value:
            self.set(pos)
        else:
            self.reset(pos)

    def __repr__(self):
        """
        Returns a string representation of the BitSet in binary format.

        Example:
        - "00101000" for a size-8 bitset with bits 3 and 5 set.
        """
        return ''.join(['1' if self.test(i) else '0' for i in range(self.size)])

class SieveOfEratosthenes:
    """
    A class that implements the Sieve of Eratosthenes algorithm to efficiently find all prime numbers
    up to a given limit `n`.

    Attributes:
    - n (int): The upper bound for prime generation.
    - is_prime (List[bool]): Boolean array indicating primality for each number from 0 to n.
    - primes (List[int]): List of all prime numbers up to n.
    """

    def __init__(self, n):
        """
        Initializes the sieve with a given upper limit and generates the list of prime numbers.

        Parameters:
        - n (int): The upper bound for prime number generation.
        """
        self.n = n
        self.is_prime = [True] * (n + 1)
        self.primes = self._generate_primes()

    def _generate_primes(self):
        """
        Executes the Sieve of Eratosthenes algorithm to find all primes up to self.n.

        Returns:
        - List[int]: A list of prime numbers less than or equal to n.
        """
        p = 2
        while (p * p <= self.n):
            if (self.is_prime[p] == True):
                for i in range(p * p, self.n + 1, p):
                    self.is_prime[i] = False
            p += 1
        return [p for p in range(2, self.n + 1) if self.is_prime[p]]

    def setN(self, newN):
        """
        Updates the upper bound `n` and regenerates the sieve and prime list.

        Parameters:
        - newN (int): The new upper bound for prime number generation.
        """
        self.n = newN
        self.is_prime = [True] * (newN + 1)
        self.primes = self._generate_primes()

    def get_primes(self):
        """
        Retrieves the list of prime numbers up to the current value of n.

        Returns:
        - List[int]: The list of prime numbers.
        """
        return self.primes

    def getPrimeArr(self):
        """
        Retrieves the boolean array indicating the primality of numbers up to n.

        Returns:
        - List[bool]: Boolean list where index i is True if i is prime, otherwise False.
        """
        return self.is_prime


class SparseTableRMQ:
    """
    A class that implements the Sparse Table data structure for efficient Range Minimum Query (RMQ) operations.
    
    This implementation supports immutable arrays and allows for answering minimum queries in constant time (O(1))
    after O(n log n) preprocessing time.
    """

    def __init__(self, array):
        """
        Initializes the Sparse Table with the given input array and precomputes minimum values for all subranges.

        Parameters:
        - array (List[int]): The input list of integers on which RMQ operations will be performed.

        Preprocessing:
        Builds a 2D list `st` where `st[i][j]` represents the minimum value in the subarray
        starting at index `i` with length `2^j`.
        """
        self.n = len(array)
        self.k = math.floor(math.log2(self.n)) + 1
        self.st = [[0] * self.k for _ in range(self.n)]

        for i in range(self.n):
            self.st[i][0] = array[i]

        j = 1
        while (1 << j) <= self.n:
            i = 0
            while (i + (1 << j) - 1) < self.n:
                self.st[i][j] = min(self.st[i][j - 1], self.st[i + (1 << (j - 1))][j - 1])
                i += 1
            j += 1

    def query(self, l, r):
        """
        Answers the Range Minimum Query (RMQ) for the subarray from index l to r (inclusive).

        Parameters:
        - l (int): Left index of the query range (0-based).
        - r (int): Right index of the query range (0-based).

        Returns:
        - int: The minimum value in the subarray array[l..r].

        Time Complexity:
        O(1)
        """
        j = math.floor(math.log2(r - l + 1))
        return min(self.st[l][j], self.st[r - (1 << j) + 1][j])

