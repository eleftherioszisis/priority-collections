# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdlib cimport free, realloc

import numpy as np

SIGNED_NUMPY_TYPE_MAP = {2 : np.int16, 4 : np.int32, 8 : np.int64}
FLOAT_NUMPY_TYPE_MAP = {4: np.float32, 8: np.float64}


ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (Node*)
    (index_t*)


# safe_realloc(&p, n) resizes the allocation of p to n * sizeof(*p) bytes or
# raises a MemoryError. It never calls free, since that's __dealloc__'s job.
#   cdef DTYPE_t *p = NULL
#   safe_realloc(&p, n)
# is equivalent to p = malloc(n * sizeof(*p)) with error checking.
cdef void safe_realloc(realloc_ptr* p, size_t nelems) nogil except *:
    # sizeof(realloc_ptr[0]) would be more like idiomatic C, but causes Cython
    # 0.20.1 to crash.

    cdef size_t nbytes = nelems * sizeof(p[0][0])

    if nbytes / sizeof(p[0][0]) != nelems:
        # Overflow in the multiplication
        with gil:
            raise MemoryError("could not allocate (%d * %d) bytes"
                              % (nelems, sizeof(p[0][0])))

    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)

    if tmp == NULL:
        with gil:
            raise MemoryError("could not allocate %d bytes" % nbytes)

    p[0] = tmp


cdef inline void swap(Node* heap, pos1, pos2):
    """Swap pos1 and pos2 in the heap array"""
    heap[heap[pos1].id].pos, heap[heap[pos2].id].pos = \
    heap[heap[pos2].id].pos, heap[heap[pos1].id].pos
    heap[pos1].id, heap[pos2].id = heap[pos2].id, heap[pos1].id
    heap[pos1].value, heap[pos2].value = heap[pos2].value, heap[pos1].value


cdef inline index_t get_left_child(index_t index):
    """Returns left child index in a heap"""
    return 2 * index + 1


cdef inline index_t get_right_child(index_t index):
    """Returns right child index in a heap"""
    return 2 * index + 2


cdef inline index_t get_parent(index_t index):
    """Returns the parent from index pos in a heap"""
    return (index - 1) / 2


cdef void min_heapify_down(Node* heap, index_t index, index_t heap_length):

    cdef :
        index_t left = get_left_child(index)
        index_t right = get_right_child(index)
        index_t largest = index

    if left < heap_length and heap[left].value <= heap[largest].value:
        largest = left

    if right < heap_length and heap[right].value <= heap[largest].value:
        largest = right

    if largest != index:
        swap(heap, index, largest)
        min_heapify_down(heap, largest, heap_length)


cdef void max_heapify_down(Node* heap, index_t index, index_t heap_length):

    cdef:
        index_t left = get_left_child(index)
        index_t right = get_right_child(index)
        index_t smallest = index

    if left < heap_length and heap[left].value >= heap[smallest].value:
        smallest = left

    if right < heap_length and heap[right].value >= heap[smallest].value:
        smallest = right

    if smallest != index:
        swap(heap, index, smallest)
        max_heapify_down(heap, smallest, heap_length)


cdef void min_heapify_up(Node* heap, index_t index):

    if index == 0:
        return

    cdef index_t parent = get_parent(index)

    if heap[parent].value > heap[index].value:
        swap(heap, parent, index)

    min_heapify_up(heap, parent)


cdef void max_heapify_up(Node* heap, index_t index):

    if index == 0:
        return

    cdef index_t parent = get_parent(index)

    if heap[parent].value < heap[index].value:
        swap(heap, parent, index)

    max_heapify_up(heap, parent)


cdef class MinHeap:

    def __cinit__(self, index_t capacity, index_t decimals=6):

        self._capacity = capacity
        self._heap_ptr = 0
        self._decimals = decimals

        safe_realloc(&self.heap, capacity)

    def __dealloc__(self):
        free(self.heap)

    def __bool__(self):
        return self._heap_ptr > 0

    def __len__(self):
        """Returns the number of elements in the heap"""
        return self._heap_ptr

    cdef inline bint empty(self):
        """Returns true if no elements in the heap"""
        return self._heap_ptr <= 0

    cdef inline float_t fixed_point_to_float(self, index_t fixed_point_value):
        return fixed_point_value / <float_t>(10 ** self._decimals)

    cdef inline index_t float_to_fixed_point(self, float_t value):
        return <index_t>(value * (10 ** self._decimals))

    cdef int cpush(self, index_t node_id, index_t value) except -1:

        # Resize if capacity not sufficient
        if self._heap_ptr >= self._capacity:

            self._capacity *= 2
            # Since safe_realloc can raise MemoryError, use `except -1`
            safe_realloc(&self.heap, self._capacity)

        # Put element as last element of heap
        self.heap[self._heap_ptr].id = node_id
        self.heap[self._heap_ptr].value = value
        self.heap[self._heap_ptr].pos = self._heap_ptr

        # Heapify up
        min_heapify_up(self.heap, self._heap_ptr)

        # Increase element count
        self._heap_ptr += 1

        return 0

    cdef int cpop(self, index_t* out_id, index_t* out_value) except -1:

        if self.empty():
            return -1

        # Take first element
        out_id[0] = self.heap[0].id
        out_value[0] = self.heap[0].value

        # swap with last element
        swap(self.heap, 0, self._heap_ptr - 1)

        # reduce the array length
        self._heap_ptr -= 1

        if not self.empty():
            min_heapify_down(self.heap, 0, self._heap_ptr)

        return 0

    cdef int cupdate(self, index_t node_id, index_t new_value) except -1:

        if 0 >= node_id >= self._heap_ptr:
            return -1

        # the actual position of the node before any swapping took place
        cdef:
            index_t pos = self.heap[node_id].pos
            index_t old_value = self.heap[pos].value

        if new_value == old_value:
            return 0

        self.heap[pos].value = new_value

        if new_value > old_value:
            min_heapify_down(self.heap, pos, self._heap_ptr)
        else:
            min_heapify_up(self.heap, pos)

        return 0

    cpdef void push(self, index_t node_id, float_t value):
        self.cpush(node_id, self.float_to_fixed_point(value))

    cpdef (index_t, float_t) pop(self):
        cdef:
            index_t node_id
            index_t fixed_point_value

        self.cpop(&node_id, &fixed_point_value)

        return node_id, self.fixed_point_to_float(fixed_point_value)

    cpdef void update(self, index_t node_id, float_t value):
        self.cupdate(node_id, self.float_to_fixed_point(value))

    @property
    def ids(self):
        cdef:
            index_t n_elements = len(self)
            index_t[:] res = np.empty(n_elements, dtype=SIGNED_NUMPY_TYPE_MAP[sizeof(index_t)])

        for i in range(n_elements):
            res[i] = self.heap[i].id

        return np.asarray(res)

    @property
    def values(self):
        cdef:
            index_t n_elements = len(self)
            float_t[:] res = np.empty(n_elements, dtype=FLOAT_NUMPY_TYPE_MAP[sizeof(float_t)])

        for i in range(n_elements):
            res[i] = self.heap[i].value

        return np.asarray(res) * 0.1 ** self._decimals


cdef class MaxHeap(MinHeap):

    cdef int cpush(self, index_t node_id, index_t value) except -1:

        # Resize if capacity not sufficient
        if self._heap_ptr >= self._capacity:

            self._capacity *= 2
            # Since safe_realloc can raise MemoryError, use `except -1`
            safe_realloc(&self.heap, self._capacity)

        # Put element as last element of heap
        self.heap[self._heap_ptr].id = node_id
        self.heap[self._heap_ptr].value = value

        # Heapify up
        max_heapify_up(self.heap, self._heap_ptr)

        # Increase element count
        self._heap_ptr += 1

        return 0

    cdef int cpop(self, index_t* out_id, index_t* out_value) except -1:

        if self.empty():
            return -1

        # Take first element
        out_id[0] = self.heap[0].id
        out_value[0] = self.heap[0].value

        # swap with last element
        swap(self.heap, 0, self._heap_ptr - 1)

        # reduce the array length
        self._heap_ptr -= 1

        if not self.empty():
            max_heapify_down(self.heap, 0, self._heap_ptr)

        return 0
