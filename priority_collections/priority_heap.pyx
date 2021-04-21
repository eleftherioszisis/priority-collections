# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport fabs, fmax
from libc.stdlib cimport free, realloc

import numpy as np

SIGNED_NUMPY_TYPE_MAP = {2 : np.int16, 4 : np.int32, 8 : np.int64}


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
    heap[pos1], heap[pos2] = heap[pos2], heap[pos1]


cdef inline index_t get_left_child(index_t index):
    """Returns left child index in a heap"""
    return 2 * index + 1


cdef inline index_t get_right_child(index_t index):
    """Returns right child index in a heap"""
    return 2 * index + 2


cdef inline index_t get_parent(index_t index):
    """Returns the parent from index pos in a heap"""
    return (index - 1) / 2


cdef inline bint isclose(x, y, atol, rtol):
    """Behaves like the math.isclose function"""
    return fabs(x - y) <= fmax(atol, rtol * fmax(fabs(x), fabs(y)))


cdef inline bint less(a, b, atol, rtol):
    """Returns True if a is strictly smaller than b"""
    return not isclose(a, b, atol, rtol) and a < b


cdef inline bint greater(a, b, atol, rtol):
    """Returns True if a is strictly greater that b"""
    return not isclose(a, b, atol, rtol) and a > b


cdef inline bint less_equal(a, b, atol, rtol):
    """Returns True if a is less or equal than b"""
    return isclose(a, b, atol, rtol) or a < b


cdef inline bint greater_equal(a, b, atol, rtol):
    """Returns True if a is greater or equal than b"""
    return isclose(a, b, atol, rtol) or a > b


cdef void min_heapify_down(Node* heap, index_t index, index_t heap_length, float_t atol, float_t rtol):

    cdef :
        index_t left = get_left_child(index)
        index_t right = get_right_child(index)
        index_t largest = index

    if left < heap_length and less_equal(heap[left].value, heap[largest].value, atol, rtol):
        largest = left

    if right < heap_length and less_equal(heap[right].value, heap[largest].value, atol, rtol):
        largest = right

    if largest != index:
        swap(heap, index, largest)
        min_heapify_down(heap, largest, heap_length, atol, rtol)


cdef void max_heapify_down(Node* heap, index_t index, index_t heap_length, float_t atol, float_t rtol):

    cdef:
        index_t left = get_left_child(index)
        index_t right = get_right_child(index)
        index_t smallest = index

    if left < heap_length and greater_equal(heap[left].value, heap[smallest].value, atol, rtol):
        smallest = left

    if right < heap_length and greater_equal(heap[right].value, heap[smallest].value, atol, rtol):
        smallest = right

    if smallest != index:
        swap(heap, index, smallest)
        max_heapify_down(heap, smallest, heap_length, atol, rtol)


cdef void min_heapify_up(Node* heap, index_t index, float_t atol, float_t rtol):

    if index == 0:
        return

    cdef index_t parent = get_parent(index)

    if greater(heap[parent].value, heap[index].value, atol, rtol):
        swap(heap, parent, index)

    min_heapify_up(heap, parent, atol, rtol)


cdef void max_heapify_up(Node* heap, index_t index, float_t atol, float_t rtol):

    if index == 0:
        return

    cdef index_t parent = get_parent(index)

    if less(heap[parent].value, heap[index].value, atol, rtol):
        swap(heap, parent, index)

    max_heapify_up(heap, parent, atol, rtol)


cdef class MinHeap:

    def __cinit__(self, index_t capacity, atol, rtol):

        self.capacity = capacity
        self.heap_ptr = 0

        self.atol = atol
        self.rtol = rtol

        safe_realloc(&self.heap, capacity)

    def __dealloc__(self):
        free(self.heap)

    def __bool__(self):
        return self.heap_ptr > 0

    def __len__(self):
        return self.heap_ptr

    cdef inline bint empty(self):
        return self.heap_ptr <= 0

    cdef int cpush(self, index_t node_id, float_t value) except -1:

        # Resize if capacity not sufficient
        if self.heap_ptr >= self.capacity:

            self.capacity *= 2
            # Since safe_realloc can raise MemoryError, use `except -1`
            safe_realloc(&self.heap, self.capacity)

        # Put element as last element of heap
        self.heap[self.heap_ptr].id = node_id
        self.heap[self.heap_ptr].value = value

        # Heapify up
        min_heapify_up(self.heap, self.heap_ptr, self.atol, self.rtol)

        # Increase element count
        self.heap_ptr += 1

        return 0

    cdef int cpop(self, index_t* out_id, float_t* out_value) except -1:

        if self.empty():
            return -1

        # Take first element
        out_id[0] = self.heap[0].id
        out_value[0] = self.heap[0].value

        # swap with last element
        swap(self.heap, 0, self.heap_ptr - 1)

        if not self.empty():
            min_heapify_down(self.heap, 0, self.heap_ptr - 1, self.atol, self.rtol)

        # reduce the array length
        self.heap_ptr -= 1

        return 0


    @property
    def ids(self):
        return self.get_ids()

    cpdef np.ndarray get_ids(self):
        cdef:
            index_t n_elements = len(self)
            index_t[:] res = np.empty(n_elements, dtype=SIGNED_NUMPY_TYPE_MAP[sizeof(index_t)])

        for i in range(n_elements):
            res[i] = self.heap[i].id

        return np.asarray(res)


    cpdef void push(self, index_t node_id, float_t value):
        self.cpush(node_id, value)

    cpdef (index_t, float_t) pop(self):
        cdef:
            index_t node_id
            float_t value

        self.cpop(&node_id, &value)
        return node_id, value


cdef class MaxHeap(MinHeap):

    cdef int cpush(self, index_t node_id, float_t value) except -1:

        # Resize if capacity not sufficient
        if self.heap_ptr >= self.capacity:

            self.capacity *= 2
            # Since safe_realloc can raise MemoryError, use `except -1`
            safe_realloc(&self.heap, self.capacity)

        # Put element as last element of heap
        self.heap[self.heap_ptr].id = node_id
        self.heap[self.heap_ptr].value = value

        # Heapify up
        max_heapify_up(self.heap, self.heap_ptr, self.atol, self.rtol)

        # Increase element count
        self.heap_ptr += 1

        return 0

    cdef int cpop(self, index_t* out_id, float_t* out_value) except -1:

        if self.empty():
            return -1

        # Take first element
        out_id[0] = self.heap[0].id
        out_value[0] = self.heap[0].value

        # swap with last element
        swap(self.heap, 0, self.heap_ptr - 1)

        if not self.empty():
            max_heapify_down(self.heap, 0, self.heap_ptr - 1, self.atol, self.rtol)

        # reduce the array length
        self.heap_ptr -= 1

        return 0
