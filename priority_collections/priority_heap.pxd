cimport numpy as np

ctypedef np.npy_int64 index_t
ctypedef np.npy_float32 float_t


cdef packed struct Node:
    index_t id
    float_t value


cdef class MinPriorityHeap:

    cdef index_t heap_ptr, capacity

    cdef float_t atol, rtol

    cdef Node* heap

    cdef inline bint empty(self)

    cdef int cpush(self, index_t node_id, float_t value) except -1
    cdef int cpop(self, index_t* out_id, float_t* out_value) except -1

    cpdef np.ndarray get_ids(self)
    cpdef void push(self, index_t node_id, float value)
    cpdef (index_t, float_t) pop(self)


cdef class MaxPriorityHeap(MinPriorityHeap):


    cdef int cpush(self, index_t node_id, float_t value) except -1
    cdef int cpop(self, index_t* out_id, float_t* out_value) except -1

