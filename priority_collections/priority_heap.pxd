cimport numpy as np

ctypedef np.npy_int64 index_t
ctypedef np.npy_float32 float_t


cdef packed struct Node:
    index_t id
    index_t pos
    index_t value


cdef class MinHeap:

    cdef readonly index_t _heap_ptr, _capacity, _decimals

    cdef Node* heap

    cdef inline bint empty(self)
    cdef inline float_t fixed_point_to_float(self, index_t fixed_point_value)
    cdef inline index_t float_to_fixed_point(self, float_t value)
    cdef inline void append(self, index_t node_id, index_t value)

    cdef int cpush(self, index_t node_id, index_t value) except -1
    cdef int cpop(self, index_t* out_id, index_t* out_value) except -1
    cdef int cupdate(self, index_t node_id, index_t value) except -1

    cpdef void push(self, index_t node_id, float_t value)
    cpdef (index_t, float_t) pop(self)
    cpdef void update(self, index_t node_id, float_t value)


cdef class MaxHeap(MinHeap):


    cdef int cpush(self, index_t node_id, index_t value) except -1
    cdef int cpop(self, index_t* out_id, index_t* out_value) except -1
    cdef int cupdate(self, index_t node_id, index_t value) except -1

