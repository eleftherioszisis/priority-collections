import pytest
import numpy.testing as npt
import priority_collections.priority_heap as tested


def _assert_heap_pair_equal(actual_pair, desired_pair):
    npt.assert_equal(actual_pair[0], desired_pair[0])
    npt.assert_almost_equal(actual_pair[1], desired_pair[1])


def _assert_heap_equal(heap, expected_ids, expected_values):

    ids = []
    values = []

    while heap:

        node_id, value = heap.pop()

        ids.append(node_id)
        values.append(value)

    try:

        npt.assert_array_equal(ids, expected_ids)
        npt.assert_allclose(values, expected_values)

    except AssertionError:

        msg = (
            f'\n\nActual ids    : {ids}\n'
            f'Desired ids   : {expected_ids}\n\n'
            f'Actual values : {values}\n'
            f'Desired values: {expected_values}'
        )
        raise AssertionError(msg)

def build_heap(ids, values, capacity, heap_class, atol, rtol):
    """
    Args:
        ids (Iterable): iterable of integer ids
        values (Iterable): iterable of float values
        capacity (int): Initial memory chunk allocation of the heap

    Returns:
        MinPriorityHeap
    """
    heap = heap_class(capacity, atol, rtol)

    for node_id, value in zip(ids, values):

        heap.push(node_id, value)

    return heap


def build_min_heap(ids, values, capacity, atol=0.0, rtol=1e-6):
    return build_heap(ids, values, capacity, tested.MinHeap, atol, rtol)


def build_max_heap(ids, values, capacity, atol=0.0, rtol=1e-6):
    return build_heap(ids, values, capacity, tested.MaxHeap, atol, rtol)


def test_min_priority_heap__capacity():
    """
    heap = MinPriorityHeap(1)

    heap.push(1, 0.0)

    npt.assert_equal(len)
    """


def test_min_priority_heap__invariant():

    ids = [1]
    values = [0.1]
    heap = build_min_heap(ids, values, capacity=10)
    _assert_heap_equal(heap, ids, values)

    ids = [0, 1]
    values = [0.0, 0.1]
    heap = build_min_heap(ids, values, capacity=10)
    _assert_heap_equal(heap, ids, values)

    ids = [0, 1, 2]
    values = [0.0, 0.1, 0.2]
    heap = build_min_heap(ids, values, capacity=10)
    _assert_heap_equal(heap, ids, values)

    heap = build_min_heap([0, 1], [0.1, 0.0], capacity=10)
    _assert_heap_equal(heap, [1, 0], [0.0, 0.1])


def test_min_priority_heap__epsilon():

    ids = [0, 1, 2]

    values = [0.1, 0.1, 0.1]
    heap = build_min_heap(ids, values, capacity=10, atol=0.0, rtol=1e-2)
    _assert_heap_equal(heap, ids, values)

    values = [0.2000003, 0.2000000001, 0.200000002]
    heap = build_min_heap(ids, values, capacity=10, atol=1e-6, rtol=0.0)
    _assert_heap_equal(heap, ids, values)

    values = [0.0000000001, 0.0000000001, 0.0000000001]
    heap = build_min_heap(ids, values, capacity=10, atol=1e-10)
    _assert_heap_equal(heap, ids, values)

    values = [0.1, 0.2, 0.1]
    heap = build_min_heap(ids, values, capacity=10, atol=1e-2, rtol=0.0)
    _assert_heap_equal(heap, [0, 2, 1], [0.1, 0.1, 0.2])


def test_max_priority_heap__epsilon():

    ids = [0, 1, 2]

    values = [0.1, 0.1, 0.1]
    heap = build_max_heap(ids, values, capacity=10, atol=0.0, rtol=1e-2)
    _assert_heap_equal(heap, ids, values)

    values = [0.2000003, 0.2000000001, 0.200000002]
    heap = build_max_heap(ids, values, capacity=10, atol=1e-6, rtol=0.0)
    _assert_heap_equal(heap, ids, values)

    values = [0.0000000001, 0.0000000001, 0.0000000001]
    heap = build_max_heap(ids, values, capacity=10, atol=1e-10)
    _assert_heap_equal(heap, ids, values)

    values = [0.1, 0.2, 0.1]
    heap = build_max_heap(ids, values, capacity=10, atol=1e-2, rtol=0.0)
    _assert_heap_equal(heap, [1, 0, 2], [0.2, 0.1, 0.1])


def test_max_priority_heap__invariant():

    ids = [1]
    values = [0.1]
    heap = build_max_heap(ids, values, capacity=10)
    _assert_heap_equal(heap, ids, values)

    ids = [0, 1]
    values = [0.1, 0.0]
    heap = build_max_heap(ids, values, capacity=10)
    _assert_heap_equal(heap, ids, values)

    ids = [0, 1, 2]
    values = [0.2, 0.1, 0.0]
    heap = build_max_heap(ids, values, capacity=10)
    _assert_heap_equal(heap, ids, values)

    heap = build_max_heap([0, 1], [0.0, 0.1], capacity=10)
    _assert_heap_equal(heap, [1, 0], [0.1, 0.0])

