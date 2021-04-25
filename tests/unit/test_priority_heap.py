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

def build_heap(ids, values, capacity, heap_class, decimals):
    """
    Args:
        ids (Iterable): iterable of integer ids
        values (Iterable): iterable of float values
        capacity (int): Initial memory chunk allocation of the heap

    Returns:
        MinPriorityHeap
    """
    heap = heap_class(capacity, decimals)

    for node_id, value in zip(ids, values):

        heap.push(node_id, value)

    return heap


def build_min_heap(ids, values, capacity, decimals=6):
    return build_heap(ids, values, capacity, tested.MinHeap, decimals)


def test_min_priority_heap__attributes_properties():

    heap = build_min_heap([0, 1, 2], [0.0, 0.1, 0.2], 3, decimals=5)

    npt.assert_array_equal(heap.ids, [0, 1, 2])
    npt.assert_allclose(heap.values, [0.0, 0.1, 0.2])
    npt.assert_almost_equal(heap._decimals, 5)
    npt.assert_equal(heap._capacity, 3)


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
    heap = build_min_heap(ids, values, capacity=10, decimals=2)
    _assert_heap_equal(heap, ids, values)

    values = [0.0000000001, 0.0000000001, 0.0000000001]
    heap = build_min_heap(ids, values, capacity=10, decimals=10)
    _assert_heap_equal(heap, ids, values)

    values = [0.1, 0.2, 0.1]
    heap = build_min_heap(ids, values, capacity=10, decimals=2)
    _assert_heap_equal(heap, [0, 2, 1], [0.1, 0.1, 0.2])


def test_min_priority_heap__ordering():

    h = tested.MinHeap(capacity=10, decimals=6)

    h.push(1, 1.)
    assert h.pop() == (1, 1.)

    h.push(0, 0.4)
    h.push(1, 11.0)
    h.push(2, 1.2)
    h.push(3, 100.)
    h.push(4, 51.)
    h.push(5, 100000.)

    _assert_heap_equal(h, [0, 2, 1, 4, 3, 5], [0.4, 1.2, 11., 51., 100., 100000.])


def test_min_priority_heap__update():

    h = tested.MinHeap(capacity=1, decimals=3)
    h.push(0, 1.3)
    h.update(0, 0.4)
    _assert_heap_equal(h, [0], [0.4])

    h = tested.MinHeap(capacity=1, decimals=3)
    h.push(0, 0.4)
    h.update(0, 1.3)
    _assert_heap_equal(h, [0], [1.3])

    h = tested.MinHeap(capacity=2, decimals=3)
    h.push(0, 0.4)
    h.push(1, 11.0)
    h.update(0, 12.0)
    _assert_heap_equal(h, [1, 0], [11., 12.])

    h = tested.MinHeap(capacity=2, decimals=3)
    h.push(0, 0.4)
    h.push(1, 11.0)
    h.update(1, 0.3)
    _assert_heap_equal(h, [1, 0], [0.3, 0.4])

    h = tested.MinHeap(capacity=2, decimals=3)
    h.push(0, 0.4)
    h.push(1, 11.0)
    h.update(0, 12.0)
    h.update(1, -1.)
    h.update(0, 0.4)
    h.update(1, 11.0)
    _assert_heap_equal(h, [0, 1], [0.4, 11.])

    h = tested.MinHeap(capacity=3, decimals=4)
    h.push(0, 0.4)
    h.push(1, 11.0)
    h.push(2, 1.2)
    h.push(3, 100.)
    h.push(4, 51.)
    h.push(5, 100000.)

    h.update(4, -1.)
    print(h.values)
    h.update(4, 10.)
    print(h.values)
    h.update(4, 0.1)

    h.update(5, 8000.)
    h.update(2, 12.)

    _assert_heap_equal(h, [4, 0, 1, 2, 3, 5], [0.1, 0.4, 11., 12., 100., 8000.])


def build_max_heap(ids, values, capacity, decimals=6):
    return build_heap(ids, values, capacity, tested.MaxHeap, decimals)


def test_max_priority_heap__attributes_properties():
    """
    heap = MinPriorityHeap(1)

    heap.push(1, 0.0)

    npt.assert_equal(len)
    """
    heap = build_max_heap([0, 1, 2], [0.2, 0.1, 0.0], capacity=3, decimals=1)

    npt.assert_array_equal(heap.ids, [0, 1, 2])
    npt.assert_allclose(heap.values, [0.2, 0.1, 0.0])
    npt.assert_equal(heap._decimals, 1)
    npt.assert_equal(heap._capacity, 3)


def test_max_priority_heap__epsilon():

    ids = [0, 1, 2]

    values = [0.1, 0.1, 0.1]
    heap = build_max_heap(ids, values, capacity=10, decimals=2)
    _assert_heap_equal(heap, ids, values)

    values = [0.2000003, 0.2000000001, 0.200000002]
    heap = build_max_heap(ids, values, capacity=10, decimals=8)
    _assert_heap_equal(heap, ids, values)

    values = [0.0000000001, 0.0000000001, 0.0000000001]
    heap = build_max_heap(ids, values, capacity=10, decimals=10)
    _assert_heap_equal(heap, ids, values)

    values = [0.1, 0.2, 0.1]
    heap = build_max_heap(ids, values, capacity=10, decimals=2)
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


def test_max_priority_heap__update():

    h = tested.MaxHeap(capacity=1, decimals=3)
    h.push(0, 1.3)
    h.update(0, 0.4)
    _assert_heap_equal(h, [0], [0.4])

    h = tested.MaxHeap(capacity=1, decimals=3)
    h.push(0, 0.4)
    h.update(0, 1.3)
    _assert_heap_equal(h, [0], [1.3])

    h = tested.MaxHeap(capacity=1, decimals=3)
    h.push(0, 0.4)
    h.update(0, 1.3)
    h.update(0, 0.4)
    _assert_heap_equal(h, [0], [0.4])

    h = tested.MaxHeap(capacity=2, decimals=3)
    h.push(0, 0.4)
    print(h.ids, h.values)
    h.push(1, 11.0)
    print(h.ids, h.values)
    h.update(0, 10.0)
    print(h.ids, h.values)
    _assert_heap_equal(h, [1, 0], [11., 10.])

    h = tested.MaxHeap(capacity=2, decimals=3)
    h.push(0, 0.4)
    h.push(1, 11.0)
    h.update(1, 0.3)
    _assert_heap_equal(h, [0, 1], [0.4, 0.3])

    h = tested.MaxHeap(capacity=2, decimals=3)
    h.push(0, 0.4)
    h.push(1, 11.0)
    h.update(0, 12.0)
    h.update(1, -1.)
    h.update(0, -2.)
    _assert_heap_equal(h, [1, 0], [-1., -2.])

    h = tested.MaxHeap(capacity=3, decimals=4)
    h.push(0, 0.4)
    h.push(1, 11.0)
    h.push(2, 1.2)
    h.push(3, 100.)
    h.push(4, 51.)
    h.push(5, 100000.)

    h.update(4, -1.)
    print(h.values)
    h.update(4, 10.)
    print(h.values)
    h.update(4, 0.1)

    h.update(5, 8000.)
    h.update(2, 12.)

    _assert_heap_equal(h, [5, 3, 2, 1, 0, 4], [8000., 100., 12., 11., 0.4, 0.1])
