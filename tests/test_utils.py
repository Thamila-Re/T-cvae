import os
import sys
from typing import List, Tuple

import numpy as np
import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import convert2onehot, padding


@pytest.mark.parametrize(('vec', 'dim', 'expected'), [
    (np.array([1, 2, 3]), 4, torch.Tensor([[0., 1., 0., 0.],[0., 0., 1., 0.],[0., 0., 0., 1.]])),
    (np.array([i for i in range(2)]), 3, torch.Tensor([[1., 0., 0.],[0., 1., 0.]])),
])
def test_convert2onehot(vec:'numpy.ndarray', dim:int, expected:'torch.Tensor'):
    assert torch.equal(convert2onehot(vec, dim), expected)


@pytest.fixture
def case_1_for_padding() -> Tuple[List['numpy.ndarray'], int, int, 'numpy.ndarray']:
    # input data  (shape = [flow_length, feature_size])
    case_1 = np.array([[1, 2], [2, 3]])
    case_2 = np.array([[10, 11], [20, 21], [30, 31]])

    # max flow length
    max_flow_len = 4

    # expected data  (shape = [number_of_cases, max_flow_length, feature_size])
    ans = np.array([[[ 1,  2], [ 2,  3], [ 0,  0], [ 0,  0]], 
                    [[10, 11], [20, 21], [30, 31], [ 0,  0]]
                    ])

    # padding value
    value = 0

    return [case_1, case_2], max_flow_len, value, ans


def test_padding(case_1_for_padding):
    assert np.array_equal(padding(case_1_for_padding[0], case_1_for_padding[1], case_1_for_padding[2]), case_1_for_padding[3])



if __name__ == "__main__":
    flow_1 = np.array([[1, 2], [2, 3]])
    flow_2 = np.array([[10, 11], [20, 21], [30, 31]])
    vecs = [flow_1, flow_2]
    flow_len = 4
    target = padding(vecs, flow_len, value=0)
    ans = np.array([[[ 1,  2], [ 2,  3], [ 0,  0], [ 0,  0]], 
                    [[10, 11], [20, 21], [30, 31], [ 0,  0]]
                    ])
    print(np.array_equal(target, ans))