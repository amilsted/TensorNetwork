# python3
"""Tests for fft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from examples.fft import fft
import tensornetwork


def test_fft():
  n = 3

  initial_state = [complex(0)] * 2**n
  initial_state[1] = 1j
  initial_state[5] = -1
  initial_node = tensornetwork.Node(np.array(initial_state).reshape((2,) * n))

  fft_out = fft.add_fft([initial_node[k] for k in range(n)])
  nodes = tensornetwork.reachable(initial_node)
  tensornetwork.check_correct(nodes)
  result = tensornetwork.contractors.auto(nodes)
  tensornetwork.flatten_edges(fft_out)
  actual = result.get_tensor()
  expected = np.fft.fft(initial_state, norm="ortho")
  assert np.allclose(expected, actual)
