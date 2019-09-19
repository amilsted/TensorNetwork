# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tensornetwork.contractors.naive."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

import tensornetwork as tn
from tensornetwork.contractors import naive_contractor
import tensorflow as tf
tf.enable_v2_behavior()
naive = naive_contractor.naive


def test_sanity_check(backend):
  a = tn.Node(np.eye(2), backend=backend)
  b = tn.Node(np.eye(2), backend=backend)
  c = tn.Node(np.eye(2), backend=backend)
  e1 = tn.connect(a[0], b[1])
  e2 = tn.connect(b[0], c[1])
  e3 = tn.connect(c[0], a[1])
  result = naive([a, b, c], edge_order=[e1, e2, e3])[0]
  np.testing.assert_allclose(result.tensor, 2.0)


def test_passed_edge_order(backend):
  a = tn.Node(np.eye(2), backend=backend)
  b = tn.Node(np.eye(2), backend=backend)
  c = tn.Node(np.eye(2), backend=backend)
  e1 = tn.connect(a[0], b[1])
  e2 = tn.connect(b[0], c[1])
  e3 = tn.connect(c[0], a[1])
  result = naive([a, b, c], [e3, e1, e2])[0]
  np.testing.assert_allclose(result.tensor, 2.0)


def test_bad_passed_edges(backend):
  a = tn.Node(np.eye(2), backend=backend)
  b = tn.Node(np.eye(2), backend=backend)
  c = tn.Node(np.eye(2), backend=backend)
  e1 = tn.connect(a[0], b[1])
  e2 = tn.connect(b[0], c[1])
  _ = tn.connect(c[0], a[1])
  with pytest.raises(ValueError):
    naive([a, b, c], [e1, e2])


def test_precontracted_network(backend):
  a = tn.Node(np.eye(2), backend=backend)
  b = tn.Node(np.eye(2), backend=backend)
  c = tn.Node(np.eye(2), backend=backend)
  e1 = tn.connect(a[0], b[1])
  e2 = tn.connect(b[0], c[1])
  edge = tn.connect(c[0], a[1])
  node = tn.contract(edge)
  # This should work now!
  result = naive([node, b], edge_order=[e1, e2])[0]
  np.testing.assert_allclose(result.tensor, 2.0)
