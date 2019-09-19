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
"""Greedy Contraction Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pytest
from typing import List, Optional, Tuple
from tensornetwork.contractors import greedy_contractor
import tensornetwork as tn


def test_greedy_sanity_check(backend):
  a = tn.Node(np.ones((2, 2, 2, 2, 2)), backend=backend)
  b = tn.Node(np.ones((2, 2, 2)), backend=backend)
  c = tn.Node(np.ones((2, 2, 2)), backend=backend)
  tn.connect(a[0], a[1])
  tn.connect(a[2], b[0])
  tn.connect(a[3], c[0])
  tn.connect(b[1], c[1])
  tn.connect(b[2], c[2])
  node = greedy_contractor.greedy([a, b, c])[0]
  np.testing.assert_allclose(node.tensor, np.ones(2) * 32.0)
