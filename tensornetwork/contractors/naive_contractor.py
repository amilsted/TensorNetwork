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
"""Naive Network Contraction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Sequence, Optional, Collection
import tensornetwork as tn
from tensornetwork import network_components


def naive(
    nodes: Collection[network_components.BaseNode],
    edge_order: Optional[Sequence[network_components.Edge]] = None,
) -> network_components.BaseNode:
  """Contract a network in the order the edges were created.

  This contraction method will usually be very suboptimal unless the edges were
  created in a deliberate way.

  Args:
    node: A collection of connected nodes.
    edge_order: An optional list of non-dangling edges. Must be equal to all non-dangling
      edges in nodes.
  Returns:
    The result of the contraction of all edges.
  Raises:
    ValueError: If the passed `edge_order` list does not contain all of the
      non-dangling edges of the network.
  """
  nodes_set = {node for node in nodes}
  if edge_order is None:
    edge_order = sorted(tn.get_all_nondangling(nodes))
  if not all([not e.is_dangling() for e in edge_order]):
    raise ValueError('not all edges are non-dangling')
  if set(edge_order) != tn.get_all_nondangling(nodes):
    raise ValueError("Set of passed edges does not match expected set."
                     "Given: {}\nExpected: {}".format(
                         edge_order, tn.get_all_nondangling(nodes)))
  for edge in edge_order:
    if not edge.is_disabled:
      #keep node1 and node2 alive
      node1 = edge.node1
      node2 = edge.node2
      nodes_set.remove(node1)
      nodes_set.remove(node2)
      nodes_set.add(tn.contract_parallel(edge))
  return list(nodes_set)
