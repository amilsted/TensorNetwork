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
"""Greedy Contraction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional, Tuple, Collection
import tensornetwork as tn
from tensornetwork import network
from tensornetwork import network_components
from tensornetwork.contractors import cost_calculators

cost_contract_parallel = cost_calculators.cost_contract_parallel


def greedy(nodes: Collection[network_components.BaseNode]
          ) -> network_components.BaseNode:
  """Contract the lowest cost pair of nodes first.
  
  Args:
    node: A set or list of connected nodes.

  Returns:
    The contracted nodes.
  """
  #we don't want nodes to be garbage collected
  nodes_set = {node for node in nodes}
  edges = tn.get_all_nondangling(nodes_set)
  # Seperate out trace edges from non-trace edges
  for edge in edges:
    if not edge.is_disabled:  #if its disabled we already contracted it
      if edge.is_trace():
        nodes_set.remove(edge.node1)
        nodes_set.add(tn.contract_parallel(edge))


# Get the edges again.
  edges = tn.get_all_nondangling(nodes_set)
  while edges:
    edge = min(edges, key=lambda x: (cost_contract_parallel(x), x))
    # replace contracted nodes in nodes_set
    # we need a reference to prevent the edge-nodes from begin garbage
    # collected after removing them from the set
    node1 = edge.node1
    node2 = edge.node2
    nodes_set.remove(node1)
    nodes_set.remove(node2)
    nodes_set.add(tn.contract_parallel(edge))
    edges = tn.get_all_nondangling(nodes_set)
  return nodes_set
