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
"""Contractors based on `opt_einsum`'s path algorithms."""

import functools
import opt_einsum
from tensornetwork import network
from tensornetwork import network_components
from tensornetwork.contractors.opt_einsum_paths import utils
from typing import Any, Optional, Sequence, Collection


def base(nodes: Collection[network_components.BaseNode],
         algorithm: utils.Algorithm,
         output_edge_order: Optional[Sequence[network_components.Edge]] = None
        ) -> Collection[network_components.BaseNode]:
  """Base method for all `opt_einsum` contractors.

  Args:
    nodes: A collection of connected nodes.
    algorithm: `opt_einsum` contraction method to use.
    output_edge_order: An optional list of edges. Edges of the
      final node in `nodes_set`
      are reordered into `output_edge_order`;
      if final node has more than one edge,
      `output_edge_order` must be provided.

  Returns:
    Final node after full contraction.
  """
  network.check_connected(nodes)
  nodes_set = {node for node in nodes}
  edges = network.get_all_nondangling(nodes_set)
  #output edge order has to be determinded before any contraction
  #(edges are refreshed after contractions)
  if output_edge_order is None:
    output_edge_order = list(
        (network.get_all_edges(nodes) - network.get_all_nondangling(nodes)))

  if set(output_edge_order) != (
      network.get_all_edges(nodes) - network.get_all_nondangling(nodes)):
    raise ValueError("output edges are not all dangling.")

  for edge in edges:
    if not edge.is_disabled:  #if its disabled we already contracted it
      if edge.is_trace():
        nodes_set.remove(edge.node1)
        nodes_set.add(network.contract_parallel(edge))

  if not network.get_all_nondangling(nodes_set):
    # There's nothing to contract.
    return list(nodes_set)[0]

  # Then apply `opt_einsum`'s algorithm
  path, nodes = utils.get_path(nodes_set, algorithm)
  for a, b in path:
    new_node = nodes[a] @ nodes[b]
    nodes.append(new_node)
    nodes = utils.multi_remove(nodes, [a, b])

  # if the final node has more than one edge,
  # output_edge_order has to be specified
  final_node = nodes[0]  # nodes were connected, we checked this
  if (len(final_node.edges) > 1) and (output_edge_order is None):
    raise ValueError("if the final node has more than one dangling edge"
                     " `output_edge_order` has to be provided")

  final_node.reorder_edges(output_edge_order)
  return final_node


def optimal(nodes: Collection[network_components.BaseNode],
            output_edge_order: Sequence[network_components.Edge] = None,
            memory_limit: Optional[int] = None) -> network_components.BaseNode:
  """Optimal contraction order via `opt_einsum`.

  This method will find the truly optimal contraction order via
  `opt_einsum`'s depth first search algorithm. Since this search is
  exhaustive, if your network is large (n>10), then the search may
  take longer than just contracting in a suboptimal way.

  Args:
    nodes: A collection of connected nodes.
    output_edge_order: An optional list of edges.
      Edges of the final node in `nodes_set`
      are reordered into `output_edge_order`;
      if final node has more than one edge,
      `output_edge_order` must be provided.
    memory_limit: Maximum number of elements in an array during contractions.

  Returns:
    Final node after full contraction.
  """
  alg = functools.partial(opt_einsum.paths.optimal, memory_limit=memory_limit)
  return base(nodes, alg, output_edge_order)


def branch(nodes: Collection[network_components.BaseNode],
           output_edge_order: Sequence[network_components.Edge] = None,
           memory_limit: Optional[int] = None,
           nbranch: Optional[int] = None) -> network_components.BaseNode:
  """Branch contraction path via `opt_einsum`.

  This method uses the DFS approach of `optimal` while sorting potential
  contractions based on a heuristic cost, in order to reduce time spent
  in exploring paths which are unlikely to be optimal.
  For more details:
    https://optimized-einsum.readthedocs.io/en/latest/branching_path.html

  Args:
    nodes: A collection of connected nodes.  
    output_edge_order: An optional list of edges.
      Edges of the final node in `nodes_set`
       are reordered into `output_edge_order`;
       if final node has more than one edge,
       `output_edge_order` must be provided.
    memory_limit: Maximum number of elements in an array during contractions.
    nbranch: Number of best contractions to explore.
      If None it explores all inner products starting with those that
      have the best cost heuristic.

  Returns:
    Final node after full contraction.
  """
  alg = functools.partial(
      opt_einsum.paths.branch, memory_limit=memory_limit, nbranch=nbranch)
  return base(nodes, alg, output_edge_order)


def greedy(nodes: Collection[network_components.BaseNode],
           output_edge_order: Sequence[network_components.Edge] = None,
           memory_limit: Optional[int] = None) -> network_components.BaseNode:
  """Greedy contraction path via `opt_einsum`.

  This provides a more efficient strategy than `optimal` for finding
  contraction paths in large networks. First contracts pairs of tensors
  by finding the pair with the lowest cost at each step. Then it performs
  the outer products.
  For more details:
    https://optimized-einsum.readthedocs.io/en/latest/greedy_path.html

  Args:
    nodes: A collection of connected nodes.
    output_edge_order: An optional list of edges.
      Edges of the final node in `nodes_set`
      are reordered into `output_edge_order`;
      if final node has more than one edge,
      `output_edge_order` must be provided.
    memory_limit: Maximum number of elements in an array during contractions.

  Returns:
    Final node after full contraction.
  """
  alg = functools.partial(opt_einsum.paths.greedy, memory_limit=memory_limit)
  return base(nodes, alg, output_edge_order)


def auto(nodes: Collection[network_components.BaseNode],
         output_edge_order: Sequence[network_components.Edge] = None,
         memory_limit: Optional[int] = None) -> network_components.BaseNode:
  """Chooses one of the above algorithms according to network size.

  Default behavior is based on `opt_einsum`'s `auto` contractor.

  Args:
    nodes: A collection of connected nodes.
    output_edge_order: An optional list of edges.
      Edges of the final node in `nodes_set`
      are reordered into `output_edge_order`;
      if final node has more than one edge,
      `output_edge_order` must be provided.
    memory_limit: Maximum number of elements in an array during contractions.

  Returns:
    Final node after full contraction.
  """
  n = len(nodes)
  if n <= 0:
    raise ValueError("Cannot contract empty tensor network.")
  if n == 1:
    edges = network.get_all_nondangling(nodes)

    if output_edge_order is None:
      output_edge_order = list(
          (network.get_all_edges(nodes) - network.get_all_nondangling(nodes)))

    final_node = network.contract_parallel(edges.pop())
    final_node.reorder_edges(output_edge_order)
    return final_node
  if n < 5:
    return optimal(nodes, output_edge_order, memory_limit)
  if n < 7:
    return branch(nodes, output_edge_order, memory_limit)
  if n < 9:
    return branch(nodes, output_edge_order, memory_limit, nbranch=2)
  if n < 15:
    return branch(nodes, output_edge_order, nbranch=1)
  return greedy(nodes, output_edge_order, memory_limit)


def custom(nodes: Collection[network_components.BaseNode],
           optimizer: Any,
           output_edge_order: Sequence[network_components.Edge] = None,
           memory_limit: Optional[int] = None) -> network_components.BaseNode:
  """Uses a custom path optimizer created by the user to calculate paths.

  The custom path optimizer should inherit `opt_einsum`'s `PathOptimizer`.
  For more details:
    https://optimized-einsum.readthedocs.io/en/latest/custom_paths.html

  Args:
    nodes: A collection of connected nodes.
    output_edge_order: An optional list of edges.
      Edges of the final node in `nodes_set`
      are reordered into `output_edge_order`;
      if final node has more than one edge,
      output_edge_order` must be provided.
    optimizer: A custom `opt_einsum.PathOptimizer` object.
    memory_limit: Maximum number of elements in an array during contractions.

  Returns:
    Final node after full contraction.
  """
  alg = functools.partial(optimizer, memory_limit=memory_limit)
  return base(nodes, alg, output_edge_order)
