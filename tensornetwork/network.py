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
"""Implementation of TensorNetwork structure."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import h5py
# pylint: disable=line-too-long
from typing import Any, Sequence, List, Set, Optional, Union, Text, Tuple, Type, Dict, BinaryIO, Collection
import numpy as np
import weakref
from tensornetwork import config
from tensornetwork import network_components
from tensornetwork.backends import backend_factory

Tensor = Any
string_type = h5py.special_dtype(vlen=str)


def get_shared_edges(
    node1: network_components.BaseNode,
    node2: network_components.BaseNode) -> Set[network_components.Edge]:
  """Get all edges shared between two nodes.

  Args:
    node1: The first node.
    node2: The second node.

  Returns:
    A (possibly empty) `set` of `Edge`s shared by the nodes.
  """
  nodes = {node1, node2}
  shared_edges = set()
  # Assuming the network is well formed, all of the edges shared by
  # these two nodes will be stored in just one of the nodes, so we only
  # have to do this loop once.
  for edge in node1.edges:
    if set(edge.get_nodes()) == nodes:
      shared_edges.add(edge)
  return shared_edges


def _flatten_trace_edges(
    edges: List[network_components.Edge],
    backend: "Backend",
    new_edge_name: Optional[Text] = None) -> network_components.Edge:
  """Flatten trace edges into single edge.

  Args:
    edges: List of trace edges to flatten
    backend: A backend object
    new_edge_name: Optional name of the new edge created.

  Returns:
    The new edge that represents the flattening of the given edges.
  """
  node = edges[0].node1  # We are in the trace case, so this is the only node.
  # Flatten all of the edge's axes into a a single list.
  perm_back = [min(e.axis1, e.axis2) for e in edges]
  perm_back += [max(e.axis1, e.axis2) for e in edges]
  perm_front = set(range(len(node.edges))) - set(perm_back)
  perm_front = sorted(perm_front)
  perm = perm_front + perm_back
  new_dim = backend.prod([backend.shape(node.tensor)[e.axis1] for e in edges])
  node.reorder_axes(perm)
  unaffected_shape = backend.shape(node.tensor)[:len(perm_front)]
  new_shape = backend.concat([unaffected_shape, [new_dim, new_dim]], axis=-1)
  node.tensor = backend.reshape(node.tensor, new_shape)
  edge1 = network_components.Edge(
      node1=node, axis1=len(perm_front), name="TraceFront")
  edge2 = network_components.Edge(
      node1=node, axis1=len(perm_front) + 1, name="TraceBack")
  node.edges = node.edges[:len(perm_front)] + [edge1, edge2]
  new_edge = connect(edge1, edge2, new_edge_name)
  [edge.disable() for edge in edges]  #disable edges!
  return new_edge


def flatten_edges(edges: List[network_components.Edge],
                  new_edge_name: Optional[Text] = None
                 ) -> network_components.Edge:
  """Flatten edges into single edge.

  If two nodes have multiple edges connecting them, it may be
  benifitial to flatten these edges into a single edge to avoid having several
  unnecessary trace edges. This can speed up computation time and reduce
  memory cost.

  Warning: This will remove all axes names.

  Args:
    edges: A list of edges to flatten.
    backend: A backend object
    new_edge_name: Optional name to give to the newly created edge.

  Returns:
    The new flattened edge.

  Raises:
    ValueError: If edges is an empty list.
    ValueError: If not all of the edges connect to the same node(s).
    ValueError: If one of the nodes connecting to these edges does not have
      edge definitions for all of its axes.
  """
  backends = [edge.node1.backend for edge in edges] + [
      edge.node2.backend for edge in edges if edge.node2 is not None
  ]

  if not all([b.name == backends[0].name for b in backends]):
    raise ValueError("Not all backends are the same.")
  backend = backends[0]
  if not edges:
    raise ValueError("At least 1 edge must be given.")
  if len(edges) == 1:
    return edges[0]  # Don't bother with reshaping.
  # Set equality is transitive (a=b, b=c, therefore a=c) so it is only
  # necessary to compare the first edge against the rest.
  expected_nodes = set(edges[0].get_nodes())
  for edge in edges:
    if expected_nodes != set(edge.get_nodes()):
      raise ValueError(
          "Two edges do not share the same nodes. "
          "'{}'s nodes: '{}', '{}'. '{}'s nodes: '{}', '{}'".format(
              edges[0], edges[0].node1, edges[0].node2, edge, edge.node1,
              edge.node2))
  if len(expected_nodes) == 1:
    return _flatten_trace_edges(edges, backend, new_edge_name)  #disables edges
  # Flatten standard or dangling edges.
  new_dangling_edges = []
  for node in expected_nodes:
    # Required for dangling case.
    if node is None:
      continue
    perm_back = []
    for edge in edges:
      # There will only be 1 edge since we are in the standard edge case.
      perm_back.append(node.edges.index(edge))
    perm_front = sorted(set(range(len(node.edges))) - set(perm_back))
    node.reorder_axes(perm_front + perm_back)
    old_tensor_shape = backend.shape(node.tensor)
    # Calculate the new axis dimension as a product of the other
    # axes dimensions.
    flattened_axis_dim = backend.prod(old_tensor_shape[len(perm_front):])
    new_tensor_shape = backend.concat(
        [old_tensor_shape[:len(perm_front)], [flattened_axis_dim]], axis=-1)
    new_tensor = backend.reshape(node.tensor, new_tensor_shape)
    # Modify the node in place. Currently, this is they only method that
    # modifies a node's tensor.
    node.tensor = new_tensor
    # This Edge is required for the connect call later.
    edge = network_components.Edge(
        node1=node, axis1=len(perm_front), name=new_edge_name)
    # Do not set the signature of 'edge' since it is dangling.
    node.edges = node.edges[:len(perm_front)] + [edge]
    new_dangling_edges.append(edge)
    # TODO: Allow renaming of the new axis.

  node1, node2 = tuple(expected_nodes)
  # Sets are returned in a random order, so this is how we deal with
  # dangling edges.
  [edge.disable() for edge in edges]  #disable edges!
  if node1 is None or node2 is None:
    return new_dangling_edges[0]

  return connect(new_dangling_edges[0], new_dangling_edges[1], new_edge_name)


def flatten_edges_between(
    node1: network_components.BaseNode,
    node2: network_components.BaseNode,
) -> Optional[network_components.Edge]:
  """Flatten all of the edges between the given two nodes.

  Args:
    node1: The first node.
    node2: The second node.

  Returns:
    The flattened `Edge` object. If there was only one edge between the two
      nodes, then the original edge is returned. If there where no edges
      between the nodes, a None is returned.
  """
  shared_edges = get_shared_edges(node1, node2)
  if shared_edges:
    return flatten_edges(list(shared_edges))
  return None


def _remove_trace_edge(edge: network_components.Edge,
                       new_node: network_components.BaseNode) -> None:
  """Collapse a trace edge. `edge` is disabled before returning.

  Take a trace edge (i.e. with edge.node1 = edge.node2),
  remove it, update the axis numbers of all remaining edges
  and move them to `new_node`.

  Args:
    edge: The edge to contract.
    new_node: The new node created after contraction.

  Returns:
    None

  Raises:
    ValueError: If edge is not a trace edge.
  """
  if edge.is_dangling():
    raise ValueError("Attempted to remove dangling edge '{}'.".format(edge))
  if edge.node1 is not edge.node2:
    raise ValueError("Edge '{}' is not a trace edge.".format(edge))
  axes = sorted([edge.axis1, edge.axis2])
  node_edges = edge.node1.edges[:]
  node_edges.pop(axes[0])
  node_edges.pop(axes[1] - 1)
  seen_edges = set()
  for tmp_edge in node_edges:
    if tmp_edge in seen_edges:
      continue
    else:
      seen_edges.add(tmp_edge)
    if tmp_edge.node1 is edge.node1:
      to_reduce = 0
      to_reduce += 1 if tmp_edge.axis1 > axes[0] else 0
      to_reduce += 1 if tmp_edge.axis1 > axes[1] else 0
      tmp_edge.axis1 -= to_reduce
      tmp_edge.node1 = new_node
    if tmp_edge.node2 is edge.node1:
      to_reduce = 0
      to_reduce += 1 if tmp_edge.axis2 > axes[0] else 0
      to_reduce += 1 if tmp_edge.axis2 > axes[1] else 0
      tmp_edge.axis2 -= to_reduce
      tmp_edge.node2 = new_node
  # Update edges for the new node.
  for i, e in enumerate(node_edges):
    new_node.add_edge(e, i)
  edge.node1.fresh_edges(edge.node1.axis_names)
  edge.disable()  #disabled edge!


def _remove_edges(edges: Set[network_components.Edge],
                  node1: network_components.BaseNode,
                  node2: network_components.BaseNode,
                  new_node: network_components.BaseNode) -> None:
  """

  Takes a set of `edges` shared between `node1` and `node2` to be contracted 
  over, and moves all other uncontracted edges from `node1` and `node2` to 
  `new_node`.
  The nodes that currently share the edges in `edges` must be supplied as
  `node1` and `node2`. The ordering of `node1` and `node2` must match the
  axis ordering of `new_node` (as determined by the contraction procedure).
  `node1` and `node2` get both a fresh set edges.
  `edges` are disabled before returning.
  Args:
    edges: The edges to contract.
    node1: The old node that supplies the first edges of `new_node`.
    node2: The old node that supplies the last edges of `new_node`.
    new_node: The new node that represents the contraction of the two old
      nodes.
  Returns:
    node1, node2L
  Raises:
    Value Error: If edge isn't in the network.
  """
  if node1 is node2:
    raise ValueError(
        "node1 and node2 are the same ('{}' == '{}'), but trace edges cannot "
        "be removed by _remove_edges.".format(node1, node2))

  node1_edges = node1.edges[:]
  node2_edges = node2.edges[:]

  nodes_set = set([node1, node2])
  for edge in edges:
    if edge.is_dangling():
      raise ValueError("Attempted to remove dangling edge '{}'.".format(edge))
    if set([edge.node1, edge.node2]) != nodes_set:
      raise ValueError(
          "Attempted to remove edges belonging to different node pairs: "
          "'{}' != '{}'.".format(nodes_set, set([edge.node1, edge.node2])))

  node1_axis_names = node1.axis_names
  node2_axis_names = node2.axis_names

  remaining_edges = []
  for (i, edge) in enumerate(node1_edges):
    if edge not in edges:  # NOTE: Makes the cost quadratic in # edges
      edge.update_axis(
          old_node=node1,
          old_axis=i,
          new_axis=len(remaining_edges),
          new_node=new_node)
      remaining_edges.append(edge)

  for (i, edge) in enumerate(node2_edges):
    if edge not in edges:
      edge.update_axis(
          old_node=node2,
          old_axis=i,
          new_axis=len(remaining_edges),
          new_node=new_node)
      remaining_edges.append(edge)

  for (i, edge) in enumerate(remaining_edges):
    new_node.add_edge(edge, i)

  node1.fresh_edges(node1_axis_names)
  node2.fresh_edges(node2_axis_names)
  [edge.disable() for edge in edges]  #disabled edges!


def _contract_trace(edge: network_components.Edge,
                    name: Optional[Text] = None) -> network_components.BaseNode:
  """Contract a trace edge.
  `edge` is disabled before returning.
  Args:
    edge: The edge name or object to contract next.
    net: An optional TensorNetwork
    name: Name to give to the new node. If None, a name will automatically be
      generated.

  Returns:
    The new node created after the contraction.

  Raise:
    ValueError: When edge is a dangling edge.
  """
  if edge.is_dangling():
    raise ValueError("Attempted to contract dangling edge '{}'".format(edge))
  if edge.node1 is not edge.node2:
    raise ValueError("Can not take trace of edge '{}'. This edge connects to "
                     "two different nodes: '{}' and '{}".format(
                         edge, edge.node1, edge.node2))
  backend = edge.node1.backend
  axes = sorted([edge.axis1, edge.axis2])
  dims = len(edge.node1.tensor.shape)
  permutation = sorted(set(range(dims)) - set(axes)) + axes
  new_tensor = backend.trace(
      backend.transpose(edge.node1.tensor, perm=permutation))

  new_node = network_components.Node(
      new_tensor, name=name, backend=backend.name)
  _remove_trace_edge(edge, new_node)  #disables edge
  return new_node


def contract(
    edge: network_components.Edge,
    name: Optional[Text] = None,
    axis_names: Optional[List[Text]] = None) -> network_components.BaseNode:
  """
  Contract an edge connecting two nodes in the TensorNetwork.
  All edges of `node1` and `node2` are passed on to the new node, 
  and `node1` and `node2` get a new set of dangling edges.
  `edge is disabled before returning.
  Args:
    edge: The edge contract next.
    name: Name of the new node created.

  Returns:
    The new node created after the contraction.

  Raises:
    ValueError: When edge is a dangling edge or if it already has been
      contracted.
  """
  for node in [edge.node1, edge.node2]:
    if not hasattr(node, 'backend'):
      raise TypeError('Node {} of type {} has no `backend`'.format(
          node, type(node)))

  if edge.node1:
    backend = edge.node1.backend
  else:
    raise ValueError("edge {} has no nodes. "
                     "Cannot perfrom a contraction".format(edge.name))

  if edge.is_dangling():
    raise ValueError("Attempting to contract dangling edge")
  backend = edge.node1.backend
  if edge.node1 is edge.node2:
    return _contract_trace(edge, name)
  new_tensor = backend.tensordot(edge.node1.tensor, edge.node2.tensor,
                                 [[edge.axis1], [edge.axis2]])
  new_node = network_components.Node(
      tensor=new_tensor, name=name, axis_names=axis_names, backend=backend.name)
  # edge.node1 and edge.node2 get new edges in _remove_edges
  _remove_edges(set([edge]), edge.node1, edge.node2, new_node)
  return new_node


def outer_product(
    node1: network_components.BaseNode,
    node2: network_components.BaseNode,
    name: Optional[Text] = None,
    axis_names: Optional[List[Text]] = None) -> network_components.BaseNode:
  """Calculates an outer product of the two nodes.

  This causes the nodes to combine their edges and axes, so the shapes are
  combined. For example, if `a` had a shape (2, 3) and `b` had a shape
  (4, 5, 6), then the node `net.outer_product(a, b) will have shape
  (2, 3, 4, 5, 6). All edges of `node1` and `node2` are passed on to
  the new node, and `node1` and `node2` get a new set of dangling edges.

  Args:
    node1: The first node. The axes on this node will be on the left side of
      the new node.
    node2: The second node. The axes on this node will be on the right side of
      the new node.
    name: Optional name to give the new node created.
    axis_names: An optional list of names for the axis of the new node
  Returns:
    A new node. Its shape will be node1.shape + node2.shape
  Raises:
    TypeError: If `node1` and `node2` have wrong types.
    ValueError: If `Node` objects `node1` or `node2` have no 
      `TensorNetwork` object.
  """
  for node in [node1, node2]:
    if not hasattr(node, 'backend'):
      raise TypeError('Node {} of type {} has no `backend`'.format(
          node, type(node)))

  if node1.backend.name != node2.backend.name:
    raise ValueError("node {}  and node {} have different backends. "
                     "Cannot perform outer product".format(node1, node2))

  backend = node1.backend
  new_tensor = backend.outer_product(node1.tensor, node2.tensor)
  node1_axis_names = node1.axis_names
  node2_axis_names = node2.axis_names
  new_node = network_components.Node(
      tensor=new_tensor, name=name, axis_names=axis_names, backend=backend.name)
  additional_axes = len(node1.tensor.shape)

  for i, edge in enumerate(node1.edges):
    edge.update_axis(i, node1, i, new_node)
  for i, edge in enumerate(node2.edges):
    edge.update_axis(i, node2, i + additional_axes, new_node)

  for i, edge in enumerate(node1.edges + node2.edges):
    new_node.add_edge(edge, i, True)

  node1.fresh_edges(node1_axis_names)
  node2.fresh_edges(node2_axis_names)

  return new_node


def contract_copy_node(copy_node: network_components.CopyNode,
                       name: Optional[Text] = None
                      ) -> network_components.BaseNode:
  """Contract all edges incident on given copy node.

  Args:
    copy_node: Copy tensor node to be contracted.
    name: Name of the new node created.

  Returns:
    New node representing contracted tensor.

  Raises:
    ValueError: If copy_node has dangling edge(s).
  """
  new_tensor = copy_node.compute_contracted_tensor()
  new_node = network_components.Node(
      new_tensor, name, backend=copy_node.backend.name)

  partners = copy_node.get_partners()
  new_axis = 0
  for partner in partners:
    for edge in partner.edges:
      if edge.node1 is copy_node or edge.node2 is copy_node:
        continue
      old_axis = edge.axis1 if edge.node1 is partner else edge.axis2
      edge.update_axis(
          old_node=partner,
          old_axis=old_axis,
          new_node=new_node,
          new_axis=new_axis)
      new_node.add_edge(edge, new_axis)
      new_axis += 1
  assert len(new_tensor.shape) == new_axis
  copy_node.fresh_edges(copy_node.axis_names)
  return new_node


def contract_between(
    node1: network_components.BaseNode,
    node2: network_components.BaseNode,
    name: Optional[Text] = None,
    allow_outer_product: bool = False,
    output_edge_order: Optional[Sequence[network_components.Edge]] = None,
    axis_names: Optional[List[Text]] = None,
) -> network_components.BaseNode:
  """Contract all of the edges between the two given nodes.

  Args:
    node1: The first node.
    node2: The second node.
    name: Name to give to the new node created.
    allow_outer_product: Optional boolean. If two nodes do not share any edges
      and `allow_outer_product` is set to `True`, then we return the outer
      product of the two nodes. Else, we raise a `ValueError`.
    output_edge_order: Optional sequence of Edges. When not `None`, must
      contain all edges belonging to, but not shared by `node1` and `node2`.
      The axes of the new node will be permuted (if necessary) to match this
      ordering of Edges.
    axis_names: An optional list of names for the axis of the new node
  Returns:
    The new node created.

  Raises:
    ValueError: If no edges are found between node1 and node2 and
      `allow_outer_product` is set to `False`.
  """
  for node in [node1, node2]:
    if not hasattr(node, 'backend'):
      raise TypeError('Node {} of type {} has no `backend`'.format(
          node, type(node)))

  if node1.backend.name != node2.backend.name:
    raise ValueError("node {}  and node {} have different backends. "
                     "Cannot perform outer product".format(node1, node2))

  backend = node1.backend
  # Trace edges cannot be contracted using tensordot.
  if node1 is node2:
    flat_edge = flatten_edges_between(node1, node2)
    if not flat_edge:
      raise ValueError("No trace edges found on contraction of edges between "
                       "node '{}' and itself.".format(node1))
    return contract(flat_edge, name)

  shared_edges = get_shared_edges(node1, node2)
  if not shared_edges:
    if allow_outer_product:
      return outer_product(node1, node2, backend, net)
    raise ValueError("No edges found between nodes '{}' and '{}' "
                     "and allow_outer_product=False.".format(node1, node2))

  # Collect the axis of each node corresponding to each edge, in order.
  # This specifies the contraction for tensordot.
  # NOTE: The ordering of node references in each contraction edge is ignored.
  axes1 = []
  axes2 = []
  for edge in shared_edges:
    if edge.node1 is node1:
      axes1.append(edge.axis1)
      axes2.append(edge.axis2)
    else:
      axes1.append(edge.axis2)
      axes2.append(edge.axis1)

  if output_edge_order:
    # Determine heuristically if output transposition can be minimized by
    # flipping the arguments to tensordot.
    node1_output_axes = []
    node2_output_axes = []
    for (i, edge) in enumerate(output_edge_order):
      if edge in shared_edges:
        raise ValueError(
            "Edge '{}' in output_edge_order is shared by the nodes to be "
            "contracted: '{}' and '{}'.".format(edge, node1, node2))
      edge_nodes = set(edge.get_nodes())
      if node1 in edge_nodes:
        node1_output_axes.append(i)
      elif node2 in edge_nodes:
        node2_output_axes.append(i)
      else:
        raise ValueError(
            "Edge '{}' in output_edge_order is not connected to node '{}' or "
            "node '{}'".format(edge, node1, node2))
    if np.mean(node1_output_axes) > np.mean(node2_output_axes):
      node1, node2 = node2, node1
      axes1, axes2 = axes2, axes1

  new_tensor = backend.tensordot(node1.tensor, node2.tensor, [axes1, axes2])
  new_node = network_components.Node(
      tensor=new_tensor, name=name, axis_names=axis_names, backend=backend.name)
  # node1 and node2 get new edges in _remove_edges
  _remove_edges(shared_edges, node1, node2, new_node)
  if output_edge_order:
    new_node = new_node.reorder_edges(list(output_edge_order))
  return new_node


def contract_parallel(
    edge: network_components.Edge) -> network_components.BaseNode:
  """Contract all edges parallel to this edge.

    This method calls `contract_between` with the nodes connected by the edge.

    Args:
      edge: The edge to contract.

    Returns:
      The new node created after contraction.
    """
  if edge.is_dangling():
    raise ValueError("Attempted to contract dangling edge: '{}'".format(edge))
  return contract_between(edge.node1, edge.node2)


def connect(edge1: network_components.Edge,
            edge2: network_components.Edge,
            name: Optional[Text] = None) -> network_components.Edge:
  for edge in [edge1, edge2]:
    if not edge.is_dangling():
      raise ValueError("Edge '{}' is not a dangling edge. "
                       "This edge points to nodes: '{}' and '{}'".format(
                           edge, edge.node1, edge.node2))
  if edge1 is edge2:
    raise ValueError("Cannot connect and edge '{}' to itself.".format(edge1))

  if edge1.dimension != edge2.dimension:
    raise ValueError("Cannot connect edges of unequal dimension. "
                     "Dimension of edge '{}': {}, "
                     "Dimension of edge '{}': {}.".format(
                         edge1, edge2.dimension, edge2, edge2.dimension))

  #edge1 and edg2 are always dangling in this case
  node1 = edge1.node1
  node2 = edge2.node1
  axis1_num = node1.get_axis_number(edge1.axis1)
  axis2_num = node2.get_axis_number(edge2.axis1)

  #we have to check if node1[axis1_num] is already connected to
  #another node or not. If yes, we break the connection
  #and update node1 and the other node
  # if not node1.edges[axis1_num].is_dangling():
  #   node1_partner = node1.edges[axis1_num].node1 if node1.edges[
  #       axis1_num].node1 is not node1 else node1.edges[axis1_num].node2
  #   node1_partner_axis_num = node1.edges[axis1_num].axis1 if node1.edges[
  #       axis1_num].node1 is not node1 else node1.edges[axis1_num].axis2

  #   new_dangling_edge = network_components.Edge(
  #       node1_partner.edges[node1_partner_axis_num].name, node1_partner,
  #       node1_partner_axis_num)
  #   node1_partner.add_edge(new_dangling_edge, node1_partner_axis_num, True)

  # if not node2.edges[axis2_num].is_dangling():
  #   node2_partner = node2.edges[axis2_num].node1 if node2.edges[
  #       axis2_num].node1 is not node2 else node2.edges[axis2_num].node2
  #   node2_partner_axis_num = node2.edges[axis2_num].axis1 if node2.edges[
  #       axis2_num].node1 is not node2 else node2.edges[axis2_num].axis2

  #   new_dangling_edge = network_components.Edge(
  #       node2_partner.edges[node2_partner_axis_num].name, node2_partner,
  #       node2_partner_axis_num)
  #   node2_partner.add_edge(new_dangling_edge, node2_partner_axis_num, True)

  new_edge = network_components.Edge(
      node1=node1, axis1=axis1_num, name=name, node2=node2, axis2=axis2_num)

  node1.add_edge(new_edge, axis1_num, override=True)
  node2.add_edge(new_edge, axis2_num, override=True)
  return new_edge


def disconnect(
    edge1, edge2, edge1_name: Optional[Text], edge2_name: Optional[Text]
) -> Tuple[network_components.BaseNode, network_components.BaseNode]:
  """
  Break an existing non-dangling edge.
  This updates both Edge.node1 and Edge.node2 by removing the 
  connecting edge from `Edge.node1.edges` and `Edge.node2.edges`
  and adding new dangling edges instead
  """
  if edge1 is not edge2:
    raise ValueError("edge1 and edge2 are not connected. "
                     "Cannot disconnect unconnected edges")
  edge.disconnect(edge1_name, edge2_name)


def conj(
    node: network_components.BaseNode,
    name: Optional[Text] = None,
    axis_names: Optional[List[Text]] = None) -> network_components.BaseNode:
  """Conjugate `node`
  Args:
    node: A `BaseNode`. 
    name: Optional name to give the new node.
    axis_names: Optional list of names for the axis.
  Returns:
    A new node. The complex conjugate of `node`.
  Raises:
    TypeError: If `node` has no `backend` attribute.
  """
  if not hasattr(node, 'backend'):
    raise TypeError('Node {} of type {} has no `backend`'.format(
        node, type(node)))
  backend = node.backend
  if not axis_names:
    axis_names = node.axis_names

  return network_components.Node(
      backend.conj(node.tensor),
      name=name,
      axis_names=axis_names,
      backend=backend.name)


def transpose(
    node: network_components.BaseNode,
    permutation: Sequence[Union[Text, int]],
    name: Optional[Text] = None,
    axis_names: Optional[List[Text]] = None) -> network_components.BaseNode:
  """Transpose `node`
  Args:
    node: A `BaseNode`. 
    permutation: A list of int ro str. The permutation of the axis
    name: Optional name to give the new node.
    axis_names: Optional list of names for the axis.
  Returns:
    A new node. The transpose of `node`.
  Raises:
    TypeError: If `node` has no `backend` attribute.
    ValueError: If either `permutation` is not the same as expected or
      if you try to permute with a trace edge.
    AttributeError: If `node` has no tensor.
  """

  if not hasattr(node, 'backend'):
    raise TypeError('Node {} of type {} has no `backend`'.format(
        node, type(node)))
  if not axis_names:
    axis_names = node.axis_names
  new_node = network_components.Node(
      node.tensor,
      name=name,
      axis_names=node.axis_names,
      backend=node.backend.name)
  return new_node.reorder_axes(permutation)


def split_node(
    node: network_components.BaseNode,
    left_edges: List[network_components.Edge],
    right_edges: List[network_components.Edge],
    max_singular_values: Optional[int] = None,
    max_truncation_err: Optional[float] = None,
    left_name: Optional[Text] = None,
    right_name: Optional[Text] = None,
    edge_name: Optional[Text] = None,
) -> Tuple[network_components.BaseNode, network_components.BaseNode, Tensor]:
  """Split a `Node` using Singular Value Decomposition.

  Let M be the matrix created by flattening left_edges and right_edges into
  2 axes. Let :math:`U S V^* = M` be the Singular Value Decomposition of 
  :math:`M`. This will split the network into 2 nodes. The left node's 
  tensor will be :math:`U \\sqrt{S}` and the right node's tensor will be 
  :math:`\\sqrt{S} V^*` where :math:`V^*` is
  the adjoint of :math:`V`.

  The singular value decomposition is truncated if `max_singular_values` or
  `max_truncation_err` is not `None`.

  The truncation error is the 2-norm of the vector of truncated singular
  values. If only `max_truncation_err` is set, as many singular values will
  be truncated as possible while maintaining:
  `norm(truncated_singular_values) <= max_truncation_err`.

  If only `max_singular_values` is set, the number of singular values kept
  will be `min(max_singular_values, number_of_singular_values)`, so that
  `max(0, number_of_singular_values - max_singular_values)` are truncated.

  If both `max_truncation_err` and `max_singular_values` are set,
  `max_singular_values` takes priority: The truncation error may be larger
  than `max_truncation_err` if required to satisfy `max_singular_values`.

  Args:
    node: The node you want to split.
    left_edges: The edges you want connected to the new left node.
    right_edges: The edges you want connected to the new right node.
    max_singular_values: The maximum number of singular values to keep.
    max_truncation_err: The maximum allowed truncation error.
    left_name: The name of the new left node. If `None`, a name will be generated
      automatically.
    right_name: The name of the new right node. If `None`, a name will be generated
      automatically.
    edge_name: The name of the new `Edge` connecting the new left and right node. 
      If `None`, a name will be generated automatically. The new axis will get the
      same name as the edge.

  Returns:
    A tuple containing:
      left_node: 
        A new node created that connects to all of the `left_edges`.
        Its underlying tensor is :math:`U \\sqrt{S}`
      right_node: 
        A new node created that connects to all of the `right_edges`.
        Its underlying tensor is :math:`\\sqrt{S} V^*`
      truncated_singular_values: 
        The vector of truncated singular values.
  """

  if not hasattr(node, 'backend'):
    raise TypeError('Node {} of type {} has no `backend`'.format(
        node, type(node)))

  if node.axis_names and edge_name:
    left_axis_names = []
    right_axis_names = [edge_name]
    for edge in left_edges:
      left_axis_names.append(node.axis_names[edge.axis1] if edge.node1 is node
                             else node.axis_names[edge.axis2])
    for edge in right_edges:
      right_axis_names.append(node.axis_names[edge.axis1] if edge.node1 is node
                              else node.axis_names[edge.axis2])
    left_axis_names.append(edge_name)
  else:
    left_axis_names = None
    right_axis_names = None

  backend = node.backend
  node.reorder_edges(left_edges + right_edges)

  u, s, vh, trun_vals = backend.svd_decomposition(
      node.tensor, len(left_edges), max_singular_values, max_truncation_err)
  sqrt_s = backend.sqrt(s)
  u_s = u * sqrt_s
  # We have to do this since we are doing element-wise multiplication against
  # the first axis of vh. If we don't, it's possible one of the other axes of
  # vh will be the same size as sqrt_s and would multiply across that axis
  # instead, which is bad.
  sqrt_s_broadcast_shape = backend.concat(
      [backend.shape(sqrt_s), [1] * (len(vh.shape) - 1)], axis=-1)
  vh_s = vh * backend.reshape(sqrt_s, sqrt_s_broadcast_shape)
  left_node = network_components.Node(
      u_s, name=left_name, axis_names=left_axis_names, backend=backend.name)
  for i, edge in enumerate(left_edges):
    left_node.add_edge(edge, i)
    edge.update_axis(i, node, i, left_node)
  right_node = network_components.Node(
      vh_s, name=right_name, axis_names=right_axis_names, backend=backend.name)
  for i, edge in enumerate(right_edges):
    # i + 1 to account for the new edge.
    right_node.add_edge(edge, i + 1)
    edge.update_axis(i + len(left_edges), node, i + 1, right_node)
  connect(left_node.edges[-1], right_node.edges[0], name=edge_name)
  node.fresh_edges(node.axis_names)
  return left_node, right_node, trun_vals


def split_node_qr(
    node: network_components.BaseNode,
    left_edges: List[network_components.Edge],
    right_edges: List[network_components.Edge],
    left_name: Optional[Text] = None,
    right_name: Optional[Text] = None,
    edge_name: Optional[Text] = None,
) -> Tuple[network_components.BaseNode, network_components.BaseNode]:
  """Split a `Node` using QR decomposition

  Let M be the matrix created by flattening left_edges and right_edges into
  2 axes. Let :math:`QR = M` be the QR Decomposition of
  :math:`M`. This will split the network into 2 nodes. The left node's
  tensor will be :math:`Q` (an orthonormal matrix) and the right node's tensor will be
  :math:`R` (an upper triangular matrix)

  Args:
    node: The node you want to split.
    left_edges: The edges you want connected to the new left node.
    right_edges: The edges you want connected to the new right node.
    left_name: The name of the new left node. If `None`, a name will be generated
      automatically.
    right_name: The name of the new right node. If `None`, a name will be generated
      automatically.
    edge_name: The name of the new `Edge` connecting the new left and right node.
      If `None`, a name will be generated automatically.

  Returns:
    A tuple containing:
      left_node:
        A new node created that connects to all of the `left_edges`.
        Its underlying tensor is :math:`Q`
      right_node:
        A new node created that connects to all of the `right_edges`.
        Its underlying tensor is :math:`R`
  """
  if not hasattr(node, 'backend'):
    raise TypeError('Node {} of type {} has no `backend`'.format(
        node, type(node)))

  if node.axis_names and edge_name:
    left_axis_names = []
    right_axis_names = [edge_name]
    for edge in left_edges:
      left_axis_names.append(node.axis_names[edge.axis1] if edge.node1 is node
                             else node.axis_names[edge.axis2])
    for edge in right_edges:
      right_axis_names.append(node.axis_names[edge.axis1] if edge.node1 is node
                              else node.axis_names[edge.axis2])
    left_axis_names.append(edge_name)
  else:
    left_axis_names = None
    right_axis_names = None

  backend = node.backend
  node.reorder_edges(left_edges + right_edges)
  q, r = backend.qr_decomposition(node.tensor, len(left_edges))
  left_node = network_components.Node(
      q, name=left_name, axis_names=left_axis_names, backend=backend.name)
  for i, edge in enumerate(left_edges):
    left_node.add_edge(edge, i)
    edge.update_axis(i, node, i, left_node)
  right_node = network_components.Node(
      r, name=right_name, axis_names=right_axis_names, backend=backend.name)
  for i, edge in enumerate(right_edges):
    # i + 1 to account for the new edge.
    right_node.add_edge(edge, i + 1)
    edge.update_axis(i + len(left_edges), node, i + 1, right_node)
  connect(left_node.edges[-1], right_node.edges[0], name=edge_name)
  return left_node, right_node


def split_node_rq(
    node: network_components.BaseNode,
    left_edges: List[network_components.Edge],
    right_edges: List[network_components.Edge],
    left_name: Optional[Text] = None,
    right_name: Optional[Text] = None,
    edge_name: Optional[Text] = None,
) -> Tuple[network_components.BaseNode, network_components.BaseNode]:
  """Split a `Node` using RQ (reversed QR) decomposition

  Let M be the matrix created by flattening left_edges and right_edges into
  2 axes. Let :math:`QR = M^*` be the QR Decomposition of
  :math:`M^*`. This will split the network into 2 nodes. The left node's
  tensor will be :math:`R^*` (a lower triangular matrix) and the right node's tensor will be
  :math:`Q^*` (an orthonormal matrix)

  Args:
    node: The node you want to split.
    left_edges: The edges you want connected to the new left node.
    right_edges: The edges you want connected to the new right node.
    left_name: The name of the new left node. If `None`, a name will be generated
      automatically.
    right_name: The name of the new right node. If `None`, a name will be generated
      automatically.
    edge_name: The name of the new `Edge` connecting the new left and right node.
      If `None`, a name will be generated automatically.

  Returns:
    A tuple containing:
      left_node:
        A new node created that connects to all of the `left_edges`.
        Its underlying tensor is :math:`Q`
      right_node:
        A new node created that connects to all of the `right_edges`.
        Its underlying tensor is :math:`R`
  """
  if not hasattr(node, 'backend'):
    raise TypeError('Node {} of type {} has no `backend`'.format(
        node, type(node)))

  if node.axis_names and edge_name:
    left_axis_names = []
    right_axis_names = [edge_name]
    for edge in left_edges:
      left_axis_names.append(node.axis_names[edge.axis1] if edge.node1 is node
                             else node.axis_names[edge.axis2])
    for edge in right_edges:
      right_axis_names.append(node.axis_names[edge.axis1] if edge.node1 is node
                              else node.axis_names[edge.axis2])
    left_axis_names.append(edge_name)
  else:
    left_axis_names = None
    right_axis_names = None
  backend = node.backend
  node.reorder_edges(left_edges + right_edges)
  r, q = backend.rq_decomposition(node.tensor, len(left_edges))
  left_node = network_components.Node(
      r, name=left_name, axis_names=left_axis_names, backend=backend.name)
  for i, edge in enumerate(left_edges):
    left_node.add_edge(edge, i)
    edge.update_axis(i, node, i, left_node)
  right_node = network_components.Node(
      q, name=right_name, axis_names=right_axis_names, backend=backend.name)
  for i, edge in enumerate(right_edges):
    # i + 1 to account for the new edge.
    right_node.add_edge(edge, i + 1)
    edge.update_axis(i + len(left_edges), node, i + 1, right_node)
  connect(left_node.edges[-1], right_node.edges[0], name=edge_name)
  return left_node, right_node


def split_node_full_svd(
    node: network_components.BaseNode,
    left_edges: List[network_components.Edge],
    right_edges: List[network_components.Edge],
    max_singular_values: Optional[int] = None,
    max_truncation_err: Optional[float] = None,
    left_name: Optional[Text] = None,
    middle_name: Optional[Text] = None,
    right_name: Optional[Text] = None,
    left_edge_name: Optional[Text] = None,
    right_edge_name: Optional[Text] = None,
) -> Tuple[network_components.BaseNode, network_components
           .BaseNode, network_components.BaseNode, Tensor]:
  """Split a node by doing a full singular value decomposition.

  Let M be the matrix created by flattening left_edges and right_edges into
  2 axes. Let :math:`U S V^* = M` be the Singular Value Decomposition of
  :math:`M`.

  The left most node will be :math:`U` tensor of the SVD, the middle node is
  the diagonal matrix of the singular values, ordered largest to smallest,
  and the right most node will be the :math:`V*` tensor of the SVD.

  The singular value decomposition is truncated if `max_singular_values` or
  `max_truncation_err` is not `None`.

  The truncation error is the 2-norm of the vector of truncated singular
  values. If only `max_truncation_err` is set, as many singular values will
  be truncated as possible while maintaining:
  `norm(truncated_singular_values) <= max_truncation_err`.

  If only `max_singular_values` is set, the number of singular values kept
  will be `min(max_singular_values, number_of_singular_values)`, so that
  `max(0, number_of_singular_values - max_singular_values)` are truncated.

  If both `max_truncation_err` and `max_singular_values` are set,
  `max_singular_values` takes priority: The truncation error may be larger
  than `max_truncation_err` if required to satisfy `max_singular_values`.

  Args:
    node: The node you want to split.
    left_edges: The edges you want connected to the new left node.
    right_edges: The edges you want connected to the new right node.
    max_singular_values: The maximum number of singular values to keep.
    max_truncation_err: The maximum allowed truncation error.
    left_name: The name of the new left node. If None, a name will be generated
      automatically.
    middle_name: The name of the new center node. If None, a name will be generated
      automatically.
    right_name: The name of the new right node. If None, a name will be generated
      automatically.
    left_edge_name: The name of the new left `Edge` connecting
      the new left node (`U`) and the new central node (`S`).
      If `None`, a name will be generated automatically.
    right_edge_name: The name of the new right `Edge` connecting
      the new central node (`S`) and the new right node (`V*`).
      If `None`, a name will be generated automatically.

  Returns:
    A tuple containing:
      left_node:
        A new node created that connects to all of the `left_edges`.
        Its underlying tensor is :math:`U`
      singular_values_node:
        A new node that has 2 edges connecting `left_node` and `right_node`.
        Its underlying tensor is :math:`S`
      right_node:
        A new node created that connects to all of the `right_edges`.
        Its underlying tensor is :math:`V^*`
      truncated_singular_values:
        The vector of truncated singular values.
  """
  if not hasattr(node, 'backend'):
    raise TypeError('Node {} of type {} has no `backend`'.format(
        node, type(node)))

  if node.axis_names and left_edge_name and right_edge_name:
    left_axis_names = []
    right_axis_names = [right_edge_name]
    for edge in left_edges:
      left_axis_names.append(node.axis_names[edge.axis1] if edge.node1 is node
                             else node.axis_names[edge.axis2])
    for edge in right_edges:
      right_axis_names.append(node.axis_names[edge.axis1] if edge.node1 is node
                              else node.axis_names[edge.axis2])
    left_axis_names.append(left_edge_name)
    center_axis_names = [left_edge_name, right_edge_name]
  else:
    left_axis_names = None
    center_axis_names = None
    right_axis_names = None

  backend = node.backend

  node.reorder_edges(left_edges + right_edges)
  u, s, vh, trun_vals = backend.svd_decomposition(
      node.tensor, len(left_edges), max_singular_values, max_truncation_err)
  left_node = network_components.Node(
      u, name=left_name, axis_names=left_axis_names, backend=backend.name)
  singular_values_node = network_components.Node(
      backend.diag(s),
      name=middle_name,
      axis_names=center_axis_names,
      backend=backend.name)

  right_node = network_components.Node(
      vh, name=right_name, axis_names=right_axis_names, backend=backend.name)

  for i, edge in enumerate(left_edges):
    left_node.add_edge(edge, i)
    edge.update_axis(i, node, i, left_node)
  for i, edge in enumerate(right_edges):
    # i + 1 to account for the new edge.
    right_node.add_edge(edge, i + 1)
    edge.update_axis(i + len(left_edges), node, i + 1, right_node)
  connect(
      left_node.edges[-1], singular_values_node.edges[0], name=left_edge_name)
  connect(
      singular_values_node.edges[1], right_node.edges[0], name=right_edge_name)
  return left_node, singular_values_node, right_node, trun_vals


def reachable(node: network_components.BaseNode,
              strategy: Optional[Text] = 'recursive'
             ) -> Collection[network_components.BaseNode]:
  """
  Computes all nodes reachable from `node` by connected edges.
  Args:
    node: A `BaseNode`
    strategy: The strategy to be used to find all nodes. Currently 
      if `recursive`, use a recursive approach, if `iterative`, use
      and iterative approach, if 'deque' use an iterative approach 
      using itertools.deque.
  Returns:
    A list of `BaseNode` objects that can be reached from `node`
    via connected edges.
  Raises:
    ValueError: If an unknown value for `strategy` is passed.
  """
  if strategy == 'recursive':
    return reachable_recursive(node)
  elif strategy == 'iterative':
    return reachable_iterative(node)
  elif strategy == 'deque':
    return reachable_deque(node)

  else:
    raise ValueError("Unknown value '{}' for `strategy`.".format(strategy))


def reachable_deque(node: network_components.BaseNode) -> None:
  """
  Computes all nodes reachable from `node` by connected edges. This function uses 
  itertools.deque.
  Args:
    node: A `BaseNode`
  Returns:
    A set of `BaseNode` objects that can be reached from `node`
    via connected edges.

  """
  # Fastest way to get a single item from a set.
  node_que = collections.deque()
  seen_nodes = {node}
  node_que.append(node)
  while node_que:
    node = node_que.popleft()
    for e in node.edges:
      for n in e.get_nodes():
        if n is not None and n not in seen_nodes:
          node_que.append(n)
          seen_nodes.add(n)
  return seen_nodes


def reachable_recursive(node: network_components.BaseNode
                       ) -> Collection[network_components.BaseNode]:
  """
  Computes all nodes reachable from `node` by connected edges. This function uses 
  recursion
  Args:
    node: A `BaseNode`
  Returns:
    A set of `BaseNode` objects that can be reached from `node`
    via connected edges.

  """
  reachable_nodes = {node}

  def _reachable(node):
    for edge in node.edges:
      if edge.is_dangling():
        continue
      next_node = edge.node1 if edge.node1 is not node else edge.node2
      if next_node in reachable_nodes:
        continue
      reachable_nodes.add(next_node)
      _reachable(next_node)

  _reachable(node)

  return reachable_nodes


def reachable_iterative(node: network_components.BaseNode
                       ) -> Collection[network_components.BaseNode]:
  """
  Computes all nodes reachable from `node` by connected edges. This function uses 
  an iterative strategy.
  Args:
    node: A `BaseNode`
  Returns:
    A list of `BaseNode` objects that can be reached from `node`
    via connected edges.
  """
  #TODO: this is recursive; use while loop for better performance
  reachable_nodes = {node}
  depth = 0
  while True:
    old_depth = depth
    for edge in node.edges:

      if edge.is_dangling():
        continue
      next_node = edge.node1 if edge.node1 is not node else edge.node2
      if next_node in reachable_nodes:
        continue
      reachable_nodes.add(next_node)
      depth += 1
      node = next_node
      continue

    depth -= 1 if depth == old_depth else 0
    if depth == 0:
      break
  return reachable_nodes


def check_correct(nodes: List[network_components.BaseNode],
                  check_connections: Optional[bool] = True) -> None:
  """
  Check if the network defined by `nodes` fulfills necessary
  consistency relations.

  Args:
    nodes: A list of `BaseNode` objects.
    check_connected: Check if the network is connected.
  Returns:
    None
  Raises:
    ValueError: If the network defined by `nodes` is not 
      correctly structured.
  """
  for node in nodes:
    for i, edge in enumerate(node.edges):
      if edge.node1 is not node and edge.node2 is not node:
        raise ValueError("Edge '{}' does not connect to node '{}'."
                         "Edge's nodes: '{}', '{}'.".format(
                             edge, node, edge.node1, edge.node2))

      is_edge_node_consistent = False
      if edge.node1 is node:
        if edge.axis1 == i:
          is_edge_node_consistent = True
      if edge.node2 is node:
        if edge.axis2 == i:
          is_edge_node_consistent = True
      if not is_edge_node_consistent:
        raise ValueError(
            "Edge '{}' does not point to '{}' on the correct axis. "
            "Edge axes: {}, {}. Node axis: {}.".format(edge, node, edge.axis1,
                                                       edge.axis2, i))
  if check_connections:
    check_connected(nodes)


def check_connected(nodes: List[network_components.BaseNode]) -> None:
  """
  Check if all nodes in `nodes` are connected.
  Args:
    nodes: A list of nodes.
  Returns:
    None:
  Raises:
    ValueError: If not all nodes in `nodes` are connected.
  """
  if not (set(nodes) <= reachable_deque(nodes[0])):
    raise ValueError("Non-connected graph")


def get_all_nondangling(nodes: Collection[network_components.BaseNode]
                       ) -> Set[network_components.Edge]:
  """Return the set of all non-dangling edges."""
  edges = set()
  for node in nodes:
    edges |= node.get_all_nondangling()
  return edges


def get_all_nodes(edges: Collection[network_components.Edge]
                 ) -> Set[network_components.BaseNode]:
  """Return the set of nodes connected to edges."""
  nodes = set()
  for edge in edges:
    if edge.node1 is not None:
      nodes |= {edge.node1}
    if edge.node2 is not None:
      nodes |= {edge.node2}

  return nodes


#Fake class to fix import issues
class TensorNetwork:

  def __init__(self):
    pass
