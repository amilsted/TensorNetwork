from tensornetwork.network_components import Edge, BaseNode, FreeNode
from typing import Set


def connect(edge1: Edge, edge2: Edge) -> Edge:
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

  if isinstance(edge1.node1, FreeNode) and isinstance(edge2.node1, FreeNode):
    node1 = edge1.node1
    node2 = edge2.node1
    axis1_num = node1.get_axis_number(edge1.axis1)
    axis2_num = node2.get_axis_number(edge2.axis1)
    new_edge = Edge(None, node1, axis1_num, node2, axis2_num)
    #we're not touching .edges; instead we're inserting
    #`new_edge` into `node1.connected_edges` and `node2.connected_edges`
    node1.connected_edges[axis1_num] = new_edge
    node2.connected_edges[axis2_num] = new_edge
    return new_edge

  elif (isinstance(edge1.node1, FreeNode) != isinstance(edge2.node1, FreeNode)):
    raise TypeError(
        "Only nodes with same types can be connected. Got nodes with "
        "different types type(node1) = {} and type(node2) = {}.".format(
            type(edge1.node1), type(edge2.node1)))
  return edge1.node1.network.connect(edge1, edge2)


def get_shared_edges(node1: BaseNode, node2: BaseNode) -> Set[Edge]:
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


# def contract_between(
#     node1: network_components.BaseNode,
#     node2: network_components.BaseNode,
#     name: Optional[Text] = None,
#     allow_outer_product: bool = False,
#     output_edge_order: Optional[Sequence[network_components.Edge]] = None,
# ) -> BaseNode:
#   """Contract all of the edges between the two given nodes.

#   Args:
#     node1: The first node.
#     node2: The second node.
#     name: Name to give to the new node created.
#     allow_outer_product: Optional boolean. If two nodes do not share any edges
#       and `allow_outer_product` is set to `True`, then we return the outer
#       product of the two nodes. Else, we raise a `ValueError`.
#     output_edge_order: Optional sequence of Edges. When not `None`, must
#       contain all edges belonging to, but not shared by `node1` and `node2`.
#       The axes of the new node will be permuted (if necessary) to match this
#       ordering of Edges.

#   Returns:
#     The new node created.

#   Raises:
#     ValueError: If no edges are found between node1 and node2 and
#       `allow_outer_product` is set to `False`.
#   """
#   # Trace edges cannot be contracted using tensordot.
#   if node1 is node2:
#     flat_edge = self.flatten_edges_between(node1, node2)
#     if not flat_edge:
#       raise ValueError("No trace edges found on contraction of edges between "
#                        "node '{}' and itself.".format(node1))
#     return self.contract(flat_edge, name)

#   shared_edges = self.get_shared_edges(node1, node2)
#   if not shared_edges:
#     if allow_outer_product:
#       return self.outer_product(node1, node2)
#     raise ValueError("No edges found between nodes '{}' and '{}' "
#                      "and allow_outer_product=False.".format(node1, node2))

#   # Collect the axis of each node corresponding to each edge, in order.
#   # This specifies the contraction for tensordot.
#   # NOTE: The ordering of node references in each contraction edge is ignored.
#   axes1 = []
#   axes2 = []
#   for edge in shared_edges:
#     if edge.node1 is node1:
#       axes1.append(edge.axis1)
#       axes2.append(edge.axis2)
#     else:
#       axes1.append(edge.axis2)
#       axes2.append(edge.axis1)

#   if output_edge_order:
#     # Determine heuristically if output transposition can be minimized by
#     # flipping the arguments to tensordot.
#     node1_output_axes = []
#     node2_output_axes = []
#     for (i, edge) in enumerate(output_edge_order):
#       if edge in shared_edges:
#         raise ValueError(
#             "Edge '{}' in output_edge_order is shared by the nodes to be "
#             "contracted: '{}' and '{}'.".format(edge, node1, node2))
#       edge_nodes = set(edge.get_nodes())
#       if node1 in edge_nodes:
#         node1_output_axes.append(i)
#       elif node2 in edge_nodes:
#         node2_output_axes.append(i)
#       else:
#         raise ValueError(
#             "Edge '{}' in output_edge_order is not connected to node '{}' or "
#             "node '{}'".format(edge, node1, node2))
#     if np.mean(node1_output_axes) > np.mean(node2_output_axes):
#       node1, node2 = node2, node1
#       axes1, axes2 = axes2, axes1

#   new_tensor = self.backend.tensordot(node1.tensor, node2.tensor,
#                                       [axes1, axes2])
#   new_node = self.add_node(new_tensor, name)
#   # The uncontracted axes of node1 (node2) now correspond to the first (last)
#   # axes of new_node. We provide this ordering to _remove_edges() via the
#   # node1 and node2 arguments.
#   self._remove_edges(shared_edges, node1, node2, new_node)

#   if output_edge_order:
#     new_node = new_node.reorder_edges(list(output_edge_order))
#   return new_node

# def flatten_edges_between(node1: BaseNode, node2: BaseNode) -> Optional[Edge]:
#   """Flatten all of the edges between the given two nodes.

#   Args:
#     node1: The first node.
#     node2: The second node.

#   Returns:
#     The flattened `Edge` object. If there was only one edge between the two
#       nodes, then the original edge is returned. If there where no edges
#       between the nodes, a None is returned.
#   """
#   shared_edges = self.get_shared_edges(node1, node2)
#   if shared_edges:
#     return self.flatten_edges(list(shared_edges))
#   return None
