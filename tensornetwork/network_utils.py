from tensornetwork.network_components import Edge, BaseNode
from tensornetwork import network
from tensornetwork import TensorNetwork
from typing import Text, Sequence, Optional, List, Union


def check_node_attributes(node1, node2):
  for node in [node1, node2]:
    if not hasattr(node, 'backend'):
      raise TypeError('Node {} of type {} has no `backend`'.format(
          node, type(node)))

  if node1.backend.name != node2.backend.name:
    raise ValueError("node {}  and node {} have different backends. "
                     "Cannot perform a contraction".format(node1, node2))


def contract(edge: Edge, name: Optional[Text] = None) -> BaseNode:
  """Contract an edge connecting two nodes.

  Args:
    edge: The edge contract next.
    name: Name of the new node created.

  Returns:
    The new node created after the contraction.

  Raises:
    ValueError: If edge has no nodes
    ValueError: If edge is a dangling edge or if it already has been
      contracted.
    ValueError: If backends of the edge.node1 and edge.node2 are different
      contracted.
    TypeError: If edge.node1 and edge.node2 are different have no 
      `backend` attribute
  """
  check_node_attributes(edge.node1, edge.node2)
  if edge.node1:
    backend = edge.node1.backend
  else:
    raise ValueError("edge {} has no nodes. "
                     "Cannot perfrom a contraction".format(edge.name))
  return network.contract(edge=edge, backend=backend, net=None, name=name)


def contract_between(
    node1: BaseNode,
    node2: BaseNode,
    name: Optional[Text] = None,
    allow_outer_product: bool = False,
    output_edge_order: Optional[Sequence[Edge]] = None,
) -> BaseNode:
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

  Returns:
    The new node created.

  Raises:
    ValueError: If no edges are found between node1 and node2 and
      `allow_outer_product` is set to `False`.
    ValueError: If backends of the `node1` and `node2` are different
      contracted.
    TypeError: If `node1` and `node2` are different have no 
      `backend` attribute
  """
  check_node_attributes(node1, node2)
  backend = node1.backend
  return network.network.contract_between(
      node1,
      node2,
      backend,
      net=None,
      name=name,
      allow_outer_product=allow_outer_product,
      output_edge_order=output_edge_order)


def outer_product(node1: BaseNode, node2: BaseNode,
                  name: Optional[Text] = None) -> BaseNode:
  """Calculates the outer product of the two nodes.

  This causes the nodes to combine their edges and axes, so the shapes are
  combined. For example, if `a` had a shape (2, 3) and `b` had a shape
  (4, 5, 6), then the node `net.outer_product(a, b) will have shape
  (2, 3, 4, 5, 6).

  Args:
    node1: The first node. The axes on this node will be on the left side of
      the new node.
    node2: The second node. The axes on this node will be on the right side of
      the new node.
    name: Optional name to give the new node created.

  Returns:
    A new node. Its shape will be node1.shape + node2.shape. It's edges 
      are inherited from `node1` and `node2`, and `node1` and `node2`
      get new dangling edges
  Raises:
    ValueError: If backends of the `node1` and `node2` are different
      contracted.
    TypeError: If `node1` and `node2` are different have no 
      `backend` attribute
  """
  check_node_attributes(node1, node2)
  backend = node1.backend
  return network.outer_product(
      node1=node1, node2=node2, backend=backend, net=None, name=name)


def conj(node: BaseNode,
         name: Optional[Text] = None,
         axis_names: Optional[List[Text]] = None) -> BaseNode:
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

  return BaseNode(
      node.tensor, name=name, axis_names=axis_names, backend=backend.name)


def transpose(node: BaseNode,
              permutation: Sequence[Union[Text, int]],
              name: Optional[Text] = None,
              axis_names: Optional[List[Text]] = None) -> BaseNode:
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
  backend = node.backend
  if not axis_names:
    axis_names = node.axis_names
  new_node = BaseNode(
      node.tensor, name=name, axis_names=node.axis_names, backend=backend.name)
  new_order = [new_node[n] for n in permutation]
  new_node.reorder_edges(new_order)
