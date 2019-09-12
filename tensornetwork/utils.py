import h5py
from tensornetwork.network import TensorNetwork
from tensornetwork.component_factory import get_component
from tensornetwork.network_components import Edge

# from tensornetwork.network_components import BaseNode

# def connect(edge1: Edge, edge2: Edge) -> Edge:
#   if edge1 is edge2:
#     if isinstance(edge1.node1, FreeNode) and isinstance(edge1.node2, FreeNode):
#       edge1, edge2 = edge1.break_edge()
#       node1 = edge1.node1
#       node2 = edge2.node1
#     else:
#       raise ValueError("Cannot connect edge '{}' to itself.".format(self))
#   else:
#     if isinstance(self.node1, FreeNode) and isinstance(other.node1, FreeNode):
#       node1 = self.node1
#       node2 = other.node1
#       edge1 = self
#       edge2 = other
#     elif not (isinstance(self.node1, FreeNode) or
#               isinstance(other.node1, FreeNode)):
#       if self.dimension != other.dimension:
#         raise ValueError("Cannot connect edges of unequal dimension. "
#                          "Dimension of edge '{}': {}, "
#                          "Dimension of edge '{}': {}.".format(
#                              self, other.dimension, other, other.dimension))

#       for edge in [self, other]:
#         if not edge.is_dangling():
#           raise ValueError("Edge '{}' is not a dangling edge. "
#                            "This edge points to nodes: '{}' and '{}'".format(
#                                edge, edge.node1, edge.node2))
#       return self.node1.network.connect(self, other)
#     else:
#       raise TypeError(
#           "Only nodes with same types can be connected. Got nodes with "
#           "different types type(node1) = {} and type(node2) = {}.".format(
#               type(self.node1), type(other.node1)))
#   axis1_num = node1.get_axis_number(edge1.axis1)
#   axis2_num = node2.get_axis_number(edge2.axis1)
#   new_edge = Edge(None, node1, axis1_num, node2, axis2_num)
#   node1.add_edge(new_edge, axis1_num)
#   node2.add_edge(new_edge, axis2_num)
#   return new_edge

# def get_shared_edges(node1: BaseNode, node2: BaseNode) -> Set[Edge]:
#   """Get all edges shared between two nodes.

#   Args:
#     node1: The first node.
#     node2: The second node.

#   Returns:
#     A (possibly empty) `set` of `Edge`s shared by the nodes.
#   """
#   nodes = {node1, node2}
#   shared_edges = set()
#   # Assuming the network is well formed, all of the edges shared by
#   # these two nodes will be stored in just one of the nodes, so we only
#   # have to do this loop once.
#   for edge in node1.edges:
#     if set(edge.get_nodes()) == nodes:
#       shared_edges.add(edge)
#   return shared_edges


def load(path: str):
  """Load a tensor network from disk.

  Args:
    path: path to folder where network is saved.
  """
  with h5py.File(path, 'r') as net_file:
    net = TensorNetwork(backend=net_file["backend"][()])
    nodes = list(net_file["nodes"].keys())
    edges = list(net_file["edges"].keys())

    for node_name in nodes:
      node_data = net_file["nodes/" + node_name]
      node_type = get_component(node_data['type'][()])
      node_type._load_node(net, node_data)

    nodes_dict = {node.name: node for node in net.nodes_set}

    for edge in edges:
      edge_data = net_file["edges/" + edge]
      Edge._load_edge(edge_data, nodes_dict)
  return net
