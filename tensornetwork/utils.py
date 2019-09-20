import h5py
from typing import Collection, Union, BinaryIO
from tensornetwork.component_factory import get_component
from tensornetwork.network_components import Edge, BaseNode
import tensornetwork as tn
import numpy as np
string_type = h5py.special_dtype(vlen=str)


def save(nodes: Collection[BaseNode], path: Union[str, BinaryIO]):
  """
  Save a collection of nodes into hdf5 format.
  Args:
    nodes: A collection of connected nodes. All nodes have to connect within
      `nodes`.
    path: path to folder where network is saved.
  """
  if tn.reachable(nodes) > set(nodes):
    raise ValueError(
        "Some nodes in `nodes` are connected to nodes not contained in `nodes`. "
        "Saving not possible.")
  if len(set(nodes)) < len(nodes):
    raise ValueError(
        'Some nodes in `nodes` appear more than once. This is not supported')
  #we need to iterate twice and order matters
  edges = list(tn.get_all_edges(nodes))
  nodes = list(nodes)

  old_edge_names = {n: edge.name for n, edge in enumerate(edges)}
  old_node_names = {n: node.name for n, node in enumerate(nodes)}

  #unique names for nodes and edges
  for n, node in enumerate(nodes):
    node.set_name('node{}'.format(n))

  for e, edge in enumerate(edges):
    edge.set_name('edge{}'.format(e))

  with h5py.File(path, 'w') as net_file:
    nodes_group = net_file.create_group('nodes')
    node_names_group = net_file.create_group('node_names')
    node_names_group.create_dataset(
        'names',
        dtype=string_type,
        data=np.array(list(old_node_names.values()), dtype=object))

    edges_group = net_file.create_group('edges')
    edge_names_group = net_file.create_group('edge_names')
    edge_names_group.create_dataset(
        'names',
        dtype=string_type,
        data=np.array(list(old_edge_names.values()), dtype=object))

    for n, node in enumerate(nodes):
      node.set_name('node{}'.format(n))

      node_group = nodes_group.create_group(node.name)
      node._save_node(node_group)
      for edge in node.edges:
        if edge.node1 == node and edge in edges:
          edge_group = edges_group.create_group(edge.name)
          edge._save_edge(edge_group)
          edges.remove(edge)

  #name edges and nodes back  to their original names
  for n, node in enumerate(nodes):
    nodes[n].set_name(old_node_names[n])

  for n, edge in enumerate(edges):
    edges[n].set_name(old_edge_names[n])


def load(path: str):
  """
  Load a set of nodes from disk.

  Args:
    path: path to folder where network is saved.
  """
  nodes_list = []
  edges_list = []
  with h5py.File(path, 'r') as net_file:
    nodes = list(net_file["nodes"].keys())
    node_names = {
        'node{}'.format(n): v
        for n, v in enumerate(net_file["node_names"]['names'][()])
    }

    edge_names = {
        'edge{}'.format(n): v
        for n, v in enumerate(net_file["edge_names"]['names'][()])
    }
    edges = list(net_file["edges"].keys())
    #print(net_file["nodes"]['node_names'][()])
    #print(net_file["edges"]['edge_names'][()])
    for n, node_name in enumerate(nodes):
      node_data = net_file["nodes/" + node_name]
      node_type = get_component(node_data['type'][()])
      nodes_list.append(node_type._load_node(node_data))
    nodes_dict = {node.name: node for node in nodes_list}
    for n, edge in enumerate(edges):
      edge_data = net_file["edges/" + edge]
      edges_list.append(Edge._load_edge(edge_data, nodes_dict))

  for edge in edges_list:
    edge.set_name(edge_names[edge.name])
  for node in nodes_list:
    node.set_name(node_names[node.name])

  return nodes_list
