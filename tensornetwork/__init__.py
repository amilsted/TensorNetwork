from __future__ import absolute_import
from tensornetwork.network import TensorNetwork
from tensornetwork.network_components import Node, Edge, CopyNode
from tensornetwork.network import conj, transpose, contract, contract_copy_node, contract_between, outer_product, split_node, split_node_qr, split_node_rq, split_node_full_svd, reachable, contract_parallel, check_connected, check_correct, connect, flatten_edges, flatten_edges_between, get_all_nondangling
from tensornetwork.ncon_interface import ncon, ncon_network
from tensornetwork.version import __version__
from tensornetwork.visualization.graphviz import to_graphviz
from tensornetwork import contractors
from tensornetwork import config
from tensornetwork.backends import backend_factory

from typing import Text, Optional, Type
from tensornetwork.utils import load


def set_default_backend(backend: Text, dtype: Optional[Type] = None) -> None:
  config.default_backend = backend
  config.default_dype = dtype
