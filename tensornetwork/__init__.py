from __future__ import absolute_import
from tensornetwork import config
from tensornetwork.backends import backend_factory
global global_backend
global_backend = backend_factory.get_backend(config.default_backend,
                                             config.default_dtype)

from tensornetwork.network import TensorNetwork
from tensornetwork.network_components import Node, Edge, CopyNode, FreeNode
from tensornetwork.ncon_interface import ncon, ncon_network
from tensornetwork.version import __version__
from tensornetwork.visualization.graphviz import to_graphviz
from tensornetwork import contractors

from typing import Text, Optional, Type
from tensornetwork.utils import load


def set_default_backend(backend: Text, dtype: Optional[Type] = None) -> None:
  global global_backend
  config.default_backend = backend
  config.default_dype = dtype
  global_backend = backend_factory.get_backend(backend, dtype)
