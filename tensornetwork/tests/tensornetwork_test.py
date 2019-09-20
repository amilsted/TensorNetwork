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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensornetwork as tn
import pytest
import numpy as np
import tensorflow as tf
import torch
import jax
from jax.config import config
import tensornetwork.network as network
from tensornetwork.backends import backend_factory
import tensornetwork.config as config_file
config.update("jax_enable_x64", True)
tf.compat.v1.enable_v2_behavior()

np_dtypes = [np.float32, np.float64, np.complex64, np.complex128, np.int32]
tf_dtypes = [tf.float32, tf.float64, tf.complex64, tf.complex128, tf.int32]
torch_dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]
jax_dtypes = [
    jax.numpy.float32, jax.numpy.float64, jax.numpy.complex64,
    jax.numpy.complex128, jax.numpy.int32
]


def test_node_names(backend):
  a = tn.Node(np.eye(2), "a", axis_names=["e0", "e1"], backend=backend)
  assert a.name == "a"
  assert a[0].name == "e0"
  assert a[1].name == "e1"


def test_copy_node(backend):
  a = tn.CopyNode(
      3, 3, name="TestName", axis_names=['a', 'b', 'c'], backend=backend)
  assert a.shape == (3, 3, 3)
  assert isinstance(a, tn.CopyNode)
  assert a.name == "TestName"
  assert a.axis_names == ['a', 'b', 'c']
  b = tn.Node(np.eye(3), backend=backend)
  e = a[0] ^ b[0]
  c = tn.contract(e)
  np.testing.assert_allclose(c.tensor, a.tensor)


def test_default_names(backend):
  a = tn.CopyNode(3, 3, backend=backend)
  assert a.name is not None
  assert a.axis_names is None


def test_set_tensor(backend):
  a = tn.Node(np.ones(2), backend=backend)
  np.testing.assert_allclose(a.tensor, np.ones(2))
  a.tensor = np.zeros(2)
  np.testing.assert_allclose(a.tensor, np.zeros(2))


def test_has_nondangling_edge(backend):
  a = tn.Node(np.ones(2), backend=backend)
  assert not a.has_nondangling_edge()
  b = tn.Node(np.ones((2, 2)))
  tn.connect(b[0], b[1])
  assert b.has_nondangling_edge()


def test_large_nodes(backend):
  a = tn.Node(np.zeros([5, 6, 7, 8, 9]), "a", backend=backend)
  b = tn.Node(np.zeros([5, 6, 7, 8, 9]), "b", backend=backend)
  for i in range(5):
    tn.connect(a[i], b[i])
  tn.check_correct([a, b])


def test_small_matmul(backend):
  a = tn.Node(np.zeros([10, 10]), name="a", backend=backend)
  b = tn.Node(np.zeros([10, 10]), name="b", backend=backend)
  edge = tn.connect(a[0], b[0], "edge")
  tn.check_correct([a, b])
  c = tn.contract(edge, name="a * b")
  assert list(c.shape) == [10, 10]
  tn.check_correct([a, b], False)
  with pytest.raises(ValueError):
    tn.check_connected([a, b])
  tn.check_correct([c])


def test_double_trace(backend):
  a = tn.Node(np.ones([10, 10, 10, 10]), name="a", backend=backend)
  edge1 = tn.connect(a[0], a[1], "edge1")
  edge2 = tn.connect(a[2], a[3], "edge2")
  tn.check_correct([a])
  node = network._contract_trace(edge1)
  tn.check_correct([a])
  val = network._contract_trace(edge2)
  tn.check_correct([a])
  np.testing.assert_allclose(val.tensor, 100.0)


def test_indirect_trace(backend):
  a = tn.Node(np.ones([10, 10]), name="a", backend=backend)
  edge = tn.connect(a[0], a[1], "edge")
  tn.check_correct([a])
  val = tn.contract(edge)
  tn.check_correct([val])
  tn.check_correct([a])

  np.testing.assert_allclose(val.tensor, 10.0)


def test_real_physics(backend):
  # Calcuate the expected value in numpy
  a_vals = np.ones([2, 3, 4, 5])
  b_vals = np.ones([4, 6, 7])
  c_vals = np.ones([5, 6, 8])
  contract1 = np.tensordot(a_vals, b_vals, [[2], [0]])
  contract2 = np.tensordot(c_vals, contract1, [[0], [2]])
  final_result = np.trace(contract2, axis1=0, axis2=4)
  # Build the network
  a = tn.Node(a_vals, name="T", backend=backend)
  b = tn.Node(b_vals, name="A", backend=backend)
  c = tn.Node(c_vals, name="B", backend=backend)
  e1 = tn.connect(a[2], b[0], "edge")
  e2 = tn.connect(c[0], a[3], "edge2")
  e3 = tn.connect(b[1], c[1], "edge3")

  tn.check_correct([a, b, c], check_connections=True)
  node_result = tn.contract(e1)
  np.testing.assert_allclose(node_result.tensor, contract1)
  tn.check_correct([a, b, c], check_connections=False)
  with pytest.raises(ValueError):
    tn.check_connected([a, b, c])

  tn.check_correct([node_result, c], check_connections=True)
  node_result = tn.contract(e2)
  np.testing.assert_allclose(node_result.tensor, contract2)
  tn.check_correct([node_result])
  val = tn.contract(e3)
  tn.check_correct([val], check_connections=True)
  np.testing.assert_allclose(val.tensor, final_result)


def test_real_physics_with_tensors(backend):
  # Calcuate the expected value in numpy
  a_vals = np.ones([2, 3, 4, 5])
  b_vals = np.ones([4, 6, 7])
  c_vals = np.ones([5, 6, 8])
  contract1 = np.tensordot(a_vals, b_vals, [[2], [0]])
  contract2 = np.tensordot(c_vals, contract1, [[0], [2]])
  final_result = np.trace(contract2, axis1=0, axis2=4)
  # Build the network

  a = tn.Node(np.ones([2, 3, 4, 5]), name="T", backend=backend)
  b = tn.Node(np.ones([4, 6, 7]), name="A", backend=backend)
  c = tn.Node(np.ones([5, 6, 8]), name="B", backend=backend)
  e1 = tn.connect(a[2], b[0], "edge")
  e2 = tn.connect(c[0], a[3], "edge2")
  e3 = tn.connect(b[1], c[1], "edge3")
  tn.check_correct([a, b, c])
  node_result = tn.contract(e1)
  tn.check_correct([node_result, c])
  np.testing.assert_allclose(node_result.tensor, contract1)
  tn.check_correct([a, b, c], check_connections=False)
  with pytest.raises(ValueError):
    tn.check_connected([a, b, c])
  node_result = tn.contract(e2)
  tn.check_correct([a, b, c], check_connections=False)
  np.testing.assert_allclose(node_result.tensor, contract2)
  tn.check_correct([node_result])
  val = tn.contract(e3)
  tn.check_correct([val])
  np.testing.assert_allclose(val.tensor, final_result)


def test_real_physics_naive_contraction(backend):
  # Calcuate the expected value in numpy
  a_vals = np.ones([2, 3, 4, 5])
  b_vals = np.ones([4, 6, 7])
  c_vals = np.ones([5, 6, 8])
  contract1 = np.tensordot(a_vals, b_vals, [[2], [0]])
  contract2 = np.tensordot(c_vals, contract1, [[0], [2]])
  final_result = np.trace(contract2, axis1=0, axis2=4)
  # Build the network

  a = tn.Node(np.ones([2, 3, 4, 5]), name="T", backend=backend)
  b = tn.Node(np.ones([4, 6, 7]), name="A", backend=backend)
  c = tn.Node(np.ones([5, 6, 8]), name="B", backend=backend)
  e1 = tn.connect(a[2], b[0], "edge")
  e2 = tn.connect(c[0], a[3], "edge2")
  e3 = tn.connect(b[1], c[1], "edge3")
  for edge in [e1, e2, e3]:
    val = tn.contract(edge)
  assert list(val.shape) == [8, 2, 3, 7]
  np.testing.assert_allclose(val.tensor, final_result)


def test_with_tensors(backend):

  a = tn.Node(np.eye(2) * 2, name="T", backend=backend)
  b = tn.Node(np.eye(2) * 3, name="A", backend=backend)
  e1 = tn.connect(a[0], b[0], "edge")
  e2 = tn.connect(a[1], b[1], "edge2")
  tn.check_correct([a, b])
  val = tn.contract(e1)
  tn.check_correct([val])
  val = tn.contract(e2)
  tn.check_correct([val])
  np.testing.assert_allclose(val.tensor, 12.0)


def test_node2_contract_trace(backend):
  a = tn.Node(np.zeros([3, 3, 1]), backend=backend)
  b = tn.Node(np.zeros([1]), backend=backend)
  tn.connect(b[0], a[2])
  trace_edge = tn.connect(a[0], a[1])
  val = network._contract_trace(trace_edge)
  tn.check_correct([val])


def test_node_get_dim_bad_axis(backend):
  node = tn.Node(np.eye(2), name="a", axis_names=["1", "2"], backend=backend)
  with pytest.raises(ValueError):
    node.get_dimension(10)


def test_named_axis(backend):
  a = tn.Node(np.eye(2), axis_names=["alpha", "beta"], backend=backend)
  e = tn.connect(a["alpha"], a["beta"])
  b = tn.contract(e)
  np.testing.assert_allclose(b.tensor, 2.0)


def test_mixed_named_axis(backend):
  a = tn.Node(np.eye(2) * 2.0, axis_names=["alpha", "beta"], backend=backend)
  b = tn.Node(np.eye(2) * 3.0, backend=backend)
  e1 = tn.connect(a["alpha"], b[0])
  # Axes should still be indexable by numbers even with naming.
  e2 = tn.connect(a[1], b[1])
  val1 = tn.contract(e1)
  val2 = tn.contract(e2)
  np.testing.assert_allclose(val2.tensor, 12.0)


def test_duplicate_name(backend):
  with pytest.raises(ValueError):
    tn.Node(np.eye(2), axis_names=["test", "test"], backend=backend)


def test_bad_axis_name_length(backend):
  with pytest.raises(ValueError):
    # This should have 2 names, not 1.
    tn.Node(np.eye(2), axis_names=["need_2_names"], backend=backend)


def test_bad_axis_name_connect(backend):
  a = tn.Node(np.eye(2), axis_names=["test", "names"], backend=backend)
  with pytest.raises(ValueError):
    a.get_edge("bad_name")


def test_node_edge_ordering(backend):
  a = tn.Node(np.zeros((2, 3, 4)), backend=backend)
  e2 = a[0]
  e3 = a[1]
  e4 = a[2]
  assert a.shape == (2, 3, 4)
  a.reorder_edges([e4, e2, e3])
  tn.check_correct([a])
  assert a.shape == (4, 2, 3)
  assert e2.axis1 == 1
  assert e3.axis1 == 2
  assert e4.axis1 == 0


def test_trace_edge_ordering(backend):
  a = tn.Node(np.zeros((2, 2, 3)), backend=backend)
  e2 = tn.connect(a[1], a[0])
  e3 = a[2]
  with pytest.raises(ValueError):
    a.reorder_edges([e2, e3])


def test_mismatch_edge_ordering(backend):
  a = tn.Node(np.zeros((2, 3)), backend=backend)
  e2_a = a[0]
  b = tn.Node(np.zeros((2,)), backend=backend)
  e_b = b[0]
  with pytest.raises(ValueError):
    a.reorder_edges([e2_a, e_b])


def test_complicated_edge_reordering(backend):
  a = tn.Node(np.zeros((2, 3, 4)), backend=backend)
  b = tn.Node(np.zeros((2, 5)), backend=backend)
  c = tn.Node(np.zeros((3,)), backend=backend)
  d = tn.Node(np.zeros((4, 5)), backend=backend)
  e_ab = tn.connect(a[0], b[0])
  e_bd = tn.connect(b[1], d[1])
  e_ac = tn.connect(a[1], c[0])
  e_ad = tn.connect(a[2], d[0])
  result = tn.contract(e_bd)
  a.reorder_edges([e_ac, e_ab, e_ad])
  tn.check_correct([a, c, result])
  assert a.shape == (3, 2, 4)


def test_edge_reorder_axis_names(backend):
  a = tn.Node(
      np.zeros((2, 3, 4, 5)), axis_names=["a", "b", "c", "d"], backend=backend)
  edge_a = a["a"]
  edge_b = a["b"]
  edge_c = a["c"]
  edge_d = a["d"]
  a.reorder_edges([edge_c, edge_b, edge_d, edge_a])
  assert a.shape == (4, 3, 5, 2)
  assert a.axis_names == ["c", "b", "d", "a"]


def test_add_axis_names(backend):
  a = tn.Node(
      np.eye(2), name="A", axis_names=["ignore1", "ignore2"], backend=backend)
  a.add_axis_names(["a", "b"])
  assert a.axis_names == ["a", "b"]


def test_reorder_axes(backend):
  a = tn.Node(np.zeros((2, 3, 4)), backend=backend)
  b = tn.Node(np.zeros((3, 4, 5)), backend=backend)
  c = tn.Node(np.zeros((2, 4, 5)), backend=backend)
  tn.connect(a[0], c[0])
  tn.connect(b[0], a[1])
  tn.connect(a[2], c[1])
  tn.connect(b[2], c[2])
  a.reorder_axes([2, 0, 1])
  tn.check_correct([a, b, c])
  assert a.shape == (4, 2, 3)


def test_flatten_consistent_result(backend):
  a_val = np.ones((3, 5, 5, 6))
  b_val = np.ones((5, 6, 4, 5))
  # Create non flattened example to compare against.
  a_noflat = tn.Node(a_val, backend=backend)
  b_noflat = tn.Node(b_val, backend=backend)
  e1 = tn.connect(a_noflat[1], b_noflat[3])
  e2 = tn.connect(a_noflat[3], b_noflat[1])
  e3 = tn.connect(a_noflat[2], b_noflat[0])
  a_dangling_noflat = a_noflat[0]
  b_dangling_noflat = b_noflat[2]
  for edge in [e1, e2, e3]:
    noflat_result_node = tn.contract(edge)
  noflat_result_node.reorder_edges([a_dangling_noflat, b_dangling_noflat])
  noflat_result = noflat_result_node.tensor

  # Create network with flattening
  a_flat = tn.Node(a_val, backend=backend)
  b_flat = tn.Node(b_val, backend=backend)
  e1 = tn.connect(a_flat[1], b_flat[3])
  e2 = tn.connect(a_flat[3], b_flat[1])
  e3 = tn.connect(a_flat[2], b_flat[0])
  a_dangling_flat = a_flat[0]
  b_dangling_flat = b_flat[2]
  final_edge = tn.flatten_edges([e1, e2, e3])
  flat_result_node = tn.contract(final_edge)
  flat_result_node.reorder_edges([a_dangling_flat, b_dangling_flat])
  flat_result = flat_result_node.tensor
  np.testing.assert_allclose(flat_result, noflat_result)


def test_flatten_consistent_tensor(backend):
  a_val = np.ones((2, 3, 4, 5))
  b_val = np.ones((3, 5, 4, 2))
  a = tn.Node(a_val, backend=backend)
  b = tn.Node(b_val, backend=backend)
  e1 = tn.connect(a[0], b[3])
  e2 = tn.connect(b[1], a[3])
  e3 = tn.connect(a[1], b[0])
  tn.flatten_edges([e3, e1, e2])
  tn.check_correct([a, b])

  # Check expected values.
  a_final = np.reshape(np.transpose(a_val, (2, 1, 0, 3)), (4, 30))
  b_final = np.reshape(np.transpose(b_val, (2, 0, 3, 1)), (4, 30))
  np.testing.assert_allclose(a.tensor, a_final)
  np.testing.assert_allclose(b.tensor, b_final)


def test_flatten_trace_consistent_result(backend):
  a_val = np.ones((5, 6, 6, 7, 5, 7))
  a_noflat = tn.Node(a_val, backend=backend)
  e1 = tn.connect(a_noflat[0], a_noflat[4])
  e2 = tn.connect(a_noflat[1], a_noflat[2])
  e3 = tn.connect(a_noflat[3], a_noflat[5])
  for edge in [e1, e2, e3]:
    result = tn.contract(edge)
  noflat_result = result.tensor
  # Create network with flattening
  a_flat = tn.Node(a_val, backend=backend)
  e1 = tn.connect(a_flat[0], a_flat[4])
  e2 = tn.connect(a_flat[1], a_flat[2])
  e3 = tn.connect(a_flat[3], a_flat[5])
  final_edge = tn.flatten_edges([e1, e2, e3])
  flat_result = tn.contract(final_edge).tensor
  np.testing.assert_allclose(flat_result, noflat_result)


def test_flatten_trace_consistent_tensor(backend):
  a_val = np.ones((5, 3, 4, 4, 5))
  a = tn.Node(a_val, backend=backend)
  e1 = tn.connect(a[0], a[4])
  e2 = tn.connect(a[3], a[2])
  tn.flatten_edges([e2, e1])
  tn.check_correct([a])
  # Check expected values.
  a_final = np.reshape(np.transpose(a_val, (1, 2, 0, 3, 4)), (3, 20, 20))
  np.testing.assert_allclose(a.tensor, a_final)


def test_contract_between_output_order(backend):
  a_val = np.ones((2, 3, 4, 5))
  b_val = np.ones((3, 5, 4, 2))
  c_val = np.ones((2, 2))
  a = tn.Node(a_val, backend=backend)
  b = tn.Node(b_val, backend=backend)
  c = tn.Node(c_val, backend=backend)
  tn.connect(a[0], b[3])
  tn.connect(b[1], a[3])
  tn.connect(a[1], b[0])
  with pytest.raises(ValueError):
    d = tn.contract_between(
        a, b, name="New Node", output_edge_order=[a[2], b[2], a[0]])
  tn.check_correct([a, b, c], check_connections=False)
  with pytest.raises(ValueError):
    d = tn.contract_between(
        a, b, name="New Node", output_edge_order=[a[2], b[2], c[0]])
  tn.check_correct([a, b, c], check_connections=False)
  d = tn.contract_between(a, b, name="New Node", output_edge_order=[b[2], a[2]])
  tn.check_correct([d, c], check_connections=False)
  a_flat = np.reshape(np.transpose(a_val, (2, 1, 0, 3)), (4, 30))
  b_flat = np.reshape(np.transpose(b_val, (2, 0, 3, 1)), (4, 30))
  final_val = np.matmul(b_flat, a_flat.T)
  np.testing.assert_allclose(d.tensor, final_val)
  assert d.name == "New Node"


def test_contract_between_trace_edges(backend):
  a_val = np.ones((3, 3))
  final_val = np.trace(a_val)
  a = tn.Node(a_val, backend=backend)
  tn.connect(a[0], a[1])
  b = tn.contract_between(a, a)
  tn.check_correct([b])
  np.testing.assert_allclose(b.tensor, final_val)


def test_contract_disable(backend):
  a = tn.Node(np.random.rand(2, 2), backend=backend)
  a.disable()
  with pytest.raises(ValueError):
    a.edges[0]
  with pytest.raises(ValueError):
    a.edges
  with pytest.raises(ValueError):
    a.signature
  with pytest.raises(ValueError):
    a.shape


def test_contract_between_dangling_edges(backend):
  a = tn.Node(np.random.rand(2, 2), backend=backend)
  b = tn.Node(np.random.rand(2, 2), backend=backend)
  tn.connect(a[0], b[0])
  out = tn.contract_between(a, b)
  assert all([e.is_dangling() for e in a.edges])
  assert all([e.is_dangling() for e in b.edges])


def test_double_trace_disable(backend):
  node = tn.Node(np.random.rand(2, 2, 2, 2), name='node1', backend=backend)
  e1 = tn.connect(node[0], node[2], name='e1')
  e2 = tn.connect(node[1], node[3], name='e2')
  result = tn.contract(e1, name='b')
  result = tn.contract(e2, name='b')
  assert all([e.is_dangling() for e in node.edges])


def test_trace_disable(backend):
  node = tn.Node(np.random.rand(2, 2, 2, 2), name='node1', backend=backend)
  e1 = tn.connect(node[0], node[2], name='e1')
  result = tn.contract(e1, name='b')
  assert all([e.is_dangling() for e in node.edges])


def test_weakref(backend):
  a = tn.Node(np.eye(2), backend=backend)
  b = tn.Node(np.eye(2), backend=backend)
  e = tn.connect(a[0], b[0])
  del a
  del b
  with pytest.raises(ValueError):
    tn.contract(e)


def test_weakref_complex(backend):
  a = tn.Node(np.eye(2), backend=backend, name='a')
  b = tn.Node(np.eye(2), backend=backend, name='b')
  c = tn.Node(np.eye(2), backend=backend, name='c')
  e1 = tn.connect(a[0], b[0], 'e1')
  e2 = tn.connect(b[1], c[0], 'e2')
  tn.contract(e1)
  # This raises an exception. We didn't store the result
  # of tn.contract(e1) and it got garbage collected, but
  # edge2.node1 is still pointing to the result of tn.contract(e1)
  with pytest.raises(ValueError):
    tn.contract(e2)


def test_set_node2(backend):
  a = tn.Node(np.eye(2), backend=backend)
  b = tn.Node(np.eye(2), backend=backend)
  e = tn.connect(a[0], b[0])
  # You should never do this, but if you do, we should handle
  # it gracefully.
  e.node2 = None
  assert e.is_dangling()


def test_set_default(backend):
  tn.set_default_backend(backend)
  assert tn.config.default_backend == backend
  node = tn.Node(np.ones(1))
  assert node.backend.name == backend


def test_copy_tensor(backend):
  a = tn.Node(np.array([1., 2., 3.]), backend=backend)
  b = tn.Node(np.array([10., 20., 30.]), backend=backend)
  c = tn.Node(np.array([5., 6., 7.]), backend=backend)
  d = tn.Node(np.array([1., -1., 1.]), backend=backend)
  cn = tn.CopyNode(rank=4, dimension=3, backend=backend)
  edge1 = tn.connect(a[0], cn[0])
  edge2 = tn.connect(b[0], cn[1])
  edge3 = tn.connect(c[0], cn[2])
  edge4 = tn.connect(d[0], cn[3])

  result = cn.compute_contracted_tensor()
  assert list(result.shape) == []
  np.testing.assert_allclose(result, 50 - 240 + 630)

  for edge in [edge1, edge2, edge3, edge4]:
    val = tn.contract(edge)
  result = val.tensor
  assert list(result.shape) == []
  np.testing.assert_allclose(result, 50 - 240 + 630)


# Include 'tensorflow' (by removing the decorator) once #87 is fixed.
@pytest.mark.parametrize('backend', ('numpy', 'jax'))
def test_copy_tensor_parallel_edges(backend):
  a = tn.Node(np.diag([1., 2, 3]), backend=backend)
  b = tn.Node(np.array([10, 20, 30]), backend=backend)
  cn = tn.CopyNode(rank=3, dimension=3, backend=backend)
  edge1 = tn.connect(a[0], cn[0])
  edge2 = tn.connect(a[1], cn[1])
  edge3 = tn.connect(b[0], cn[2])

  result = cn.compute_contracted_tensor()
  assert list(result.shape) == []
  np.testing.assert_allclose(result, 10 + 40 + 90)

  for edge in [edge1, edge2, edge3]:
    val = tn.contract(edge)
  result = val.tensor
  assert list(result.shape) == []
  np.testing.assert_allclose(result, 10 + 40 + 90)


def test_contract_copy_node_connected_neighbors(backend):
  a = tn.Node(np.array([[1, 2, 3], [10, 20, 30]]), backend=backend)
  b = tn.Node(np.array([[2, 1, 1], [2, 2, 2]]), backend=backend)
  c = tn.Node(np.array([3, 4, 4]), backend=backend)
  cn = tn.CopyNode(rank=3, dimension=3, backend=backend)
  tn.connect(a[0], b[0])
  tn.connect(a[1], cn[0])
  tn.connect(b[1], cn[1])
  tn.connect(c[0], cn[2])

  n = tn.contract_copy_node(cn)
  assert len(n.edges) == 2
  assert n.edges[0] == n.edges[1]

  val = tn.contract_parallel(n.edges[0])
  result = val.tensor
  assert list(result.shape) == []
  np.testing.assert_allclose(result, 26 + 460)


def test_bad_backend():
  with pytest.raises(ValueError):
    tn.Node(np.ones(1), backend="NOT_A_BACKEND")


def test_at_operator(backend):
  a = tn.Node(np.ones((2,)), backend=backend)
  b = tn.Node(np.ones((2,)), backend=backend)
  tn.connect(a[0], b[0])
  c = a @ b
  assert isinstance(c, tn.Node)
  np.testing.assert_allclose(c.tensor, 2.0)


def test_edge_sorting(backend):
  a = tn.Node(np.eye(2), backend=backend)
  b = tn.Node(np.eye(2), backend=backend)
  c = tn.Node(np.eye(2), backend=backend)
  e1 = tn.connect(a[0], b[1])
  e2 = tn.connect(b[0], c[1])
  e3 = tn.connect(c[0], a[1])
  e1.set_signature(0)
  e2.set_signature(1)
  e3.set_signature(2)
  sorted_edges = sorted([e2, e3, e1])
  assert sorted_edges == [e1, e2, e3]


def test_connect_alias(backend):
  a = tn.Node(np.ones((2, 2)), backend=backend)
  b = tn.Node(np.ones((2, 2)), backend=backend)
  e = a[0] ^ b[0]
  assert set(e.get_nodes()) == {a, b}
  assert e is a[0]
  assert e is b[0]


def test_disconnect_alias(backend):
  a = tn.Node(np.ones((2, 2)), backend=backend)
  b = tn.Node(np.ones((2, 2)), backend=backend)
  e = a[0] ^ b[0]
  tn.check_connected([a, b])
  a[0] | b[0]
  with pytest.raises(ValueError):
    tn.check_connected([a, b])
  assert all([e.is_dangling() for e in a.edges])
  assert all([e.is_dangling() for e in b.edges])


def test_disconnect_flatten(backend):
  a = tn.Node(np.ones((2, 2)), backend=backend)
  b = tn.Node(np.ones((2, 2)), backend=backend)
  e1 = a[0] ^ b[0]
  e2 = a[1] ^ b[1]
  tn.flatten_edges([e1, e2])
  tn.check_connected([a, b])
  a[0] | b[0]
  with pytest.raises(ValueError):
    tn.check_connected([a, b])
  assert all([e.is_dangling() for e in a.edges])
  assert all([e.is_dangling() for e in b.edges])


def test_split_node_full_svd_names(backend):
  a = tn.Node(np.random.rand(10, 10), backend=backend)
  e1 = a[0]
  e2 = a[1]
  left, s, right, _, = tn.split_node_full_svd(
      a, [e1], [e2],
      left_name='left',
      middle_name='center',
      right_name='right',
      left_edge_name='left_edge',
      right_edge_name='right_edge')
  assert left.name == 'left'
  assert s.name == 'center'
  assert right.name == 'right'
  assert left.edges[-1].name == 'left_edge'
  assert s[0].name == 'left_edge'
  assert s[1].name == 'right_edge'
  assert right.edges[0].name == 'right_edge'


def test_split_node_rq_names(backend):
  a = tn.Node(np.zeros((2, 3, 4, 5, 6)), backend=backend)
  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, right = tn.split_node_rq(
      a,
      left_edges,
      right_edges,
      left_name='left',
      right_name='right',
      edge_name='edge')
  assert left.name == 'left'
  assert right.name == 'right'
  assert left.edges[-1].name == 'edge'
  assert right.edges[0].name == 'edge'


def test_split_node_qr_names(backend):
  a = tn.Node(np.zeros((2, 3, 4, 5, 6)), backend=backend)
  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, right = tn.split_node_qr(
      a,
      left_edges,
      right_edges,
      left_name='left',
      right_name='right',
      edge_name='edge')
  assert left.name == 'left'
  assert right.name == 'right'
  assert left.edges[-1].name == 'edge'
  assert right.edges[0].name == 'edge'


def test_split_node_names(backend):
  a = tn.Node(np.zeros((2, 3, 4, 5, 6)), backend=backend)
  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, right, _ = tn.split_node(
      a,
      left_edges,
      right_edges,
      left_name='left',
      right_name='right',
      edge_name='edge')
  assert left.name == 'left'
  assert right.name == 'right'
  assert left.edges[-1].name == 'edge'
  assert right.edges[0].name == 'edge'


def test_split_node_rq(backend):
  a = tn.Node(np.random.rand(2, 3, 4, 5, 6), backend=backend)
  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, right = tn.split_node_rq(a, left_edges, right_edges)
  tn.check_correct([left, right])
  np.testing.assert_allclose(a.tensor, tn.contract(left[3]).tensor)


def test_conj(backend):
  if backend == "pytorch":
    pytest.skip("Complex numbers currently not supported in PyTorch")
  if backend == "jax":
    pytest.skip("Complex QR crashes jax")

  a = tn.Node(np.random.rand(3, 3) + 1j * np.random.rand(3, 3), backend=backend)
  abar = tn.conj(a)
  np.testing.assert_allclose(abar.tensor, a.backend.conj(a.tensor))


def test_split_node_qr_unitarity_complex(backend):
  if backend == "pytorch":
    pytest.skip("Complex numbers currently not supported in PyTorch")
  if backend == "jax":
    pytest.skip("Complex QR crashes jax")

  a = tn.Node(np.random.rand(3, 3) + 1j * np.random.rand(3, 3), backend=backend)
  q, _ = tn.split_node_qr(a, [a[0]], [a[1]])
  q[1].disconnect()
  qbar = tn.conj(q)
  q[1] ^ qbar[1]
  u1 = q @ qbar
  q[0] ^ qbar[0]
  u2 = qbar @ q

  np.testing.assert_almost_equal(u1.tensor, np.eye(3))
  np.testing.assert_almost_equal(u2.tensor, np.eye(3))


def test_split_node_qr_unitarity_float(backend):
  a = tn.Node(np.random.rand(3, 3), backend=backend)
  q, _ = tn.split_node_qr(a, [a[0]], [a[1]])

  q[1].disconnect()
  qbar = tn.conj(q)
  q[1] ^ qbar[1]
  u1 = q @ qbar
  q[0] ^ qbar[0]
  u2 = qbar @ q

  np.testing.assert_almost_equal(u1.tensor, np.eye(3))
  np.testing.assert_almost_equal(u2.tensor, np.eye(3))


def test_split_node_qr(backend):
  a = tn.Node(np.random.rand(2, 3, 4, 5, 6), backend=backend)
  left_edges = []
  for i in range(3):
    left_edges.append(a[i])
  right_edges = []
  for i in range(3, 5):
    right_edges.append(a[i])
  left, right = tn.split_node_qr(a, left_edges, right_edges)
  tn.check_correct([left, right])
  np.testing.assert_allclose(a.tensor, tn.contract(left[3]).tensor)


@pytest.mark.parametrize("backend,dtype", [
    *list(zip(['numpy'] * len(np_dtypes), np_dtypes)),
    *list(zip(['tensorflow'] * len(tf_dtypes), tf_dtypes)),
    *list(zip(['pytorch'] * len(torch_dtypes), torch_dtypes)),
    *list(zip(['jax'] * len(jax_dtypes), jax_dtypes)),
])
def test_node_backend_dtype_1(backend, dtype):
  be = backend_factory.get_backend(backend, dtype)
  n1 = tn.Node(be.zeros((2, 2)), backend=backend)
  assert n1.dtype == dtype
  assert n1.tensor.dtype == dtype


def test_get_all_nodes(backend):
  nodes = [
      tn.Node(
          np.random.rand(2, 2, 2), name='node{}'.format(n), backend=backend)
      for n in range(20)
  ]
  edges = []
  for n in range(100):
    n1 = np.random.randint(len(nodes))
    n2 = np.random.randint(len(nodes))
    #print(nodes[n1].has_dangling_edge())
    if n1 == n2:
      continue
    if nodes[n1].has_dangling_edge() and nodes[n2].has_dangling_edge():
      #print(n1,n2)
      axis1 = np.random.randint(len(nodes[n1].edges))
      while not nodes[n1].edges[axis1].is_dangling():
        axis1 = np.random.randint(len(nodes[n1].edges))
      axis2 = np.random.randint(len(nodes[n2].edges))
      while not nodes[n2].edges[axis2].is_dangling():
        axis2 = np.random.randint(len(nodes[n2].edges))
      edges.append(nodes[n1][axis1] ^ nodes[n2][axis2])
    else:
      continue
  assert {node for node in nodes if node.has_nondangling_edge()
         } == tn.get_all_nodes(edges)
