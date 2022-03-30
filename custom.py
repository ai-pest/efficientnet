# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
# Copyright 2022 Northern System Service Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================
"""Implementation of tf.metrics module."""

from tensorflow.python.ops.metrics_impl import *


def metric_sum_tensor(values, name=None):
  """Computes the element-wise sum of the given tensors.
  This function returns an average tensor with the same shape as the
  input tensors.
  The `sum_tensor` function creates a local variable called `total_tensor` that 
  are used to compute the sum of `values`. This sum is ultimately returned as `sum`
   which is an idempotent operation that simply sums all `values` up.
  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `sum`.
  `update_op` increments `total` with the sum of `values`.
  Args:
    values: A `Tensor` of arbitrary dimensions.
    name: An optional variable_scope name.
  Returns:
    sum: A float `Tensor` representing the current sum, the value of `total`.
    update_op: An operation that increments the `total` variable appropriately 
      and whose value matches `total_tensor`.
  Raises:
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.mean_tensor is not supported when '
                       'eager execution is enabled.')

  with variable_scope.variable_scope(name, 'sum', (values)):
    values = math_ops.cast(values, dtypes.float32)
    total = metric_variable(
        values.get_shape(), dtypes.float32, name='total_tensor')

    num_values = array_ops.ones_like(values)

    update_op = state_ops.assign_add(total, values)

    compute_sum = lambda _, t: t

    sum_t = _aggregate_across_replicas(
        [], compute_sum, total)

    return sum_t, update_op

