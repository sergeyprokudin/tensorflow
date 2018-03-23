# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""The Von Mises (circular normal) distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
# get rid of this dependency!!!
import numpy as np

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


__all__ = [
    "VonMises",
    "VonMisesWithSoftplusKappa",
]


class VonMises(distribution.Distribution):
  """The Von Mises distribution with location `loc` and `kappa` parameters.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; mu, kappa) = exp(kappa * cos(x-mu) ) / Z
  Z = 2 pi I_0(kappa)

  where I_0(kappa) is the modified Bessel function of order 0
  ```

  where `loc = mu` is the mean, `scale = sigma` is the std. deviation, and, `Z`
  is the normalization constant.

  The von Mises distribution (also known as the circular normal distribution) is a
  continuous probability distribution on the unit circle. It may be thought of as
  the circular analogue of the normal distribution.

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  # Define a single scalar von Mises distribution.
  dist = tf.contrib.distributions.VonMises(loc=0., kappa=3.)


  # Define a batch of two scalar valued von Mises.
  # The first has mean 1 and standard deviation 11, the second 2 and 22.
  dist = tf.contrib.distributions.VonMises(loc=[1, 2.], scale=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  dist.prob([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  dist.sample([3])
  ```

  Arguments are broadcast when possible.

  ```python
  # Define a batch of two scalar valued Normals.
  # Both have mean 1, but different standard deviations.
  dist = tf.contrib.distributions.VonMises(loc=1., scale=[11, 22.])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.0)
  ```

  """

  def __init__(self,
               loc,
               kappa,
               validate_args=False,
               allow_nan_stats=True,
               name="VonMises"):
    """Construct von Mises distributions with mean and kappa `loc` and `kappa`.

    Args:
      loc: Floating point tensor; the means of the distribution(s).
      kappa: Floating point tensor; the inverse variance of the distribution(s).
        Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `loc` and `kappa` have different `dtype`.
    """
    parameters = locals()
    with ops.name_scope(name, values=[loc, kappa]):
      with ops.control_dependencies([check_ops.assert_positive(kappa)] if
                                    validate_args else []):
        self._loc = array_ops.identity(loc, name="loc")
        self._kappa = array_ops.identity(kappa, name="kappa")
        contrib_tensor_util.assert_same_float_dtype([self._loc, self._kappa])
    super(VonMises, self).__init__(
        dtype=self._kappa.dtype,
        reparameterization_type=distribution.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._loc, self._kappa],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("loc", "kappa"), ([ops.convert_to_tensor(
            sample_shape, dtype=dtypes.int32)] * 2)))

  @property
  def loc(self):
    """Distribution parameter for the mean."""
    return self._loc

  @property
  def kappa(self):
    """Distribution parameter for dispersion."""
    return self._kappa

  def _batch_shape_tensor(self):
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.loc),
        array_ops.shape(self.kappa))

  def _batch_shape(self):
    return array_ops.broadcast_static_shape(
        self.loc.get_shape(),
        self.kappa.get_shape())

  def _event_shape_tensor(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    # sample from von Mises distribution
    raise NotImplementedError("Sampling from von Mises not implemented.")

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _log_unnormalized_prob(self, x):
    return self.kappa * math_ops.cos(x - self.loc)

  def _log_normalization(self):
    return 0.5 * math.log(2. * math.pi) + self._log_bessel_approx_tf(self.kappa)

  def _log_bessel_approx_tf(self, x):

    def _log_bessel_approx_0(x):
        # change to tf.constant!!!
        bessel_taylor_coefs = np.asarray([1.00000000e+00, 2.50000000e-01, 1.56250000e-02,
                                          4.34027778e-04, 6.78168403e-06], dtype='float32')
        m = bessel_taylor_coefs.shape[0]
        deg = array_ops.reshape(math_ops.range(0, m, 1)*2, [1, -1])
        n_rows = array_ops.shape(x)[0]
        x_tiled = array_ops.tile(x, [1, m])
        deg_tiled = array_ops.tile(deg, [n_rows, 1])
        coef_tiled = array_ops.tile(bessel_taylor_coefs[0:m].reshape(1, m), [n_rows, 1])
        val = math_ops.log(math_ops.reduce_sum(math_ops.pow(x_tiled, math_ops.to_float(deg_tiled))*coef_tiled, axis=1))
        return array_ops.reshape(val, [-1, 1])

    def _log_bessel_approx_large(x):
        return x - 0.5*math_ops.log(2*math.pi*x)

    bessel0 = array_ops.where(x > 5.0, _log_bessel_approx_large(x), _log_bessel_approx_0(x))

    return bessel0

  def _log_cdf(self, x):
    raise NotImplementedError("Von Mises log_cdf is not implemented.")

  def _cdf(self, x):
    raise NotImplementedError("Von Mises cdf is not implemented.")


class VonMisesWithSoftplusKappa(VonMises):
  """Von Mises with softplus applied to `kappa`."""

  def __init__(self,
               loc,
               kappa,
               validate_args=False,
               allow_nan_stats=True,
               name="NormalWithSoftplusScale"):
    parameters = locals()
    with ops.name_scope(name, values=[kappa]):
      super(VonMisesWithSoftplusKappa, self).__init__(
          loc=loc,
          kappa=nn.softplus(kappa, name="softplus_kappa"),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
    self._parameters = parameters
