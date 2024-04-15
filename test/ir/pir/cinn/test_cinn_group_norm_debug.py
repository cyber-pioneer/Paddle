# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np
import utils

import paddle
from paddle.base import core

# paddle.device.set_device("cpu")


class GroupNormSubGraph(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, bias, data_format="NHWC"):
        return paddle.nn.functional.group_norm(
            x,
            num_groups=32,
            epsilon=1e-6,
            weight=weight,
            bias=bias,
            data_format=data_format,
        )


class TestGroupNormSubGraph(unittest.TestCase):
    def setUp(self):
        np.random.seed(2024)
        self.shape = [8, 128, 256, 128]
        self.dtype = "float32"
        self.compute_base()
        self.prepare_data()

    def compute_base(self):
        self.np_x = np.random.random(self.shape).astype("float64")
        self.np_weight = np.random.random([128]).astype("float64")
        self.np_bias = np.random.random([128]).astype("float64")
        self.x = paddle.to_tensor(self.np_x, stop_gradient=False)
        self.weight = paddle.to_tensor(self.np_weight, stop_gradient=False)
        self.bias = paddle.to_tensor(self.np_bias, stop_gradient=False)
        (
            self.refer_out,
            self.refer_x_grad,
            self.refer_weight_grad,
            self.refer_bias_grad,
        ) = self.eval(False)

    def prepare_data(self):
        self.x = paddle.to_tensor(
            self.np_x, dtype=self.dtype, stop_gradient=False
        )
        self.weight = paddle.to_tensor(
            self.np_weight, dtype=self.dtype, stop_gradient=False
        )
        self.bias = paddle.to_tensor(
            self.np_bias, dtype=self.dtype, stop_gradient=False
        )

    def eval(self, use_cinn):
        if self.x.grad is not None:
            self.x.clear_grad()
            self.weight.clear_grad()
            self.bias.clear_grad()

        if use_cinn:
            core._set_prim_all_enabled(True)
        net = GroupNormSubGraph()
        net = utils.apply_to_static(net, use_cinn=False)
        out = net(self.x, self.weight, self.bias)
        loss = out.sum()
        loss.backward()

        core._set_prim_all_enabled(False)
        return (
            out,
            self.x.gradient(),
            self.weight.gradient(),
            self.bias.gradient(),
        )

    # def _test_eval(self):
    #     cinn_out, cinn_x_grad, cinn_weight_grad, cinn_bias_grad = self.eval(
    #         use_cinn=False
    #     )
    #     np.testing.assert_allclose(
    #         cinn_out, self.refer_out, atol=5e-3, rtol=5e-3
    #     )
    #     np.testing.assert_allclose(
    #         cinn_x_grad, self.refer_x_grad, atol=5e-3, rtol=5e-3
    #     )
    #     np.testing.assert_allclose(
    #         cinn_weight_grad, self.refer_weight_grad, atol=5e-3, rtol=5e-3
    #     )
    #     np.testing.assert_allclose(
    #         cinn_bias_grad, self.refer_bias_grad, atol=5e-3, rtol=5e-3
    #     )

    def test_eval(self):
        cinn_out, cinn_x_grad, cinn_weight_grad, cinn_bias_grad = self.eval(
            use_cinn=True
        )
        out, dy_x_grad, dy_weight_grad, dy_bias_grad = self.eval(use_cinn=False)
        np.testing.assert_allclose(cinn_out, out, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(cinn_x_grad, dy_x_grad, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(
            cinn_weight_grad, dy_weight_grad, atol=1e-5, rtol=2e-5
        )
        np.testing.assert_allclose(
            cinn_bias_grad, dy_bias_grad, atol=1e-3, rtol=2e-3
        )


if __name__ == '__main__':
    unittest.main()
