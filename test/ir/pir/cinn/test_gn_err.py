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

paddle.seed(2024)


# paddle.device.set_device("cpu")

# GLOG_vmodule=compiler=3 FLAGS_pir_apply_shape_optimization_pass=0 FLAGS_enable_pir_api=1 FLAGS_prim_enable_dynamic=fasle FLAGS_cinn_new_group_scheduler=1 FLAGS_group_schedule_tiling_first=1 FLAGS_cinn_bucket_compile=True FLAGS_support_reduce_stride_read=1 python test_cinn_group_norm_prim.py


class GroupNormSubGraph(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, bias):
        return paddle.nn.functional.group_norm(
            x,
            num_groups=32,
            epsilon=1e-6,
            weight=weight,
            bias=bias,
            data_format="NHWC",
        )


class TestGroupNormSubGraph(unittest.TestCase):
    def setUp(self):
        self.prepare_data()
        (
            self.base_out,
            self.base_x_grad,
            self.base_weight_grad,
            self.base_bias_grad,
        ) = self.eval(use_prim=False, dtype="float64")

    def prepare_data(self):
        self.half_dtype = "float16"
        self.np_x = np.random.random([4, 128, 256, 128])
        self.np_weight = np.random.random([128])
        self.np_bias = np.random.random([128])

    def eval(self, use_prim, dtype="float64"):
        if dtype == "float16":
            self.x = paddle.to_tensor(self.np_x, dtype="float16")
            self.weight = paddle.to_tensor(self.np_weight, dtype="float16")
            self.bias = paddle.to_tensor(self.np_bias, dtype="float16")
        else:
            self.x = paddle.to_tensor(self.np_x, dtype="float64")
            self.weight = paddle.to_tensor(self.np_weight, dtype="float64")
            self.bias = paddle.to_tensor(self.np_bias, dtype="float64")
        self.x.stop_gradient = False
        self.weight.stop_gradient = False
        self.bias.stop_gradient = False

        if self.x.grad is not None:
            self.x.clear_grad()
            self.weight.clear_grad()
            self.bias.clear_grad()

        if use_prim:
            core._set_prim_all_enabled(True)
        net = GroupNormSubGraph()
        # net.eval()
        net = utils.apply_to_static(net, False)

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

    def test_eval1(self):
        cinn_out, cinn_x_grad, cinn_weight_grad, cinn_bias_grad = self.eval(
            use_prim=True, dtype="float16"
        )
        dy_out, dy_x_grad, dy_weight_grad, dy_bias_grad = self.eval(
            use_prim=False, dtype="float16"
        )
        print(" cinn_out************** ", cinn_out)
        print(" dy_out************** ", dy_out)
        np.testing.assert_allclose(self.base_x_grad, dy_x_grad, atol=1e-2)
        print(" cinn_weight_grad ===== ", cinn_weight_grad)
        print(" dy_weight_grad ===== ", dy_weight_grad)
        np.testing.assert_allclose(
            cinn_weight_grad, dy_weight_grad, rtol=0, atol=1e-16
        )
        np.testing.assert_allclose(cinn_bias_grad, dy_bias_grad, atol=1e-16)

    def test_eval2(self):
        base_out, base_x_grad, base_weight_grad, base_bias_grad = self.eval(
            use_prim=False, dtype="float64"
        )
        cinn_out, cinn_x_grad, cinn_weight_grad, cinn_bias_grad = self.eval(
            use_prim=True, dtype="float16"
        )
        dy_out, dy_x_grad, dy_weight_grad, dy_bias_grad = self.eval(
            use_prim=False, dtype="float16"
        )
        print(" cinn_out************** ", cinn_out)
        print(" dy_out************** ", dy_out)
        np.testing.assert_allclose(self.base_x_grad, dy_x_grad, atol=1e-2)
        print(" cinn_weight_grad ===== ", cinn_weight_grad)
        print(" dy_weight_grad ===== ", dy_weight_grad)
        np.testing.assert_allclose(
            cinn_weight_grad, dy_weight_grad, rtol=0, atol=1e-16
        )
        np.testing.assert_allclose(cinn_bias_grad, dy_bias_grad, atol=1e-16)


if __name__ == '__main__':
    unittest.main()
