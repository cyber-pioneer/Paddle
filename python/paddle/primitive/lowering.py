# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import typing

from paddle import ir
from paddle.fluid import core
from paddle.fluid.libpaddle.ir import Block, Program

# To be removed in future
from paddle.incubate.autograd.composite_rules import _composite
from paddle.incubate.autograd.primreg import lookup_composite


def _as_tensors(xs):
    # breakpoint()
    if isinstance(xs, ir.OpResult):
        return (xs,)
    elif isinstance(xs, typing.Sequence):
        return tuple(xs)
    else:
        return xs


def _prepare_python_api_arguments(op):
    """For standard api of operator, its inputs should keep consistent with organization of its inputs and attrs."""
    op_inputs = [x.source() for x in op.operands()]
    # breakpoint()

    # Todo: api to get all attr values
    # op_attrs_dict = op.attrs()

    # op_attrs_name = op.get_attr_names()
    # op_attrs = [op_attrs_dict[x] for x in op_attrs_name]
    api_arguments = op_inputs
    return api_arguments


def _check_op_results(op_name, orig_outs, new_outs):
    assert len(orig_outs) == len(new_outs), (
        f'when replace origin op {op_name} with composite rule, num of origin outs should be equal to new outs, '
        f'but len(orig_outs) = {len(orig_outs)} and len(new_outs) = {len(new_outs)}'
    )

    for orig_out, new_out in zip(
        orig_outs,
        new_outs,
    ):
        if (orig_out is None or new_out is None) and (
            op_name not in core.ops_contain_none
        ):
            raise ValueError(
                f"op {op_name} should not contain any None value. original outs={orig_outs} and its composite rule outs={new_outs}"
            )
        if orig_out is None:
            # to keep same as phi op definition, orig_out may receive None
            continue
        elif new_out is not None:
            orig_dtype = ir.get_op_result_dtype(orig_out)
            new_dtype = ir.get_op_result_dtype(new_out)
            orig_shape = ir.get_op_result_shape(orig_out)
            new_shape = ir.get_op_result_shape(new_out)
            assert orig_dtype == new_dtype, (
                f'when replace origin op {op_name} with composite rule, origin out dtype should be equal to new out dtype, '
                f'but orig_out: {orig_out.name}.dtype={orig_dtype} and new_out: {new_out.name}.dtype={new_dtype}'
            )
            assert (
                -1 not in new_shape
            ), f'when replace origin op {op_name} with composite rule, composite out shape has -1.'
            # breakpoint()
            # assert orig_shape == new_shape, (
            #     f'when replace origin op {op_name} with composite rule, origin out shape should be equal to new out shape, '
            #     f'but orig_out: {orig_out.name}.shape={orig_shape} and new_out: {new_out.name}.shape={new_shape}'
            # )
            assert not (orig_out is None) ^ (
                new_out is None
            ), "orig_out and new_out should match."
        return


def lowering(
    program,
    blacklist=frozenset(),
    whitelist=frozenset(),
):
    """Search nonbasic ops which have be registered composite rules and replace them with primitive ops.
    The operators in blacklist will be excluded from program when decomposed into primitives, and only the
    operators in whitelist will be decomposed. The priority of blacklist is higher than whitelist, it means
    an operator both in blacklist and whitelist will not be decomposed.

    The finally set that will be decomposed is:
        (block.ops & ops have decomposite rule & whitelist) - blacklist

    Args:
        program (Program): The program to be processed.
        blacklist (frozenset): The Operators that will be exclude when decomposed into primitives.
        whitelist (frozenset): Only the operators in whitelist will be decomposed into primitives.
    """
    if not isinstance(program, Program):
        raise TypeError(f"Expect type Program, but got type {type(program)}.")
    block = program.block()

    if not isinstance(blacklist, (set, frozenset)):
        raise TypeError(
            f'Expected type of blacklisst is set|frozenset, but got {type(blacklist)}.'
        )
    if not isinstance(whitelist, (set, frozenset)):
        raise TypeError(
            f'Expected type of whiltelist is set|frozenset, but got {type(whitelist)}.'
        )

    blacklist = core.prim_config["forward_blacklist"] | blacklist

    logging.debug("Decompose composite forward ops begin...")

    if len(blacklist) > 0 and len(whitelist) > 0:
        op_filter = (
            lambda x: x.name() in whitelist and x.name() not in blacklist
        )
    elif len(blacklist) > 0 and len(whitelist) == 0:
        op_filter = lambda x: x.name() not in blacklist
    elif len(blacklist) == 0 and len(whitelist) > 0:
        op_filter = lambda x: x.name() in whitelist
    else:
        op_filter = lambda x: True
    with ir.core.program_guard(program):
        _lowering_subgraph(
            block,
            op_filter,
        )
    replace_ops = core.prim_config["composite_ops_record"]
    logging.debug(f"Decompose composite forward ops finish: {replace_ops}")


def _lowering_subgraph(block, op_filter):
    """The operators in block wich satisfy the filter conditon will be decomposed into primitives."""

    if isinstance(block, Block):
        # Todo1:
        # temp solution: python rule consisting of new ir prim api.
        # formal solution: c++ rule consisting of new ir prim api.
        lower_fn = _composite
        lookup_fn = lookup_composite

        # Todo2:
        # if output var of composite rule is None, this means this var is not needed
        # new ir should cover such case
        none_vars_to_remove = set()

        change = None

        # Todo3:
        # How to handle index of ops after inserting ops in new ir?

        ops_list = block.get_ops()

        lower = False  # Flag of routing to lower or copy branch
        # Step2: Process all ops in the target block
        input_args = _prepare_python_api_arguments(ops_list[1])
        # breakpoint()
        for idx, op in enumerate(ops_list):
            op_name = op.name()

            lower = (lookup_fn(op_name) is not None) and op_filter(op)

            if lower:
                change = True
                core.prim_config["composite_ops_record"].add(op_name)
                # Todo4:
                # How to get indeed dict of inputs, attrs, outputs of origin op and map them to composite rule?
                input_args = _prepare_python_api_arguments(op)

                ir.set_insertion_point(op)
                orig_outs = op.results()
                new_outs = _as_tensors(lower_fn(op, *input_args))
                # new_outs = _as_tensors(paddle.mean(*input_args))
                # check dtype and shape
                _check_op_results(op_name, orig_outs, new_outs)

                op.replace_all_uses_with(new_outs)
                block.remove_op(op)

        # Todo7:
        # to be determined whether to be removed
        # for op in block.ops:
        #     if op._has_kernel(op.desc.type()):
        #         op.desc.infer_var_type(block.desc)
        #         op.desc.infer_shape(block.desc)

        # Todo8:
        # composite ops may contain other composite ops, thus, call _lower_composite again.
        # Indeed, recursive call will be done inside composite rule?
        if change:
            _lowering_subgraph(block, op_filter)
        return

    elif isinstance(block, typing.Sequence):
        for item in block:
            _lowering_subgraph(item, op_filter)
        return
    else:
        raise TypeError
