from typing import Tuple
import numpy as np
from numba import njit, prange
from .autodiff import Context
from .tensor import Tensor
from .tensor_data import MAX_DIMS, Index, Shape, Strides, broadcast_index, index_to_position, to_index
from .tensor_functions import Function
to_index = njit(inline='always')(to_index)
index_to_position = njit(inline='always')(index_to_position)
broadcast_index = njit(inline='always')(broadcast_index)

def _tensor_conv1d(out: Tensor, out_shape: Shape, out_strides: Strides, out_size: int, input: Tensor, input_shape: Shape, input_strides: Strides, weight: Tensor, weight_shape: Shape, weight_strides: Strides, reverse: bool) -> None:
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch_, in_channels, width = input_shape
    out_channels, _, k_width = weight_shape

    # For each output position
    for batch in prange(batch_):
        for out_channel in prange(out_channels):
            for w in prange(width):
                # Sum up all the values of the input * weights
                acc = 0.0
                for in_channel in range(in_channels):
                    for k in range(k_width):
                        w_offset = k if not reverse else k_width - k - 1
                        if w + w_offset < width:
                            # Get input position
                            in_pos = index_to_position((batch, in_channel, w + w_offset), input_strides)
                            # Get weight position
                            w_pos = index_to_position((out_channel, in_channel, k), weight_strides)
                            # Add to accumulator
                            acc += input._tensor._storage[in_pos] * weight._tensor._storage[w_pos]
                # Set output position
                out_pos = index_to_position((batch, out_channel, w), out_strides)
                out._tensor._storage[out_pos] = acc
tensor_conv1d = njit(parallel=True)(_tensor_conv1d)

class Conv1dFun(Function):

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
            batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, width = input.shape
        out_channels, _, k_width = weight.shape
        
        # Create output tensor
        out = input.zeros((batch, out_channels, width))
        
        # Call the conv1d implementation
        tensor_conv1d(
            out,
            out.shape,
            out.strides,
            out.size,
            input,
            input.shape,
            input.strides,
            weight,
            weight.shape,
            weight.strides,
            False,
        )
        return out
conv1d = Conv1dFun.apply

def _tensor_conv2d(out: Tensor, out_shape: Shape, out_strides: Strides, out_size: int, input: Tensor, input_shape: Shape, input_strides: Strides, weight: Tensor, weight_shape: Shape, weight_strides: Strides, reverse: bool) -> None:
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch_, in_channels, height, width = input_shape
    out_channels, _, k_height, k_width = weight_shape

    # For each output position
    for batch in prange(batch_):
        for out_channel in prange(out_channels):
            for h in prange(height):
                for w in prange(width):
                    # Sum up all the values of the input * weights
                    acc = 0.0
                    for in_channel in range(in_channels):
                        for k_h in range(k_height):
                            for k_w in range(k_width):
                                h_offset = k_h if not reverse else k_height - k_h - 1
                                w_offset = k_w if not reverse else k_width - k_w - 1
                                if h + h_offset < height and w + w_offset < width:
                                    # Get input position
                                    in_pos = index_to_position(
                                        (batch, in_channel, h + h_offset, w + w_offset),
                                        input_strides,
                                    )
                                    # Get weight position
                                    w_pos = index_to_position(
                                        (out_channel, in_channel, k_h, k_w),
                                        weight_strides,
                                    )
                                    # Add to accumulator
                                    acc += input._tensor._storage[in_pos] * weight._tensor._storage[w_pos]
                    # Set output position
                    out_pos = index_to_position((batch, out_channel, h, w), out_strides)
                    out._tensor._storage[out_pos] = acc
tensor_conv2d = njit(parallel=True, fastmath=True)(_tensor_conv2d)

class Conv2dFun(Function):

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, height, width = input.shape
        out_channels, _, k_height, k_width = weight.shape
        
        # Create output tensor
        out = input.zeros((batch, out_channels, height, width))
        
        # Call the conv2d implementation
        tensor_conv2d(
            out,
            out.shape,
            out.strides,
            out.size,
            input,
            input.shape,
            input.strides,
            weight,
            weight.shape,
            weight.strides,
            False,
        )
        return out
conv2d = Conv2dFun.apply