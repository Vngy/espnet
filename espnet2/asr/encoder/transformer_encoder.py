# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
from typing import Optional
from typing import Tuple

import echotorch.nn as etnn

import echotorch.nn.reservoir as etnn
from echotorch.utils.matrix_generation import matrix_factory
import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import check_short_utt
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet2.asr.encoder.abs_encoder import AbsEncoder


class TransformerEncoder(AbsEncoder):
    """Transformer encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        res_size: int = 200,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size
        self.input_layer = input_layer
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, output_size, dropout_rate, conv_transpose=True)
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(input_size, output_size, dropout_rate, conv_transpose=True)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(input_size, output_size, dropout_rate)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "echo":
            self.reservoir = True
            print("In Sz: ", input_size , "Out Sz: ", output_size)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            res_out = 192
            self.res_out = res_out #define res_out
            w_connectivity = 0.85
            win_connectivity = 0.50
            spectral_radius = 0.9
            leak_rate = 0.95
            input_scaling = 1.0
            num_layers = 1
            bias_scaling = 0.1
            deep_esn_type = 'IF'       
            output_dim = output_size
            output_layer = False
            _dtype = torch.float32
            

            
            #Generate weight matrices
            w_generator = matrix_factory.get_generator(name='normal', spectral_radius=spectral_radius, connectivity=w_connectivity)
            win_generator = matrix_factory.get_generator(name='normal', connectivity=win_connectivity)
            wbias_generator = matrix_factory.get_generator(name='normal', scale=bias_scaling)
            
            
            '''
            self.embed = torch.nn.Sequential(etnn.DeepESN(n_layers = num_layers,
                                      input_dim = input_size,
                                      hidden_dim = hidden_dim,
                                      output_dim = output_size,
                                      leak_rate = leak_rate,
                                      w_generator = w_generator,
                                      win_generator = win_generator,
                                      wbias_generator = wbias_generator,
                                      input_scaling = input_scaling,
                                      input_type = deep_esn_type,
                                      create_output=output_layer), pos_enc_class(output_size, positional_dropout_rate, pos_trans=True)).to(self.device)
            '''

            w = w_generator.generate(size = (output_dim, output_dim), dtype=_dtype)
            w_in = win_generator.generate(size=(output_dim, input_size), dtype=_dtype)
            w_bias = wbias_generator.generate(size = output_dim, dtype=_dtype)



            self.embed = torch.nn.Sequential(etnn.ESNCell(input_dim = input_size,
                                                            output_dim = output_dim,
                                                            w = w,
                                                            w_in = w_in,
                                                            w_bias = w_bias,
                                                            input_scaling = input_scaling),
                                            pos_enc_class(output_dim, positional_dropout_rate, pos_trans=True)).to(self.device)

            print("Initialized Echo State Embedding Layer")
        elif input_layer == "echoConv": #@todo add in convs here
            self.embed = torch.nn.Sequential(etnn.ESNCell(input_dim=input_size, output_dim=output_size, spectral_radius=0.9))    
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                MultiHeadedAttention(
                    attention_heads, output_size, attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def get_esn_state(self):
        return self.esn_state

    def get_input_layer(self):
        return self.input_layer

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        print("Processing Encoder Forward")
        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad.to(self.device)
            print("x padded before embed: ", xs_pad.shape)
            #xs_pad = self.embed[0](xs_pad, xs_pad) #try transposing?
            xs_pad = self.embed[0](xs_pad) #try transposing?
            print("x padded after embed: ", xs_pad.shape)
            self.esn_state = xs_pad.transpose(1, 2).unsqueeze(1)
            xs_pad = self.embed[1](xs_pad.transpose(1,2))
        xs_pad, masks = self.encoders(xs_pad, masks)
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens, None
