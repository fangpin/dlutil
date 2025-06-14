import torch
import copy

from fast_pytorch_kmeans import KMeans
from collections import namedtuple


def get_quantized_range(bitwidth: int):
    q_min = -(1 << (bitwidth - 1))
    q_max = (1 << (bitwidth - 1)) - 1
    return q_min, q_max


def liner_quantize(
    fp_tensor: torch.Tensor, bitwidth, dtype=torch.int8
) -> tuple[torch.Tensor, float, int]:
    """
    linear quantization for feature tensor
    :param fp_tensor: [torch.(cuda.)Tensor] floating feature to be quantized
    :param bitwidth: [int] quantization bit width
    :return:
        [torch.(cuda.)Tensor] quantized tensor
        [float] scale tensor
        [int] zero point
    """
    scale, zero = get_scale_and_zero(fp_tensor, bitwidth)
    return (
        liner_quantize_with_scale_zero(fp_tensor, bitwidth, scale, zero, dtype),
        scale,
        zero,
    )


def liner_quantize_with_scale_zero(
    fp_tensor: torch.Tensor, bitwidth: int, scale, zero_point, dtype=torch.int8
):
    """
    quantize tensor using linear quantization
    :param fp_tensor:
    :param bitwidth: [int] quantization bit width, default=4
    :return:
        [torch.(cuda.)LongTensor] quantized tensor
    """
    assert fp_tensor.dtype == torch.float
    assert isinstance(scale, float) or (
        scale.dtype == fp_tensor.dtype and scale.dim() == fp_tensor.dim()
    )
    assert isinstance(zero_point, int) or (
        zero_point.dtype == dtype and zero_point.dim() == fp_tensor.dim()
    )

    scaled_tensor = fp_tensor / scale
    rouned_tensor = scaled_tensor.round()
    rouned_tensor = rouned_tensor.to(dtype)
    shift_tensor = rouned_tensor + zero_point

    q_min, q_max = get_quantized_range(bitwidth)
    return shift_tensor.clamp(q_min, q_max)


def get_scale_and_zero(fp_tensor: torch.Tensor, bitwidth: int) -> tuple[float, int]:
    """
    get quantization scale for single tensor
    :param fp_tensor: [torch.(cuda.)Tensor] floating tensor to be quantized
    :param bitwidth: [int] quantization bit width
    :return:
        [float] scale
        [int] zero_point
    """
    q_min, q_max = get_quantized_range(bitwidth)
    r_min = fp_tensor.min().item()
    r_max = fp_tensor.max().item()
    scale = (r_max - r_min) / (q_max - q_min)

    zero = q_min - r_min / scale
    if zero < q_min:
        zero = q_min
    if zero > q_max:
        zero = q_max
    return scale, int(round(zero))


def get_quantization_scale_for_weight(weight: torch.Tensor, bitwidth: int) -> float:
    """
    get quantization scale for weight tensor, suppose the zero point is zero because, in most cases, the weights satisify the norm distribution.
    :param weight: [torch.(cuda.)Tensor] weight tensor to be quantized
    :param bitwidth: [int] quantization bit width
    :return:
        [float] scale
    """
    fp_max = max(weight.abs().max().item(), 5e-7)
    _, q_max = get_quantized_range(bitwidth)
    return fp_max / q_max


def liner_quantize_weight_per_output_channel(
    weight: torch.Tensor, bitwidth: int
) -> tuple[torch.Tensor, float | torch.Tensor, int]:
    """
    linear quantization for weight tensor
        using different scales and zero_points for different output channels
    :param tensor: [torch.(cuda.)Tensor] floating weight to be quantized
    :param bitwidth: [int] quantization bit width
    :return:
        [torch.(cuda.)Tensor] quantized tensor
        [torch.(cuda.)Tensor] scale tensor
        [int] zero point (which is always 0)
    """
    n_dim = 4
    output_channel_dim = 0
    assert weight.dim() == n_dim, "weight tensor must be 4d"
    assert weight.dtype == torch.float

    scales = torch.zeros(n_dim, device=weight.device)
    for i in range(n_dim):
        w = weight.select(output_channel_dim, i)
        s = get_quantization_scale_for_weight(w, bitwidth)
        scales[i] = s
    scales = scales.view([-1] + [1] * (n_dim - 1))
    q_weight = liner_quantize_with_scale_zero(weight, bitwidth, scales, 0)
    return q_weight, scales, 0


# above is all about quantization
# below is all about inference based on quantized weights


def liner_quantize_bias_per_output_channel(
    bias: torch.Tensor, weight_scale: float | torch.Tensor, input_scale: float
) -> tuple[torch.Tensor, float | torch.Tensor, int]:
    """
    linear quantization for single bias tensor
        quantized_bias = fp_bias / bias_scale
    :param bias: [torch.FloatTensor] bias weight to be quantized
    :param weight_scale: [float or torch.FloatTensor] weight scale tensor
    :param input_scale: [float] input scale
    :return:
        [torch.IntTensor] quantized bias tensor
        [float] quantized bias scale
        [float] quantized bias zero (which is always 0)
    """
    assert bias.dim() == 1
    assert bias.dtype == torch.float
    if isinstance(weight_scale, torch.Tensor):
        assert weight_scale.dtype == torch.float
        weight_scale = weight_scale.view(-1)
        assert bias.numel() == weight_scale.numel()

    bias_scale = input_scale * weight_scale
    bias_quantization = liner_quantize_with_scale_zero(
        bias, 32, bias_scale, 0, dtype=torch.int32
    )
    return bias_quantization, bias_scale, 0


def shift_quantized_liner_bias(
    q_bias: torch.Tensor, q_weight: torch.Tensor, input_zero_point: int
):
    """
    shift quantized bias to incorporate input_zero_point for nn.Linear
        shifted_quantized_bias = quantized_bias - Linear(input_zero_point, quantized_weight)
    :param quantized_bias: [torch.IntTensor] quantized bias (torch.int32)
    :param quantized_weight: [torch.CharTensor] quantized weight (torch.int8)
    :param input_zero_point: [int] input zero point
    :return:
        [torch.IntTensor] shifted quantized bias tensor
    """
    assert q_bias.dtype == torch.int32
    return q_bias - q_weight.sum(1).to(torch.int32) * input_zero_point


def quantized_liner_infer(
    q_input: torch.Tensor,
    q_weight,
    s_q_bias,
    feature_bitwidth: int,
    output_zero_point: int,
    input_scale: float,
    weight_scale,
    output_scale: float,
):
    """
    quantized fully-connected layer
    :param input: [torch.CharTensor] quantized input (torch.int8)
    :param weight: [torch.CharTensor] quantized weight (torch.int8)
    :param bias: [torch.IntTensor] shifted quantized bias
    :param feature_bitwidth: [int] quantization bit width of input and output
    :param weight_bitwidth: [int] quantization bit width of weight
    :param input_scale: [float] input feature scale
    :param weight_scale: [torch.FloatTensor] weight per-channel scale
    :param output_scale: [float] output feature scale
    :return:
        [torch.CharIntTensor] quantized output feature (torch.int8)
    """
    assert q_input.dtype == torch.int8
    assert q_weight.dtype == torch.int8
    assert s_q_bias is None or s_q_bias.dtype == torch.int32
    assert weight_scale.dtype == torch.float

    if "cpu" in q_input.device.type:
        output = torch.nn.functional.linear(
            q_input.to(torch.int32), q_weight.to(torch.int32), s_q_bias
        )
    else:
        # current version pytorch does not yet support integer-based linear() on GPUs
        output = torch.nn.functional.linear(
            q_input.float(), q_weight.float(), s_q_bias.float()
        )
    output = output * input_scale * weight_scale / output_scale + output_zero_point

    output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
    return output


def shift_quantized_conv2d_bias(
    q_bias: torch.Tensor, q_weight: torch.Tensor, input_zero_point: int
) -> torch.Tensor:
    """
    shift quantized bias to incorporate input_zero_point for nn.Conv2d
        shifted_quantized_bias = quantized_bias - Conv(input_zero_point, quantized_weight)
    :param quantized_bias: [torch.IntTensor] quantized bias (torch.int32)
    :param quantized_weight: [torch.CharTensor] quantized weight (torch.int8)
    :param input_zero_point: [int] input zero point
    :return:
        [torch.IntTensor] shifted quantized bias tensor
    """
    assert q_bias.dtype == torch.int32
    return q_bias - q_weight.sum((1, 2, 3)).to(torch.int32) * input_zero_point


def quantized_conv2d_infer(
    q_input: torch.Tensor,
    q_weight,
    q_bias,
    feature_bitwidth: int,
    input_zero_point: int,
    output_zero_point: int,
    input_scale: float,
    weight_scale,
    output_scale: float,
    stride: int,
    padding,
    dilation: int,
    groups: int,
) -> torch.Tensor:
    """
    quantized 2d convolution
    :param q_input: [torch.CharTensor] quantized input (torch.int8)
    :param q_weight: [torch.CharTensor] quantized weight (torch.int8)
    :param q_bias: [torch.IntTensor] shifted quantized bias or None (torch.int32)
    :param feature_bitwidth: [int] quantization bit width of input and output
    :param input_zero_point: [int] input zero point
    :param output_zero_point: [int] output zero point
    :param input_scale: [float] input feature scale
    :param weight_scale: [torch.FloatTensor] weight per-channel scale
    :param output_scale: [float] output feature scale
    :return:
        [torch.(cuda.)CharTensor] quantized output feature
    """
    assert len(padding) == 4
    assert q_input.dtype == torch.int8
    assert q_weight.dtype == q_input.dtype
    assert weight_scale.dtype == torch.float

    input = torch.nn.functional.pad(q_input, padding, "constant", input_zero_point)
    if "cpu" in q_input.device.type:
        output = torch.nn.functional.conv2d(
            input.to(torch.int32),
            q_weight.to(torch.int32),
            q_bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    else:
        # current version pytorch does not yet support integer-based conv2d() on GPUs
        output = torch.nn.functional.conv2d(
            input.float(),
            q_weight.float(),
            q_bias.float(),
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
        )
        output = output.round().to(torch.int32)
    output = output + q_bias.view(1, -1, 1, 1)
    output = output * input_scale * weight_scale / output_scale + output_zero_point
    output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
    return output


def fuse_conv_bn(conv, bn):
    assert conv.bias is None
    factor = bn.weight.data / torch.sqrt(bn.running_var.data + bn.eps)
    conv.weight.data = conv.weight.data = conv.weight.data * factor.reshape(-1, 1, 1, 1)
    conv.bias = torch.nn.Parameter(-bn.running_mean.data * factor + bn.bias.data)
    return conv


def model_fuse_conv_bn(model: torch.nn.Sequential) -> torch.nn.Sequential:
    """we will fuse a BatchNorm layer into its previous convolutional layer, which is a standard practice before quantization. Fusing batchnorm reduces the extra multiplication during inference."""
    model_fused = copy.deepcopy(model)
    fused_backbone = []
    idx = 0
    while idx < len(model_fused):
        if (
            idx < len(model_fused) - 1
            and isinstance(model_fused[idx], torch.nn.Conv2d)
            and isinstance(model_fused[idx + 1], torch.nn.BatchNorm2d)
        ):
            fused_backbone.append(fuse_conv_bn(model_fused[idx], model_fused[idx + 1]))
            idx += 2
        else:
            fused_backbone.append(model_fused[idx])
            idx += 1
    return torch.nn.Sequential(*fused_backbone)


class QuantizedConv2d(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
        input_zero_point,
        output_zero_point,
        input_scale,
        weight_scale,
        output_scale,
        stride,
        padding,
        dilation,
        groups,
        feature_bitwidth=8,
        weight_bitwidth=8,
    ):
        super().__init__()
        # current version Pytorch does not support IntTensor as nn.Parameter
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer("weight_scale", weight_scale)
        self.output_scale = output_scale

        self.stride = stride
        self.padding = (padding[1], padding[1], padding[0], padding[0])
        self.dilation = dilation
        self.groups = groups

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth

    def forward(self, x):
        return quantized_conv2d_infer(
            x,
            self.weight,
            self.bias,
            self.feature_bitwidth,
            self.input_zero_point,
            self.output_zero_point,
            self.input_scale,
            self.weight_scale,
            self.output_scale,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuantizedLinner(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
        input_zero_point,
        output_zero_point,
        input_scale,
        weight_scale,
        output_scale,
        feature_bitwidth=8,
        weight_bitwidth=8,
    ):
        super().__init__()
        # current version Pytorch does not support IntTensor as nn.Parameter
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer("weight_scale", weight_scale)
        self.output_scale = output_scale

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth

    def forward(self, x):
        return quantized_liner_infer(
            x,
            self.weight,
            self.bias,
            self.feature_bitwidth,
            self.output_zero_point,
            self.input_scale,
            self.weight_scale,
            self.output_scale,
        )


class QuantizedMaxpool2d(torch.nn.MaxPool2d):
    def forward(self, input):
        return super().forward(input.float()).to(torch.int8)


class QuantizedAvgPool2d(torch.nn.AvgPool2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input.float()).to(torch.int8)
