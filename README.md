A collection of useful notes/implementations for deep learning, including:

# DeepLearning Framework implementation

- automatic differentiation implementation
- cpu/gpu based numpy-like library implementation
- DeepLearning framework abstraction

# Inference Optimization

## Model Pruning

- [channel pruning](https://github.com/fangpin/dlutil/blob/main/dlutil/pruning/channel_pruning.py)
- [fine grained pruning](https://github.com/fangpin/dlutil/blob/main/dlutil/pruning/fine_grained_pruning.py)

## Model Quantization

- [K means quantization](https://github.com/fangpin/dlutil/blob/main/dlutil/quantization/k_means_quantization.py)
- [liner quantization](https://github.com/fangpin/dlutil/blob/main/dlutil/quantization/liner_quantization.py)

# Classical Models

## CNN

- [Inception Net](https://github.com/fangpin/dlutil/blob/main/dlutil/models/inception.py)
- [ResNet](https://github.com/fangpin/dlutil/blob/main/dlutil/models/res_net.py)
- [DenseNet](https://github.com/fangpin/dlutil/blob/main/dlutil/models/dense_net.py)

## Transformers

- [transformer](https://github.com/fangpin/dlutil/blob/main/dlutil/models/transformer.py)
- [vision transformer (ViT) in both pytorch and jax&flax version](https://github.com/fangpin/dlutil/blob/main/dlutil/models/vision_transfomer.py)

# Distributed Parallel Training

Single GPU training tec: 1) mixed precision, 2) gradient checkpoint, 3) gradient accumulation.
