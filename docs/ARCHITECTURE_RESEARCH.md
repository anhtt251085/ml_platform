# Architecture research for the library

Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.

The architecture library (`backend/archlib.py`, Architecture library page) holds model architectures that the
platform can train, quantize to INT8 and hand to the SAI compiler. This note records which published TinyML
designs were surveyed, how each was adapted to the platform's operator set, and which ones cannot be expressed
yet.

## The operator set the library can use

`conv`, `dwconv` (depthwise), `pwconv` (1x1), `maxpool`, `avgpool`, `gap` (global average), `fc` (dense; flattens
its input), `residual` (two equal-width convolutions plus an identity add), `dropout` (training only), each with
ReLU where it applies. Kernels are square (k x k on 2-D inputs, k on 1-D), at most 31. The output head is added
automatically.

## Method

1. Survey: MLPerf Tiny reference models, keyword-spotting and audio models, mobile CNNs, sensor and HAR models,
   classic CNNs, anomaly autoencoders.
2. Adapt each to the operator set and record the adaptation (`ref` field of every entry).
3. Check with SAI: each entry is exported with random weights at a reference input (time series 3x1x256,
   2-D 1x49x10, features 64) and SAI's plan search runs on MD6605 and Cortex-M7. RAM, flash, latency and fit are
   stored per board and shown on the Architecture library page ("Verify with SAI" refreshes them).

## Results

32 architectures; SAI compiles all of them. 31 fit both boards at the reference input; one (large-kernel
separable TCN) needs a Cortex-M7 class part. Fit depends on the customer's input, so every project costs its
choices again on the real data.


### Time series [C, 1, T]

| Architecture | Family | Params | MD6605 | Cortex-M7 | Based on / adaptation |
|---|---|---:|:-:|:-:|---|
| DS-TCN XS | Depthwise-separable TCN | 320 | ✓ | ✓ | Platform design |
| DS-TCN S | Depthwise-separable TCN | 592 | ✓ | ✓ | Platform design |
| DS-TCN M | Depthwise-separable TCN | 1,972 | ✓ | ✓ | Platform design |
| DS-TCN L | Depthwise-separable TCN | 8,668 | ✓ | ✓ | Platform design |
| Dilated TCN S | Dilated TCN | 1,488 | ✓ | ✓ | Platform design |
| Dilated TCN M | Dilated TCN | 3,972 | ✓ | ✓ | Platform design |
| Wide-kernel CNN | Wide-kernel CNN | 8,252 | ✓ | ✓ | Platform design |
| Residual TCN M | Residual TCN | 12,876 | ✓ | ✓ | Platform design |
| CNN-1D S | Standard CNN-1D | 1,804 | ✓ | ✓ | Platform design |
| Inverted-bottleneck TCN S | Inverted-bottleneck TCN | 9,084 | ✓ | ✓ | TENet (Li et al., Interspeech 2020) inverted bottleneck blocks; block skip connections removed (the platform's residual only wraps two equal-width convolutions). |
| Large-kernel separable TCN | Time-channel separable TCN | 8,740 | ✗ (RAM) | ✓ | MatchboxNet / QuartzNet (NVIDIA, 2020) time-channel separable blocks; single repeats, no residuals. |
| Strided DS-TCN (no pooling) | Depthwise-separable TCN | 1,948 | ✓ | ✓ | MobileNet-style strided depthwise downsampling applied to 1-D. |
| Raw-signal CNN (M5-style) | Raw-waveform CNN | 24,932 | ✓ | ✓ | M5 (Dai et al., ICASSP 2017); first kernel 80 -> 31 (the platform's maximum). |
| Residual TCN S | Residual TCN | 2,924 | ✓ | ✓ | Platform design |
| TCN + dense head | Depthwise-separable TCN | 13,372 | ✓ | ✓ | Classic CNN-1D + dense classifier head used in many HAR and fault-diagnosis papers. |

### Spectrograms and images [C, H, W]

| Architecture | Family | Params | MD6605 | Cortex-M7 | Based on / adaptation |
|---|---|---:|:-:|:-:|---|
| DS-CNN 2D S | Depthwise-separable CNN-2D | 1,260 | ✓ | ✓ | Platform design |
| DS-CNN 2D M | Depthwise-separable CNN-2D | 5,596 | ✓ | ✓ | Platform design |
| CNN-2D S | Standard CNN-2D | 4,828 | ✓ | ✓ | Platform design |
| Mobile-style 2D M | Inverted bottleneck CNN-2D | 10,268 | ✓ | ✓ | Platform design |
| DS-CNN 2D L (64 ch) | Depthwise-separable CNN-2D | 20,100 | ✓ | ✓ | DS-CNN (Zhang et al., 'Hello Edge', 2017; MLPerf Tiny KWS); stem 10x4 kernel -> square 3x3. |
| MobileNetV1-style 0.25 | MobileNet | 9,876 | ✓ | ✓ | MobileNetV1 (Howard et al., 2017) at width 0.25 as in MLPerf Tiny visual wake words; truncated depth. |
| ResNet-style 2D S | Residual CNN-2D | 30,324 | ✓ | ✓ | ResNet-8 (MLPerf Tiny image classification) at half width (8-16-32); projection shortcuts replaced by a strided convolution before each identity residual block. |
| TinyConv (micro-speech style) | Single-layer CNN | 16,740 | ✓ | ✓ | TensorFlow Lite Micro 'micro_speech' tiny_conv (8 filters 10x8, stride 2); kernel made square. |
| LeNet-5 style | Classic CNN-2D | 29,424 | ✓ | ✓ | LeNet-5 (LeCun et al., 1998) with SAME padding and smaller dense layers. |
| VGG-tiny 2D | Classic CNN-2D | 4,220 | ✓ | ✓ | VGG (Simonyan & Zisserman, 2014) pattern at MCU scale. |

### Feature vectors

| Architecture | Family | Params | MD6605 | Cortex-M7 | Based on / adaptation |
|---|---|---:|:-:|:-:|---|
| MLP S | Dense network | 2,676 | ✓ | ✓ | Platform design |
| MLP M | Dense network | 6,836 | ✓ | ✓ | Platform design |
| Autoencoder S | Dense autoencoder | 3,576 | ✓ | ✓ | Platform design |
| Autoencoder M | Dense autoencoder | 8,136 | ✓ | ✓ | Platform design |
| Linear baseline | Linear model | 260 | ✓ | ✓ | Platform design |
| MLP L | Dense network | 18,788 | ✓ | ✓ | Platform design |
| Autoencoder L (MLPerf-style) | Dense autoencoder | 51,784 | ✓ | ✓ | MLPerf Tiny anomaly-detection FC autoencoder (ToyADMOS), 4 hidden layers per side reduced to 2. |

Params at the reference input with 4 outputs (64 for autoencoders). ✓ = SAI finds a plan within the board's AI
arena, stack reserve and AI flash budget.

## Surveyed but not expressible yet

| Design | Needs | Closest library entry |
|---|---|---|
| MobileNetV2 / MCUNet inverted residuals | identity add around an expand-depthwise-project block | Inverted-bottleneck TCN S, Mobile-style 2D M |
| ResNet downsampling blocks | projection (1x1, strided) shortcut on the add | ResNet-style 2D S |
| Inception / InceptionTime, DenseNet | concatenation of branches | Dilated TCN, DS-TCN L |
| Squeeze-and-excitation (EfficientNet, MobileNetV3) | channel-wise multiply by a pooled gate; hard-swish | Mobile-style 2D M |
| BC-ResNet (broadcasted residual) | broadcast add between 1-D and 2-D branches | DS-CNN 2D M |
| DeepConvLSTM, GRU/LSTM keyword spotters | recurrent layers | Dilated TCN M, Large-kernel separable TCN |
| Attention / transformer (KWT, TinyTransformer) | matmul between activations, softmax, layer norm | DS-CNN 2D L |
| Convolutional autoencoders | transposed convolution / upsampling | Dense autoencoders S/M/L |
| Non-square kernels (DS-CNN 10x4 stem, TinyConv 10x8) | k_h != k_w | DS-CNN 2D, TinyConv (square kernels) |

Each row is an operator the platform's model builder would need first (and, for the deployed program, an INT8
kernel); they are listed in the order that would unlock the most designs for MCU-class targets.
