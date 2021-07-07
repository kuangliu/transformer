# CIFAR-Transformer
Simple transformer model for CIFAR10.

Reference:
  - https://www.tensorflow.org/text/tutorials/transformer
  - https://github.com/huggingface/transformers/blob/master/src/transformers/models/detr
  - [Early Convolutions Helps Transformers See Better](https://arxiv.org/pdf/2106.14881.pdf)

## Accuracy
Model | Acc.
------|------
ResViT|93.17%

The original ViT model performs really bad on the CIFAR10 dataset, which is disappointing. To achieve a better accuracy I use ResNet as the stem network.
