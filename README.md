# CIFAR-Transformer
Simple transformer model for CIFAR10.

Reference:
  - https://www.tensorflow.org/text/tutorials/transformer
  - https://github.com/huggingface/transformers/blob/master/src/transformers/models/detr
  - [Early Convolutions Helps Transformers See Better](https://arxiv.org/pdf/2106.14881.pdf)

## Accuracy
The original ViT model performs really bad on CIFAR10.  
To achieve better accuracy, I add extra conv layers before transformer encoder.

Model | Acc.
------|------
VGGStem ViT| 91.92%
ResNetStem ViT| 92.44%
SEViT | 92.30%
SEViT (with nn.TransformerEncoder) | 92.74%

