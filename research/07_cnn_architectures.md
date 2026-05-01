# Research: CNN & Architecture Opaque Wrapper Atoms

## Goal

Define opaque wrapper specifications for pretrained neural network
architectures used across CDGs. These should NOT be full implementations —
they should be `is_opaque=true` atoms that document the interface contract
for architecture stages. Target repo: `sciona-atoms-dl`.

## CDG stages this research covers (~29 stages)

- EfficientNet backbone (Alaska2, Aptos, Cassava, Melanoma, many more)
- ResNet/ResNeXt/SE-ResNeXt (Bengali Grapheme, RSNA series)
- Swin Transformer (BMS Molecular, RSNA Cervical)
- DenseNet (Melanoma, RSNA Pneumonia)
- U-Net / 1D U-Net (Child Mind Sleep, TGS Salt, HubMap)
- YOLO / object detector head (Great Barrier Reef, Sartorius)
- Transformer encoder/decoder (BMS Molecular, Stanford Ribonanza)
- GRU/LSTM sequence models (Web Traffic, OpenVaccine)
- Whisper / ASR model (Bengali Speech)
- MIL aggregator — Multiple Instance Learning (PANDA Prostate)
- Autoregressive decoder (BMS Molecular Translation)
- SlowFast video network (DFL Bundesliga)

## What to research

The goal is NOT to implement these architectures but to define their
interface contracts so the CDG system can treat them as opaque atoms.

### For each architecture, document:

1. **Standard input format**: tensor shape, dtype, value range
   - e.g., EfficientNet: `(B, 3, H, W)` float32, ImageNet-normalized
   - e.g., U-Net: `(B, C, H, W)` → `(B, num_classes, H, W)`

2. **Standard output format**: what the model produces
   - Classification: `(B, num_classes)` logits or probabilities
   - Detection: `(B, num_boxes, 5+num_classes)` boxes + scores
   - Segmentation: `(B, num_classes, H, W)` pixel-level predictions
   - Feature extraction: `(B, D)` embedding vector

3. **Configurable parameters**: what varies across CDGs
   - Backbone variant (efficientnet-b0 through b7)
   - Pretrained weights (ImageNet, noisy-student, etc.)
   - Number of output classes
   - Input resolution

4. **Common variants used in competitions**
   - EfficientNet: b0-b7, v2 variants, noisy-student pretraining
   - ResNet: 18/34/50/101/152, ResNeXt-50/101, SE variants
   - U-Net: vanilla, Res-UNet, attention-UNet, nested-UNet (UNet++)

### Atom specification format:

```python
@register_atom
def efficientnet_backbone(
    images: NDArray,  # (B, 3, H, W) float32, ImageNet-normalized
    variant: str = "b4",
) -> NDArray:  # (B, D) feature embeddings
    """EfficientNet feature extraction backbone.

    Opaque pretrained model — the internal architecture is not decomposed.
    This atom represents the contract: normalized images in, feature
    embeddings out.
    """
    raise NotImplementedError("Opaque atom — bind to a pretrained model at runtime")
```

## Research questions

1. For each architecture family, what are the standard input/output shapes?
2. What are the most commonly used variants in Kaggle competitions?
3. What is the standard pretrained weight source? (timm, torchvision, HuggingFace)
4. What contracts can we enforce even on opaque models?
   (input shape validation, output shape validation, value range)
5. Should we have one atom per architecture family (e.g., `efficientnet_backbone`)
   or one per variant (e.g., `efficientnet_b4_backbone`)?
   Recommend: one per family with variant parameter.

## Output format

For each architecture family:
```
Name: efficientnet_backbone
Description: EfficientNet CNN feature extraction from normalized images.
  Opaque pretrained model — internal architecture not decomposed.
Concept type: neural_network
is_opaque: true
Source: URL to the canonical model implementation or pretrained-weight provider
License: MIT, BSD, Apache-2.0, or public domain; flag any incompatible license
Input contract: images (B, 3, H, W) float32, H and W divisible by 32
Output contract: features (B, D) float32, D depends on variant
Variants: b0 (1280), b1 (1280), b2 (1408), b3 (1536), b4 (1792), b5 (2048), b6 (2304), b7 (2560)
Pure function boundary: normalized input tensor and explicit configuration in,
  output tensor contract documented; no training loop, checkpoint download, file
  I/O, or hidden preprocessing inside the atom.
Witness: shape-only validation fixture with a tiny batch and documented expected
  output shape for one variant; implementation may raise NotImplementedError
  because the atom is opaque.
Dependencies: timm, torchvision, HuggingFace, or torch as required by the runtime
  binding; document optional versus required dependencies.
CDG stages covered: alaska2/model_training, aptos/training, cassava/training, ...
```
