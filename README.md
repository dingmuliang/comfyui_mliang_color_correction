# MLiang Color Correction for ComfyUI

Two small image color nodes for ComfyUI.

## Nodes

- `MLiang Color Correction`
  - exposure
  - brightness
  - contrast
  - saturation
  - gamma
  - temperature
  - tint
  - hue shift
  - shadows
  - highlights
  - strength

- `MLiang White Balance`
  - red / green / blue channel multipliers
  - optional luminance normalization
  - strength

Both nodes accept and return ComfyUI `IMAGE` tensors in `[batch, height, width, channels]` format.

## Example Workflow

The clean showcase workflow is included here:

`examples/MLiang_Color_Correction_Showcase.json`

It uses only built-in ComfyUI image nodes plus `MLiang Color Correction`:

`Load Image -> MLiang Color Correction -> Preview Image / Save Image`
