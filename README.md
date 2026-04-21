# CFGFE: Coarse-Fine Grained Feature Enhancement Network for Pansharpening

This folder provides the **core model implementation** of **CFGFE** for pansharpening, extracted from the integrated framework and prepared for later GitHub release.

## Paper

- **Title**: CFGFE: Coarse-Fine Grained Feature Enhancement Network for Pansharpening  
- **Journal**: *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS)*

## What’s included

- **Model code only** (no datasets / metrics / training / testing scripts):
  - `cfgfe/models/CFGFE.py`
  - `cfgfe/models/WDAM.py`
  - `cfgfe/models/refine.py`

## Quick import

```python
from cfgfe.models import CFGFE

model = CFGFE(num_channels=16)
```

Forward signature:

```python
pred = model(l_ms, bms, pan)
```

- `l_ms`: low-resolution MS (used by the original framework; kept for API compatibility)
- `bms`: upsampled MS at PAN resolution (typically bicubic-upsampled LRMS)
- `pan`: panchromatic image

## Notes

- This is intended for **code sharing and reproduction** of the CFGFE architecture.  
- Training and evaluation pipelines are intentionally **not** included per current requirement.

