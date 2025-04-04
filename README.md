# RealESRGANPlus

**RealESRGANPlus** is an unofficial PyTorch-based implementation of Real-ESRGAN: Practical Algorithms for General Image Restoration. It is designed for upscaling images and videos with optional face enhancement. This package provides an easy-to-use interface for restoring low-resolution content with high fidelity.

---

## 📦 Package Information

- **Name:** RealESRGANPlus
- **Version:** 1.0.0
- **Author:** M. Hassan Ibrar
- **Email:** hassanibrar632@gmail.com
- **Repository:** [https://github.com/Hassanibrar632/RealESRGANPlus](https://github.com/Hassanibrar632/RealESRGANPlus)
- **License:** MIT
- **Python Version:** >= 3.6

---

## 🚀 Installation

Install the package directly from GitHub using pip:

```bash
pip install git+https://github.com/Hassanibrar632/RealESRGANPlus
```

---

## ⚙️ Post-Installation Fixes

### 1. Fix RetinaFace ResNet50 Loading (facexlib)

Navigate to:
```
C:\Users\YourUsername\miniconda3\envs\upscale\Lib\site-packages\facexlib\detection\retinaface.py
```

Replace lines 94 and 95:
```python
from torchvision.models import resnet50
backbone = resnet50(pretrained=True)
```

With:
```python
from torchvision.models import resnet50, ResNet50_Weights
backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
```

📝 **Note:** This resolves issues due to changes in how torchvision loads pretrained weights.

### 2. Fix RGB to Grayscale Import (basicsr)

Navigate to:
```
C:\Users\YourUsername\miniconda3\envs\upscale\Lib\site-packages\basicsr\data\degradations.py
```

Replace:
```python
from torchvision.transforms.functional_tensor import rgb_to_grayscale
```

With:
```python
from torchvision.transforms.functional import rgb_to_grayscale
```

✅ After these changes, your installation is fully functional.

---

## 📚 Usage Example

### Image Upscaling

```python
from RealESRGANPlus import RealESRGANPlus
import cv2

# Load the model
model = RealESRGANPlus(
    model_name='RealESRGAN_x4plus',
    model_path=None,
    gpu_id=0,
    denoise_strength=0.0,
    outscale=4,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    face_enhance=True,
    fp32=True,
    alpha_upsampler='realesrgan'
)

# Load a low-resolution image
img = cv2.imread('input/lr_image.png', cv2.IMREAD_UNCHANGED)

# Upscale the image
sr_image = model.upscale_image(img)

# Save the output
cv2.imwrite('output/sr_image.png', sr_image)
```

### Video Upscaling

```python
out_vid = model.upscale_video('input/lr_video.mp4', 'output', ffmpeg_bin='ffmpeg')
print(f'Upscaled video saved to {out_vid}')
```

---

## 🔧 Model Parameters

```python
RealESRGANPlus(
    model_name='RealESRGAN_x4plus',
    model_path=None,
    gpu_id=0,
    denoise_strength=0.0,
    outscale=4,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    face_enhance=True,
    fp32=True,
    alpha_upsampler='realesrgan'
)
```

### Parameter Descriptions

| Parameter          | Type    | Description |
|-------------------|---------|-------------|
| `model_name`       | str     | ESRGAN model to use: `RealESRGAN_x4plus`, `RealESRNet_x4plus`, `RealESRGAN_x4plus_anime_6B`, `RealESRGAN_x2plus`, `realesr-animevideov3` |
| `model_path`       | str     | Optional path to custom model weights |
| `gpu_id`           | int     | GPU device ID (default: 0) |
| `denoise_strength` | float   | Denoising strength (range: 0.0 to 1.0) |
| `outscale`         | float   | Image upscaling factor (e.g., 4 for 4x) |
| `tile`             | int     | Tiling size (0 = no tiling) |
| `tile_pad`         | int     | Padding between tiles |
| `pre_pad`          | int     | Padding before processing |
| `face_enhance`     | bool    | Use GFPGAN (ResNet50) for face enhancement |
| `fp32`             | bool    | Use full precision (FP32) instead of FP16 |
| `alpha_upsampler`  | str     | Upsampler for alpha channel: `realesrgan` or `bicubic` |

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgements

- [BasicSR](https://github.com/XPixelGroup/BasicSR)
- [facexlib](https://github.com/xinntao/facexlib)

