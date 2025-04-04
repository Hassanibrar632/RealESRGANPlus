# Usage example for RealESRGANPlus
from RealESRGANPlus import RealESRGANplus
import cv2

# Load the model
model = RealESRGANplus(model_name='RealESRGAN_x4plus',
            model_path=None,
            gpu_id=0,
            denoise_strength=0.0,
            outscale=4,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            face_enhance=True,
            fp32=True,
            alpha_upsampler='realesrgan')

# loading the image
img = cv2.imread('input\lr_image.png', cv2.IMREAD_UNCHANGED)

# Upscale the image
sr_image = model.upscale_image(img)

# Save the output image
cv2.imwrite(f'output/sr_image.png', sr_image)
