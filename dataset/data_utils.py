# dataset/data_utils.py

from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random

def augment_image(image):

    # 随机水平翻转
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    
    # 随机旋转，角度在 -10 到 10 度之间
    angle = random.uniform(-10, 10)
    image = image.rotate(angle)
    
    # 随机调整亮度
    brightness_factor = random.uniform(0.8, 1.2)  # 调整因子在0.8到1.2之间
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)
    
    # 随机调整对比度
    contrast_factor = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)
    
    # 随机调整饱和度（颜色）
    color_factor = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(color_factor)
    
    # 添加随机高斯噪声
    image_np = np.array(image).astype(np.int16)
    # 均值0，标准差5的高斯噪声
    noise = np.random.normal(0, 5, image_np.shape).astype(np.int16)
    image_np = image_np + noise
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    augmented_image = Image.fromarray(image_np)
    
    return augmented_image
