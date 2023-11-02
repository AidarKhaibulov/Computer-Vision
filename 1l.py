import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(
            -((x - (size - 1) / 2)**2 + (y - (size - 1) / 2)**2) / (2 * sigma**2)
        ),
        (size, size)
    )
    return kernel / np.sum(kernel)

def apply_gaussian_blur(image, kernel):
    height, width, channels = image.shape
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2

    blurred_image = np.zeros_like(image, dtype=np.float32)

    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            for c in range(channels):
                blurred_image[i, j, c] = np.sum(
                    image[i - pad:i + pad + 1, j - pad:j + pad + 1, c] * kernel
                )

    return blurred_image.astype(np.uint8)

image = Image.open('1.png')
image_array = np.array(image)

# Визуализация исходного изображения
plt.figure(figsize=(6, 6))
plt.imshow(image_array)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Инвертируем изображение
inverted_image_array = 255 - image_array

# Визуализация инвертированного изображения
plt.figure(figsize=(6, 6))
plt.imshow(inverted_image_array)
plt.title('Inverted Image')
plt.axis('off')
plt.show()

# Преобразуем изображение в полутоновое, используя усреднение по каналам
grayscale_image = np.mean(image_array, axis=2).astype(np.uint8)

# Визуализация полутонового изображения
plt.figure(figsize=(6, 6))
plt.imshow(grayscale_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

# Добавляем случайный шум (нормальное распределение)
noise = np.random.normal(loc=0, scale=50, size=image_array.shape).astype(np.uint8)
noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)

# Визуализация изображения с добавленным шумом
plt.figure(figsize=(6, 6))
plt.imshow(noisy_image)
plt.title('Noisy Image')
plt.axis('off')
plt.show()

# Строим гистограмму полученного изображения
histogram = np.histogram(noisy_image.ravel(), bins=256, range=(0, 256))[0]

# Визуализация гистограммы
plt.figure(figsize=(10, 5))
plt.bar(np.arange(256), histogram, width=1.0, color='b')
plt.title('Histogram of the Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

# Задаем параметры для фильтра Гаусса
kernel_size = 5
sigma = 256

# Создаем и применяем ядро Гаусса
gaussian_kernel_matrix = gaussian_kernel(kernel_size, sigma)
blurred_image = apply_gaussian_blur(noisy_image, gaussian_kernel_matrix)

# Визуализация размытого изображения
plt.figure(figsize=(6, 6))
plt.imshow(blurred_image)
plt.title('Blurred Image (σ={})'.format(sigma))
plt.axis('off')
plt.show()

# Производим операцию нерезкого маскирования
sharp_mask = noisy_image - blurred_image

# Визуализация операции нерезкого маскирования
plt.figure(figsize=(6, 6))
plt.imshow(sharp_mask)
plt.title('Sharpened Image')
plt.axis('off')
plt.show()

# Реализуем эквализацию гистограммы изображения
def histogram_equalization(image):
    height, width = image.shape
    num_pixels = height * width

    histogram = np.histogram(image.ravel(), bins=256, range=(0, 256))[0]
    normalized_histogram = histogram / num_pixels
    equalization_function = np.cumsum(normalized_histogram) * 255

    equalized_image = equalization_function[image]

    return equalized_image.astype(np.uint8)

# Применяем эквализацию гистограммы к полутоновому изображению
equalized_image = histogram_equalization(grayscale_image)

# Визуализация эквализированного изображения
plt.figure(figsize=(6, 6))
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')
plt.show()
