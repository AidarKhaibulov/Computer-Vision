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

image = Image.open('5.jpg')
image_array = np.array(image)

# Создаем сетку для отображения изображений
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
fig.subplots_adjust(hspace=0.5)

# Визуализация исходного изображения
axs[0, 0].imshow(image_array)
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

# Инвертируем изображение
inverted_image_array = 255 - image_array
axs[0, 1].imshow(inverted_image_array)
axs[0, 1].set_title('Inverted Image')
axs[0, 1].axis('off')

# Преобразуем изображение в полутоновое, используя усреднение по каналам
grayscale_image = np.mean(image_array, axis=2).astype(np.uint8)
axs[0, 2].imshow(grayscale_image, cmap='gray')
axs[0, 2].set_title('Grayscale Image')
axs[0, 2].axis('off')

# Добавляем случайный шум (нормальное распределение)
noise = np.random.normal(loc=0, scale=50, size=image_array.shape).astype(np.uint8)
noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
axs[1, 0].imshow(noisy_image)
axs[1, 0].set_title('Noisy Image')
axs[1, 0].axis('off')

# Строим гистограмму полученного изображения
histogram = np.histogram(noisy_image.ravel(), bins=256, range=(0, 256))[0]
axs[1, 1].bar(np.arange(256), histogram, width=1.0, color='b')
axs[1, 1].set_title('Histogram of Noisy Image')

# Задаем параметры для фильтра Гаусса
kernel_size = 5
sigma = 2

# Создаем и применяем ядро Гаусса
gaussian_kernel_matrix = gaussian_kernel(kernel_size, sigma)
blurred_image = apply_gaussian_blur(noisy_image, gaussian_kernel_matrix)
axs[1, 2].imshow(blurred_image)
axs[1, 2].set_title('Blurred Image (σ={})'.format(sigma))
axs[1, 2].axis('off')

# Производим операцию нерезкого маскирования
sharp_mask = noisy_image - blurred_image
axs[2, 0].imshow(sharp_mask)
axs[2, 0].set_title('Sharpened Image')
axs[2, 0].axis('off')

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
axs[2, 1].imshow(equalized_image, cmap='gray')
axs[2, 1].set_title('Equalized Image')
axs[2, 1].axis('off')


# Отображаем все изображения
plt.show()
