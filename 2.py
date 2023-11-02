import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import label

image1 = Image.open('1.png')
image2 = Image.open('2.jpg')
image3 = Image.open('3.jpeg')

image1_array = np.array(image1)
image2_array = np.array(image2)
image3_array = np.array(image3)

def bin_image_otsu(image_array):

    # переводим в оттенки серого, каждый пиксель будет содержать одно значение яркости
    grayscale_image = np.mean(image_array, axis=2).astype(np.uint8)
    hist, column_edges = np.histogram(grayscale_image, bins=256, range=(0, 256))

    i = 0
    j = 1
    res = [] # хранит пороговые значения после анализа гистрограммы
    res = np.array(res)
    # проходимся по интервалам (между парами столбцов)
    while i != len(column_edges)-1:
        tmp = (column_edges[i] + column_edges[j]) / 2
        res = np.append(res, tmp)
        i += 1
        j += 1
    hist = hist / hist.sum()
    variances = []
    # теперь в res лежат потенциальные пороговые значения

    # нормируем гистограмму, чтобы сумма всех значений =1
    for t in res:
        q1 = hist[:int(t)].sum()
        q2 = hist[int(t):].sum()
        if q1 == 0 or q2 == 0:
            continue
        m1 = (hist[:int(t)] * res[:int(t)]).sum() / q1
        m2 = (hist[int(t):] * res[int(t):]).sum() / q2
        v1 = ((res[:int(t)] - m1) ** 2 * hist[:int(t)]).sum() / q1
        v2 = ((res[int(t):] - m2) ** 2 * hist[int(t):]).sum() / q2
        # вычисляем дисперсию внутриклассовых интенсивностей для каждого порогового значения, сохраняем в variances
        variances.append(q1 * v1 + q2 * v2)
    min = variances[0]
    min_index = 0

    for i in range(len(variances)):
        if variances[i] < min:
            min = variances[i]
            min_index = i
    # нашли пороговое значение, соответствующее минимальной дисперсии
    ot = res[min_index]

    # бинаризируем на основе найденного порогового значения ot
    binary_image = (grayscale_image > ot).astype(np.uint8) * 255
    return binary_image

def remove_noise(binary_image, kernel_size=1):# приняли изобрражение и размер окрестности (по умолчанию 1)
    height, width = binary_image.shape
    cleaned_image = np.copy(binary_image)

    # обходим каждый пиксель кроме граней шириной и высотой равных kernel_size
    for x in range(kernel_size, height - kernel_size):
        for y in range(kernel_size, width - kernel_size):
            neighborhood = binary_image[x - kernel_size: x + kernel_size + 1, y - kernel_size: y + kernel_size + 1]

            white_pixels = 0
            black_pixels = 0
            # для каджой окрестности подсчитываем черные и белые пиксели
            for i in range(neighborhood.shape[0]):
                for j in range(neighborhood.shape[1]):
                    if neighborhood[i, j] == 255:
                        white_pixels += 1
                    elif neighborhood[i, j] == 0:
                        black_pixels += 1
            # для каждого центрального пикселя в окрестности устанавливается цвет в зависимости от соотношения черных и белых в окретсности
            if white_pixels > black_pixels:
                cleaned_image[x, y] = 255
            else:
                cleaned_image[x, y] = 0

    return cleaned_image

def grow(seed, label, binary_image, visited, result_image):
    height, width = binary_image.shape
    arr = [seed] # координаты стартового пикселя
    while arr:
        # проходимся по пикселя и проверяем являяется ли он белым и лежит ли в пределах границ изображения
        x, y = arr.pop()
        if 0 <= x < height and 0 <= y < width and binary_image[x, y] == 255 and visited[x, y] == 0:
            # если да, то помечаем, что пиксель принадлежит текущему сегменту
            result_image[x, y] = label
            visited[x, y] = 1
            # закидываем соседние пиксели в стек
            arr.append((x + 1, y))
            arr.append((x - 1, y))
            arr.append((x, y + 1))
            arr.append((x, y - 1))

def seed_grow(binary_image):
    # приняли изображение с только черными и белыми пикслеями
    height, width = binary_image.shape
    visited = np.zeros_like(binary_image)
    result_image = np.zeros_like(binary_image)

    label = 1
    for x in range(height):
        for y in range(width):
            if binary_image[x, y] == 255 and visited[x, y] == 0:
                # попытка расширения сегментов с перебираемых точек
                grow((x, y), label, binary_image, visited, result_image)
                label += 1
    # разбили на несколько сегментов с уникальными метками
    return result_image

def segmentate(seed, label, visited, result_image, image_array, t):
    grayscale_image = np.mean(image_array, axis=2).astype(np.uint8)
    height, width = grayscale_image.shape

    arr = [seed]

    while arr:
        x, y = arr.pop()
        # проверяем, находится ли пиксель в пределах границ оттенкового изображения, и не был ли он уже посещен
        if 0 <= x < height and 0 <= y < width and visited[x, y] == 0:
            result_image[x, y] = label
            visited[x, y] = 1
            # проверяем соседей
            if 0 <= x + 1 < height and t[x, y] == t[x+1,y]:
                arr.append((x + 1, y))
            if 0 <= x - 1 < height and t[x, y] == t[x-1,y]:
                arr.append((x - 1, y))
            if 0 <= y + 1 < width and t[x,y] == t[x, y+1]:
                arr.append((x, y + 1))
            if 0 <= y - 1 < width and t[x,y] == t[x, y-1]:
                arr.append((x, y - 1))

                # в result_image будет лежать сегментированное изображение


def combined_segmentation(image_array):
    grayscale_image = np.mean(image_array, axis=2).astype(np.uint8)

    t = grayscale_image.copy()
    # создаем гистрограмму интенсивности распределения пикселей на оттенках серого
    hist, column_edges = np.histogram(grayscale_image, bins=256, range=(0, 256))
    res = [0]
    for i in range(len(hist)):
        is_local = True
        # проверка на локальный миниимум для каждого оттенка серого
        for ind in range(max(0, i - 10), min(len(hist), i + 10)):
            if hist[ind] < hist[i]:
                is_local = False
                break
        if is_local:
            res.append(i)
    # сохраняем локальные минимумы, в коцне добавляем 255 для обеспечения бинаризации оттенков, превышающего это значение
    res.append(255)

    for i in range(len(grayscale_image)):
        for z in range(len(grayscale_image[i])):
            for j in range(len(res)-1):
                # сравниваем интенсивность пикселя с пороговым значением и переприсваиваем новую интенсивность
                if grayscale_image[i][z] >= res[j] and grayscale_image[i][z] <= res[j+1]:
                    t[i][z] = j

    # в резльтате изображение разбито на сегменты
    height, width = grayscale_image.shape
    visited = np.zeros_like(grayscale_image)
    result_image = np.zeros_like(grayscale_image)
    label = 1
    # разбиваем изображение на области различных интенсивностей
    for x in range(height):
        for y in range(width):
            if visited[x, y] == 0:
                segmentate((x, y), label, visited, result_image, image_array, t)
                label += 1

    return result_image

comb_image1 = combined_segmentation(image1_array)
comb_image2 = combined_segmentation(image2_array)
comb_image3 = combined_segmentation(image3_array)

binary_image1 = bin_image_otsu(image1_array)
binary_image2 = bin_image_otsu(image2_array)
binary_image3 = bin_image_otsu(image3_array)
#
cleaned_image1 = remove_noise(binary_image1)
cleaned_image2 = remove_noise(binary_image2)
cleaned_image3 = remove_noise(binary_image3)
#
segmented_image1 = seed_grow(cleaned_image1)
segmented_image2 = seed_grow(cleaned_image2)
segmented_image3 = seed_grow(cleaned_image3)


plt.figure(figsize=(10, 10))
plt.subplot(331), plt.imshow(image1_array), plt.title('Изображение 1'), plt.axis('off')
plt.subplot(332), plt.imshow(cleaned_image1, cmap='gray'), plt.title('Бинаризированное без шума'), plt.axis('off')
plt.subplot(333), plt.imshow(segmented_image1, cmap='rainbow'), plt.title('Сегментированное'), plt.axis('off')
plt.subplot(334), plt.imshow(image2_array), plt.title('Изображение 2'), plt.axis('off')
plt.subplot(335), plt.imshow(cleaned_image2, cmap='gray'), plt.title('Бинаризированное без шума'), plt.axis('off')
plt.subplot(336), plt.imshow(segmented_image2, cmap='rainbow'), plt.title('Сегментированное'), plt.axis('off')
plt.subplot(337), plt.imshow(image3_array), plt.title('Изображение 3'), plt.axis('off')
plt.subplot(338), plt.imshow(cleaned_image3, cmap='gray'), plt.title('Бинаризированное без шума'), plt.axis('off')
plt.subplot(339), plt.imshow(segmented_image3, cmap='rainbow'), plt.title('Сегментированное'), plt.axis('off')
plt.show()

plt.figure(figsize=(10, 10))
plt.subplot(331), plt.imshow(comb_image1, cmap='rainbow'), plt.title('Сегментированное (лок. мин.) 1'), plt.axis('off')
plt.subplot(334), plt.imshow(comb_image2, cmap='rainbow'), plt.title('Сегментированное (лок. мин.) 2'), plt.axis('off')
plt.subplot(337), plt.imshow(comb_image3, cmap='rainbow'), plt.title('Сегментированное (лок. мин.) 3'), plt.axis('off')
plt.show()
