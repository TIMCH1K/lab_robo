import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "Sunflower_from_Silesia2.jpg"

image = cv2.imread(image_path)

# Преобразование в градации серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Размытие по Гауссу
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# 3. Повышение резкости (два метода)

# Метод 1: Свертка с ядром
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
sharpened_image_1 = cv2.filter2D(image, -1, kernel)

# Метод 2: Маска нерезкости
blurred_for_unsharp = cv2.GaussianBlur(image, (5, 5), 0)
sharpened_image_2 = cv2.addWeighted(image, 1.5, blurred_for_unsharp, -0.5, 0)

# 4. Выделение границ с помощью оператора Собеля
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
sobel_edges = cv2.magnitude(sobel_x, sobel_y)
sobel_edges = cv2.convertScaleAbs(sobel_edges)

# 5. Комбинирование результатов
blurred_resized = cv2.resize(blurred_image, (sobel_edges.shape[1], sobel_edges.shape[0]))
sharpened_resized = cv2.resize(sharpened_image_2, (sobel_edges.shape[1], sobel_edges.shape[0]))
sobel_edges_colored = cv2.cvtColor(sobel_edges, cv2.COLOR_GRAY2BGR)

combined = cv2.addWeighted(blurred_resized, 0.5, sobel_edges_colored, 0.5, 0)
combined = cv2.addWeighted(combined, 0.5, sharpened_resized, 0.5, 0)

def show_images(original, blurred, edges, sharpened, combined):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.title('Оригинальное изображение')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Размытие по Гауссу')
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Выделение границ (Собель)')
    plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('Повышение резкости')
    plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Комбинация изображений')
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

show_images(image, blurred_image, sobel_edges_colored, sharpened_image_2, combined)
