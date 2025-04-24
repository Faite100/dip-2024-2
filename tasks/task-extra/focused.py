import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega a imagem original
img = cv2.imread("moon.tif", cv2.IMREAD_GRAYSCALE)

# Aplica um desfoque gaussiano
gauss = cv2.GaussianBlur(img, (9, 9), 2)

# Máscara de nitidez: imagem original + (imagem original - desfoque) * fator
sharp = cv2.addWeighted(img, 1.5, gauss, -0.5, 0)

# Exibe o resultado
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Imagem com Foco Aprimorado")
plt.imshow(sharp, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
