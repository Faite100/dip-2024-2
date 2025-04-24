import cv2
import matplotlib.pyplot as plt

# Carrega a imagem em escala de cinza
imagem = cv2.imread("aerial.tif", cv2.IMREAD_GRAYSCALE)

# Aplica equalização de histograma
imagem_equalizada = cv2.equalizeHist(imagem)

# Exibe a imagem original e a equalizada
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Imagem Original")
plt.imshow(imagem, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Equalização de Histograma")
plt.imshow(imagem_equalizada, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()