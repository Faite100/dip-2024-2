import numpy as np
import matplotlib.pyplot as plt

# Tamanho da imagem
h, w = 512, 512

# Coordenadas normalizadas centradas (0 no centro da imagem)
y, x = np.indices((h, w))
x = x - w // 2
y = y - h // 2

# Calcula o ângulo polar (em radianos)
theta = np.arctan2(y, x)

# Número de ciclos (10 ciclos completos ao redor do centro)
frequencia = 10

# Gera padrão radial com senoide
imagem = np.sin(frequencia * theta)

# Normaliza para intervalo [0, 255]
imagem_norm = ((imagem + 1) / 2 * 255).astype(np.uint8)

# Exibe a imagem
plt.imshow(imagem_norm, cmap='gray')
plt.axis('off')
plt.title("Padrão Radial Senoidal")
plt.show()
