import cv2
import numpy as np

# Função para calcular a entropia
def calcular_entropia(imagem):
    # Obtém os valores únicos e suas contagens
    valores, contagens = np.unique(imagem, return_counts=True)
    # Calcula as probabilidades
    p = contagens / contagens.sum()
    # Calcula a entropia com log base 2
    return -np.sum(p * np.log2(p))

# Carregar as imagens em escala de cinza
img1 = cv2.imread("q1a.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("q1b.png", cv2.IMREAD_GRAYSCALE)

# Calcular entropias
entropia1 = calcular_entropia(img1)
entropia2 = calcular_entropia(img2)

# Mostrar os resultados
print(f"Entropia da imagem q1a.png: {entropia1}")
print(f"Entropia da imagem q1b.png: {entropia2}")
