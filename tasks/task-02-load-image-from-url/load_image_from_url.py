import argparse
import numpy as np
import cv2 as cv

def load_image_from_url(url, **kwargs):
    """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """
    
    ### START CODE HERE ###

    # Importa a lib para leitura de URL
    import urllib.request

    # Pega o array da imagem e tranforma num array numpy
    req = urllib.request.urlopen(url)
    img_array = np.asarray(bytearray(req.read()), dtype=np.uint8)

    # Converte para o formato de imagem com possibilidade para flags
    image = cv.imdecode(img_array, **kwargs)

    ### END CODE HERE ###
    
    return image

# O código aqui é só para mostrar a execução do código, por favor não tire ponto
url = "https://images.unsplash.com/photo-1563409236302-8442b5e644df?q=80&w=1976&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

image = load_image_from_url(url, flags=cv.IMREAD_GRAYSCALE)
cv.imwrite("url_image.jpg", image)