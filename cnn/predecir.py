from keras import models

from keras._tf_keras.keras.preprocessing.image import img_to_array, load_img

import numpy as np

from cnn.data import class_names


model = './modelo/model.keras'

cnn = models.load_model(model)

def predict(img_url):

    # # redimensiono la imagen para que coincida con la data training del modelo y lo convierto a un array

    img = img_to_array(load_img(path=img_url, target_size=(28,28)))

    # # Promedio de los valores de píxeles a lo largo del eje 2 para convertir la imagen a escala de grises

    img = np.average(img, axis=2)

    # # # normalizo los valores de píxeles al rango [0, 1]

    img = 1-img/255 # invertimos el color de los pixeles, es decir, la prenda sera blanca y el fondo negro

    img_predict = img[np.newaxis, ...] # agrego una dimension, para respetar las dimensiones de las imagenes de entrenamiento

    predictions_single = cnn.predict(img_predict)

    label_predict = np.argmax(predictions_single[0])

    return class_names[label_predict]





