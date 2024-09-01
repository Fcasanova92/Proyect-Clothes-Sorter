
from keras import Sequential, models

from keras._tf_keras.keras.layers import Flatten, Dense

from data import  train_images_rezice, train_labels, test_images_rezice, test_labels

from tensorflowjs import converters


model = Sequential([
    Flatten(input_shape=(28, 28,1)),
    Dense(128, activation='relu'),
    Dense(128,activation='relu' ),
    Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# entreno al modelo, relaciona imagenes y etiquetas

model.fit(train_images_rezice, train_labels, epochs=10, batch_size=256, verbose=2)  


# evaluo la exactitud del modelo

test_loss, test_acc = model.evaluate(test_images_rezice,  test_labels, verbose=2)

# guardo el modelo ya entrenado 

models.save_model(filepath='./modelo/model.h5', model=model)

converters.save_keras_model(model, './modelo')





