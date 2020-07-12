import numpy as np 
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

num_classes = 10

def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, 28, 28, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y

train_file = "train.csv"
raw_data = pd.read_csv(train_file)

X, y = data_prep(raw_data)
print(y.shape)

model = Sequential()
model.add(Conv2D(16, kernel_size = (3, 3),
                 activation = 'relu',
                 input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation = 'relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(num_classes, activation = 'softmax'))

#model.summary()


model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = 'adam',
              metrics = ['accuracy'])
model.fit(X, y,
          batch_size = 128,
          epochs = 6,
          validation_split = 0.2)