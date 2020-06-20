import numpy
import tensorflow as tf
# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model

ParentPath = "E:\\Study\\Pill\\Shape\\Binary\\"

LossFunction = 'categorical_crossentropy'
ClassMode = 'categorical'
img_width, img_height = 150, 150

nb_train_samples = 30755
nb_validation_samples = 368
NumberOfClass = 15
epochs = 100
batchSize = 16

EarlyStoppingPatience = 10


train_data_dir = ParentPath +'Train'
validation_data_dir = ParentPath + 'Validation'


SaveModelPath = ParentPath +  'Model'
SaveModelPathForEarlyStopping = SaveModelPath +  "/{epoch:02d}-{val_acc:.4f}.hdf5"

def GetTopModel(model_input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=model_input_shape))
    model.add(Dense((NumberOfClass*2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NumberOfClass, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss=LossFunction, metrics=['accuracy'])

    return model


base_model = applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
top_model = GetTopModel(base_model.output_shape[1:])

model = Model(input= base_model.input, output= top_model(base_model.output))

model.compile(loss=LossFunction,
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])



train_datagen = ImageDataGenerator(rescale=1. / 255)


validation_datagen = ImageDataGenerator(
        rescale=1. / 255,
        )


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batchSize,
    class_mode=ClassMode)

print(train_generator.class_indices)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size= batchSize,
    class_mode= ClassMode)

from keras.callbacks import ModelCheckpoint,EarlyStopping
checkpointer = ModelCheckpoint(filepath=SaveModelPathForEarlyStopping, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=EarlyStoppingPatience)

import time

start = time.time()
# fine-tune the model
history = model.fit_generator(
    train_generator,
    samples_per_epoch= nb_train_samples // batchSize,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps = nb_validation_samples// batchSize,
    callbacks=[early_stopping_callback, checkpointer])

time = time.time() - start
print("학습 시 소요 시간 : {}".format(time))
#model.save(Model_Path)

import numpy as np
import matplotlib.pyplot as plt

y_vloss=history.history['val_loss']
y_acc=history.history['acc']

x_len = np.arange(len(y_acc))
plt.plot(x_len, y_vloss, "o", c="red", markersize=3)
plt.plot(x_len, y_acc, "o", c="blue", markersize=3)

plt.show()