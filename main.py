import tensorflow as tf # pytorch would not work for some reason so i had to use the dreaded tensorflow
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tensordata
import matplotlib.pyplot as plt
# mnist dataabase (pretty good for being free)
ds_train, ds_info = tensordata.load('mnist', split='train', shuffle_files=True, as_supervised=True, with_info=True)
ds_test = tensordata.load('mnist', split='test', shuffle_files=True, as_supervised=True)
def changedata(image, label):
    image = tf.image.resize(image, [28, 28])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, depth=10)
    return image, label
# compiling mnist databse into something that is good for the cnn
ds_train = ds_train.map(changedata).batch(64).prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test.map(changedata).batch(64).prefetch(tf.data.experimental.AUTOTUNE)
# might change this later
model = models.Sequential()
# 3 layers, try changing relu to sigmoid or tanh
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) # adam optimizer for loss bc its efficient
stages = int(input("how many training stages (epochs): "))
history = model.fit(ds_train, epochs=stages, validation_data=ds_test)
test_loss, test_acc = model.evaluate(ds_test)
print("accuracy: ", test_acc)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='test accuracy')
plt.xlabel('stages')
plt.ylabel('accuracy')
plt.legend()
plt.show()
model.save('mnist_cnn_model.h5')
model.summary()
