# =============================================================================
# PROBLEM C2
#
# Create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 91%
# =============================================================================

import tensorflow as tf

class StopCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # keys = list(logs.keys())
        if (epoch > 15):
          if(logs['accuracy'] > 0.92 and logs['val_accuracy'] > 0.92):
              print("Target Reached !, Stopping ...")
              self.model.stop_training = True

def solution_C2():
    mnist = tf.keras.datasets.mnist

    (training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # NORMALIZE YOUR IMAGE HERE
    training_images = training_images / 255.0
    test_images = test_images / 255.0

    # DEFINE YOUR MODEL HERE
    # End with 10 Neuron Dense, activated by softmax
    model = tf.keras.models.Sequential([
        # Add convolutions and max pooling
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Add the same layers as before
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.summary()
    # COMPILE MODEL HERE

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # TRAIN YOUR MODEL HERE
    model.fit(training_images, training_labels, epochs=100, validation_data=(test_images,test_labels), batch_size=128, callbacks=[StopCallback()])


    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C2()
    model.save("model_C2.h5")
