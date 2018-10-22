# Dense ML Model for classifying datasets of small images.
# This set expects 28x28 RGB or grayscale images.
# Author: Charles McPherson, Jr.

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# load_data()
# Purpose:  This function loads in the image data and converts the images to a numpy
#           array of grayscale images with values from 0 to 1.  The function expects
#           the images to be a certain size, and will not accept images of any other.
def load_images(fPath):
    IMAGE_WIDTH = 32
    IMAGE_HEIGHT = 32
    stack = np.array([]) # Create empty array for image stack.
    extensions = ['png','jpg']
    # List all of the files in the directory.
    try:
        fileList = os.listdir(fPath)
    except OSError:
        print('Error: invalid path.');
        return -1
    # Go through the files, and read them into the array.
    for f in fileList:
        t = f.rsplit('.') # Check the file extension.
        if(len(t) == 2):
            if(t[1] in extensions):
                I = plt.imread(os.path.join(fPath,f))
                # Check image dimensions,
                if(I.shape[0] == IMAGE_WIDTH and I.shape[1] == IMAGE_HEIGHT):
                    if(len(I.shape) == 3): # If the image is RGB, convert to grayscale.
                        I = 0.3*I[:,:,0] + 0.6*I[:,:,1] + 0.1*I[:,:,2]
                    if(stack.size == 0):
                        stack = np.dstack(np.transpose(I))
                    else:
                        I = np.dstack(np.transpose(I))
                        stack = np.append(stack,I,axis=0) # Append to image stack.
                else:
                    print("Warning: file '%s' rejected.  Unexpected image size." % f)
            else:
                print("Warning: file '%s' regected. Not an accepted format." % f)
    return stack # Return the image stack.

# load_labels()
# Purpose:  This function loads the labels for the image data set.
def load_labels(labelFile):
    labels = []
    try:
        with open(labelFile, 'r') as f:
            for line in f:
                try:
                    labels.append(int(line))
                except ValueError:
                    print('Label must be an integer.  Closing file.')
                    break
    except OSError:
        print('Unable to open label file.')
    labelArray = np.array(labels) # Convert to numpy array.
    return labelArray # Return the numpy array.

if __name__ == '__main__':
    train_images = load_images('./training_small')
    train_labels = load_labels('./training_small/training_small.txt')

    test_images = load_images('./test_small')
    test_labels = load_labels('./test_small/test_small.txt')

    train_images = train_images/255.0
    
    class_names = ['Drum Set','Piano']

    # Build the model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32,32)),
        keras.layers.Dense(64,activation=tf.nn.relu),
        keras.layers.Dense(2,activation=tf.nn.softmax)])

    # Compile
    model.compile(optimizer = tf.train.AdamOptimizer(),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'])

    # Train the model.
    model.fit(train_images, train_labels, epochs=5)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Model accuracy: ' + str(test_acc))
