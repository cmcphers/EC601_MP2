# Dense ML Model for classifying datasets of small images.
# This set expects 28x28 RGB or grayscale images.
# Author: Charles McPherson, Jr.

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

# load_data()
# Purpose:  This function loads in the image data and converts the images to a numpy
#           array of grayscale images with values from 0 to 1.  The function expects
#           the images to be a certain size, and will not accept images of any other.
def load_images(fPath):
    stack = np.array([]) # Create empty array for image stack.
    extensions = ['jpg']
    # List all of the files in the directory.
    try:
        fileList = os.listdir(fPath)
    except OSError:
        print('Error: invalid path.');
        return -1
    # Count the jpg images.
    count = 0
    for f in fileList:
        t = f.rsplit('.')
        if(len(t) == 2):
            if t[1] == 'jpg':
                count = count + 1
    
    # Go through the files, and read them into the array.
    # os.listdir() outputs will not be ordered by filename, so grab them in order here.
    i = 0 # Image index
    n = 0 # Number of images pulled up.
    while n < count and i < 1000:
        try:
            f = '%03d.jpg' % i
            if f in fileList:
                fileList.pop(fileList.index(f)) # Remove that item from the list.
                n = n + 1
                I = plt.imread(os.path.join(fPath,f)) # Read in the image.
                # Check dimensions.
                if(I.shape[0] == IMAGE_WIDTH and I.shape[1] == IMAGE_HEIGHT):
                    if(len(I.shape) == 3): # If the image is RGB, convert to grayscale.
                        I = 0.3*I[:,:,0] + 0.6*I[:,:,1] + 0.1*I[:,:,2]
                    if(stack.size == 0):
                        stack = np.dstack(np.transpose(I))
                    else:
                        I = np.dstack(np.transpose(I))
                        stack = np.append(stack,I,axis=0) # Append to image stack.
                else:
                    print("Warning: file %s rejected. Unexpected image size." % f)
            else:
                print("Warning: file %s does not exist.  Skipping." % f)
            i = i + 1
            
        except KeyboardInterrupt:
            break
    return stack # Return the image list.


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

    # Reshape images for input to CNN.
    train_images = train_images.reshape(train_images.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, 1)
    test_images = test_images.reshape(test_images.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, 1)

    if((train_images.shape[0] == len(train_labels)) and (test_images.shape[0] == len(test_labels))):

        train_images = train_images/255.0
        test_images = test_images/255.0

        class_names = ['Drum Set','Piano']

        # Build the model
        model = keras.Sequential([
            keras.layers.Conv2D(32, kernel_size=(5,5), strides=(1,1), activation='relu', input_shape=(32,32,1)),
            keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)),
            keras.layers.Conv2D(64, kernel_size=(5,5), strides=(1,1), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(2, activation='softmax')])

        # Compile
        model.compile(optimizer = tf.train.AdamOptimizer(),
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])

        # Train the model.
        model.fit(train_images, train_labels, epochs=7)

        test_loss, test_acc = model.evaluate(test_images, test_labels)

        print('Model accuracy: ' + str(test_acc))

        # Get the predictions for each of the test images. 
        pred = model.predict(test_images)

        # Reshape images for display.
        test_images = test_images.reshape(test_images.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT)
        
        # Display each image in the test set with its prediction.
        for i in range(len(pred)):
            plt.imshow(test_images[i])
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            if(np.argmax(pred[i]) == test_labels[i]):
                plt.title('Predicted: %s (%d %%), Actual: %s' % (class_names[np.argmax(pred[i])], 100*np.max(pred[i]), class_names[test_labels[i]]), color='#0000FF')
            else:
                plt.title('Predicted: %s (%d %%), Actual: %s' % (class_names[np.argmax(pred[i])], 100*np.max(pred[i]), class_names[test_labels[i]]), color='#FF0000')
            plt.show() # Display the plot.
            
    elif(train_images.shape[0] == len(train_labels)):
        print('Error: Number of images in test set does not match number of labels. Aborting.')
        print('N images = %d, N labels = %d' % (test_images.shape[0], len(test_labels)))
    else:
        print('Error: Number of images in training set does not match number of labels.  Aborting.')
        print('N images = %d, N labels = %d' % (train_images.shape[0], len(train_labels)))
