# EC601_MP2

1. Introduction:

These are two simple binary classifiers for images.  They determine 
whether the item in the image is a piano or a drum set. 

2. System Requirements:

The programs here require the following Python libraries to be installed:

    a. numpy
    b. matplotlib
    c. PIL (pillow)
    d. tensorflow

Tensorflow requires Python 3.5.0 or 3.6.0.  For Mac OSX, it requires at
Mac OS 10.12 (Sierra)

3. Instructions

    i. Begin by installing the required libraries mentioned in section 2.
        a. pip install numpy
        b. pip install matplotlib
        c. pip install pil
        d. pip install tensorflow

    ii. Then, unzip 'training_small.zip' and 'test_small.zip.'  These are
    the training and test sets, respectively.

    iii. Run 'MP02_DenseModel.py' to run the fully-connected neural network
    model.

    iv. Run 'MP02_CNNModel.py' to run the convolutional nerual network model.

4. Data Requirements

    i. Images must be 32x32 jpg images (with .jpg extension).  Images can by
    RGB or grayscale.

    ii. Training images must be in a folder titled 'training_small' with file
    names running from 000.jpg to 999.jpg.  Gaps in this order are allowed.

    iii. 'training_small' folder must contain a text file called 
    'training_small.txt' containing the labels.  The labels must be 0 for 
    'drum set' or 1 for 'piano' and only one label per line, followed by a
    newline character.  The zeros and ones for the labels must be in the
    same order as the files in the folder.

    iv. Test images must be in a folder titled 'test_small' with file names
    running from 000.jpg to 999.jpg.  Gaps in this order are allowed.

    v. 'test_small' folder must contain a text file called 'test_small.txt'
    containing the labels.  The labels must be 0 for 'drum set' or 1 for 
    'piano' and only one label per line, followed by a newline character.
    The zeros and ones must be in the order of the files in the folder.
