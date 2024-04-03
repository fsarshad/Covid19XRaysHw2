import os
import pickle
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from itertools import repeat
import matplotlib.pyplot as plt


# Load data as images from a directory
def load_images(base_path, image_size=(256, 256)):
    categories = ['COVID/', 'Normal/', 'Viral Pneumonia/']

    # load file names to fnames list object
    fnames = []
    for category in categories:
        image_folder = os.path.join(base_path, category)
        file_names = os.listdir(image_folder)
        full_path = [os.path.join(image_folder, file_name) for file_name in file_names]
        fnames.append(full_path)

    value_counts = [len(f) for f in fnames]
    # print('Number of images for each category:', value_counts)

    def preprocessor(img_path):
        img = Image.open(img_path).convert("RGB").resize(image_size) # import image, make sure it's RGB and resize to height and width you want.
        img = (np.float32(img)-1.)/(255-1.) # min max transformation
        img=img.reshape(image_size + (3,)) # Create final shape as array with correct dimensions for Keras
        return img

    # Import image files iteratively and preprocess them into array of correctly structured data
    # Create list of file paths
    image_filepaths=fnames[0]+fnames[1]+fnames[2]

    # Iteratively import and preprocess data using map function
    # map functions apply your preprocessor function one step at a time to each filepath
    preprocessed_image_data=list(map(preprocessor, image_filepaths))

    # Object needs to be an array rather than a list for Keras (map returns to list object)
    X = np.array(preprocessed_image_data) # Assigning to X to highlight that this represents feature input data for our model

    # Normalize the image pixel values to 0-1 range
    X = (X - X.min())/(X.max()-X.min())

    # Create y data made up of correctly ordered labels from file folders
    # Recall that we have five folders with the following number of images in each folder
    #...corresponding to each X-ray category
    covid=list(repeat("COVID", value_counts[0]))
    normal=list(repeat("NORMAL", value_counts[1]))
    pneumonia=list(repeat("PNEUMONIA", value_counts[2]))

    #combine into single list of y labels
    y_labels = covid + normal + pneumonia

    #check length, same as X above
    # print('Size of y:', len(y_labels))

    # Need to one hot encode for Keras.  Let's use Pandas
    import pandas as pd
    y = pd.get_dummies(y_labels)

    return X, y

# Load data from pickle files
def load_data_from_pkl(): 
    with open('../data/X_train.pkl', 'rb') as file: X_train = pickle.load(file)
    with open('../data/y_train.pkl', 'rb') as file: y_train = pickle.load(file)
    with open('../data/X_test.pkl', 'rb') as file: X_test = pickle.load(file)
    with open('../data/y_test.pkl', 'rb') as file: y_test = pickle.load(file)
    return X_train, y_train, X_test, y_test

# Define the augmentation parameters
def get_image_augmentation_pipeline():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1), # Rotate up to 10% * 2pi
        tf.keras.layers.RandomZoom(0.2), # Zoom up to 20%
        tf.keras.layers.RandomContrast(0.2), # Adjust contrast up to 20%
        tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode='nearest'), # Translate up to 10% in x and y
        # tf.keras.layers.RandomBrightness(0.2), # Adjust brightness up to 20%
    ])

def get_datasets(augment=True):
    # Load the data
    X_train, y_train, X_test, y_test = load_data_from_pkl()

    # Create datasets using the tf.data API
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # Apply the augmentation to the datasets
    if augment:
        image_augmentations = get_image_augmentation_pipeline()
        train_dataset = train_dataset.batch(16).map(lambda x, y: (image_augmentations(x), y))
        test_dataset = test_dataset.batch(16).map(lambda x, y: (image_augmentations(x), y))
        # ^ NOTE: Akarsh mentioned in class to augment the test/validation dataset as well  
    return train_dataset, test_dataset

# Display an example of the image augmentation
if __name__ == '__main__':
    X_train, _, _, _ = load_data_from_pkl()  # Only need the training data to display an example

    image_augmentations = get_image_augmentation_pipeline()

    # Get a random image from the training data to augment
    example_image = random.choice(X_train)
    example_image = example_image.reshape((1,) + example_image.shape)

    # Display the original and augmented images
    num_examples = 6
    fig, ax = plt.subplots(nrows=1, ncols=num_examples, figsize=(10, 2))
    ax[0].imshow(example_image[0].astype("float32"))
    ax[0].set_title('Original')
    ax[0].axis('off')
    for i in range(1, num_examples):
        augmented_image = image_augmentations(example_image)
        ax[i].imshow(augmented_image[0].numpy().astype("float32"))
        ax[i].axis('off')
        ax[i].set_title(f'Augmented v{i+1}')
    plt.show()