import random
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from utils import load_data

def get_augmented_image_generator():
    # Define the augmentation parameters
    return ImageDataGenerator(
        rotation_range=20, # Rotate the image by up to 15 degrees
        width_shift_range=0.1, # Shift the image horizontally by up to 10%
        height_shift_range=0.1, # Shift the image vertically by up to 10%
        shear_range=0.1, # Shear the image by up to 10%
        zoom_range=0.1, # Zoom the image by up to 10%
        # horizontal_flip=True, # Flip the image horizontally
        brightness_range=[0.8, 1.2], # Adjust the brightness of the image
        fill_mode='nearest' # Fill in missing pixels,
    )

# If this script is run directly, display an example of the augmentation
if __name__ == '__main__':
    X_train, _, _, _ = load_data() # Only need the training data to display an example
    augmented_image_generator = get_augmented_image_generator()
    
    # Apply the augmentation to an example image 
    example_image = random.choice(X_train)
    example_image = example_image.reshape((1,) + example_image.shape)
    augmented_images = augmented_image_generator.flow(example_image, batch_size=1)

    # Display the original and augmented images
    num_examples = 6
    fig, ax = plt.subplots(nrows=1, ncols=num_examples, figsize=(15, 15))
    ax[0].imshow(example_image[0])
    ax[0].set_title('Original')
    ax[0].axis('off')
    for i in range(1,num_examples):
        augmented_image = next(augmented_images)[0]
        ax[i].imshow(augmented_image)
        ax[i].axis('off')
        ax[i].set_title(f'Augmented v{i+1}')