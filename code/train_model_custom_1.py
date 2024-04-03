from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import Adam

from utils import get_datasets
from data import get_image_augmentation_pipeline

# Load datasets
train_dataset, test_dataset = get_datasets(augment=True)

# Define model architecture
input_shape = (256, 256, 3)
model_custom_1 = Sequential([
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=1024, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile the model
model_custom_1.compile(
    optimizer=Adam(learning_rate=0.0001), # Reduced learning rate from 1e-3 to 1e-4
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
NUM_EPOCHS = 15

model_custom_1_history = model_custom_1.fit(
    train_dataset,
    epochs=NUM_EPOCHS,
    validation_data=test_dataset,
)