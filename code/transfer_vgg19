from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras import backend as K

base_model = VGG19(input_shape=(192,192,3), include_top=False, weights='imagenet')
base_model.summary()

base_model.trainable = False

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

flat1 = Flatten()(base_model.layers[-1].output)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(3, activation='softmax')(class1)
# define new model
model_vgg19 = Model(inputs=base_model.inputs, outputs=output)

# summarize
model_vgg19.summary()

with tf.device('/device:GPU:0'):
  from tensorflow.python.keras.callbacks import ReduceLROnPlateau
  from tensorflow.python.keras.callbacks import ModelCheckpoint

  mc = ModelCheckpoint('best_model.h5', monitor='val_acc',mode='max', verbose=1, save_best_only=True)
  red_lr= ReduceLROnPlateau(monitor='val_acc',patience=2,verbose=1,factor=0.5, min_lr=0.001) # dividing lr by 2 when val_accuracy fails to improve after 2 epochs

  model_vgg19.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])


  model_vgg19.fit(train_dataset,batch_size=1,
          epochs = 20, verbose=1,validation_data=(test_dataset),callbacks=[mc,red_lr])

import numpy as np
#Extract learning rate callback
learning_rates=model_vgg19.history.history['lr'] # learning rates at each epoch

best_model_epoch=np.argmax(model_vgg19.history.history['val_acc'])+1 # epoch of best model

print(learning_rates)
print(best_model_epoch)

# Plot training & validation loss & accuracy values

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot training & validation loss values in the first subplot
axs[0].plot(model_vgg19.history.history['loss'])
axs[0].plot(model_vgg19.history.history['val_loss'])
axs[0].set_title('Model loss')
axs[0].set_ylabel('Loss')
axs[0].set_xlabel('Epoch')
axs[0].legend(['Train', 'Test'], loc='upper left')

# Plot training & validation accuracy values in the second subplot
axs[1].plot(model_vgg19.history.history['acc'])
axs[1].plot(model_vgg19.history.history['val_acc'])
axs[1].set_title('Model accuracy')
axs[1].set_ylabel('Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].legend(['Train', 'Test'], loc='upper left')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the combined plot
plt.show()
