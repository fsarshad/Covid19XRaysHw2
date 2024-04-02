from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.applications import ResNet101

IMG_SHAPE = (192, 192, 3)

base_model_3 = ResNet101(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

base_model_3.summary()

base_model_3.trainable = False

gap3 = GlobalAveragePooling2D()(base_model_3.layers[-1].output)
output_3 = Dense(3, activation='softmax')(gap3)
# define new model
model_resnet101 = Model(inputs=base_model_3.inputs, outputs=output_3)

# summarize
model_resnet101.summary()

with tf.device('/device:GPU:0'):
  from tensorflow.python.keras.callbacks import ReduceLROnPlateau
  from tensorflow.python.keras.callbacks import ModelCheckpoint

  mc = ModelCheckpoint('best_model_3.h5', monitor='val_acc',mode='max', verbose=0, save_best_only=True)
  red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=0,factor=0.5, min_lr=0.001) # dividing lr by 2 when val_accuracy fails to improve after 2 epochs

  model_resnet101.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])

  model_resnet101.fit(train_dataset,batch_size=16,
          epochs = 20, verbose=1,validation_data=(test_dataset),callbacks=[mc,red_lr])

learning_rates_3=model_resnet101.history.history['lr'] # learning rates at each epoch

best_model_epoch_3=np.argmax(model_resnet101.history.history['val_acc'])+1 # epoch of best model

print(learning_rates_3)
print(best_model_epoch_3)

# Plot training & validation loss & accuracy values

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot training & validation loss values in the first subplot
axs[0].plot(model_resnet101.history.history['loss'])
axs[0].plot(model_resnet101.history.history['val_loss'])
axs[0].set_title('Model loss')
axs[0].set_ylabel('Loss')
axs[0].set_xlabel('Epoch')
axs[0].legend(['Train', 'Test'], loc='upper left')

# Plot training & validation accuracy values in the second subplot
axs[1].plot(model_resnet101.history.history['acc'])
axs[1].plot(model_resnet101.history.history['val_acc'])
axs[1].set_title('Model accuracy')
axs[1].set_ylabel('Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].legend(['Train', 'Test'], loc='upper left')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the combined plot
plt.show()
