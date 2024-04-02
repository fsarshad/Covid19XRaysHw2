from tensorflow.keras.applications import DenseNet201
IMG_SHAPE = (192, 192, 3)
base_model_2 = DenseNet201(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model_2.summary()

base_model_2.trainable = False

flat2 = Flatten()(base_model_2.layers[-1].output)
class2 = Dense(1024, activation='relu')(flat2)
output_2 = Dense(3, activation='softmax')(class2)
# define new model
model_dense201 = Model(inputs=base_model_2.inputs, outputs=output_2)

# summarize
model_dense201.summary()

with tf.device('/device:GPU:0'):
  from tensorflow.python.keras.callbacks import ReduceLROnPlateau
  from tensorflow.python.keras.callbacks import ModelCheckpoint

  mc = ModelCheckpoint('best_model_2.h5', monitor='val_acc',mode='max', verbose=1, save_best_only=True)
  red_lr= ReduceLROnPlateau(monitor='val_acc',patience=2,verbose=1,factor=0.5, min_lr=0.001) # dividing lr by 2 when val_accuracy fails to improve after 2 epochs

  model_dense201.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])

  model_dense201.fit(train_dataset,batch_size=1,
          epochs = 20, verbose=1,validation_data=(test_dataset),callbacks=[mc,red_lr])

learning_rates_2=model_dense201.history.history['lr'] # learning rates at each epoch

best_model_epoch_2=np.argmax(model_dense201.history.history['val_acc'])+1 # epoch of best model

print(learning_rates_2)
print(best_model_epoch_2)

# Plot training & validation loss & accuracy values

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot training & validation loss values in the first subplot
axs[0].plot(model_dense201.history.history['loss'])
axs[0].plot(model_dense201.history.history['val_loss'])
axs[0].set_title('Model loss')
axs[0].set_ylabel('Loss')
axs[0].set_xlabel('Epoch')
axs[0].legend(['Train', 'Test'], loc='upper left')

# Plot training & validation accuracy values in the second subplot
axs[1].plot(model_dense201.history.history['acc'])
axs[1].plot(model_dense201.history.history['val_acc'])
axs[1].set_title('Model accuracy')
axs[1].set_ylabel('Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].legend(['Train', 'Test'], loc='upper left')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the combined plot
plt.show()
