import matplotlib.pyplot as plt

def plot_history(history, title='Model Performance'):
    # Plot the training history (loss + accuracy)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    fig.suptitle(title, fontsize=16)

    # Plot training & validation loss values in the first subplot
    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].set_title('Model loss')
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation accuracy values in the second subplot
    if 'accuracy' in history.history:
        axs[1].plot(history.history['accuracy'])
        axs[1].plot(history.history['val_accuracy'])
    elif 'acc' in history.history:
        axs[1].plot(history.history['acc'])
        axs[1].plot(history.history['val_acc'])
    axs[1].set_title('Model accuracy')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Test'], loc='upper left')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the combined plot
    plt.show()