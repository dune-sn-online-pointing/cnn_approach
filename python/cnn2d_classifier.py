import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.sparse import SparseTensor, to_dense
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
import sys

sys.path.append('/afs/cern.ch/work/d/dapullia/public/dune/online-pointing-utils/python/tps_text_to_image')
import create_images_from_tps_libs as tp2img
import cnn2d_classifier_libs as cnn2d

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", help="input data file", default="/eos/user/d/dapullia/tp_dataset/dataset_img.npy", type=str)
parser.add_argument("--input_label", help="input label file", default="/eos/user/d/dapullia/tp_dataset/dataset_lab.npy", type=str)
parser.add_argument("--output_folder", help="save path", default="/eos/user/d/dapullia/tp_dataset/", type=str)
parser.add_argument("--model_name", help="model name", default="model.h5", type=str)
parser.add_argument('--load_model', action='store_true', help='save the model')

args = parser.parse_args()
input_data = args.input_data
input_label = args.input_label
output_folder = args.output_folder
model_name = args.model_name
load_model = args.load_model


if __name__=='__main__':

    if not os.path.exists(input_data) or not os.path.exists(input_label):
        print(input_data)
        print(input_label)
        print("Exists input data: ", os.path.exists(input_data))
        print("Exists input label: ", os.path.exists(input_label))
        print("Input file not found.")
        exit()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    # Check if GPU is available
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

    # Load the dataset
    print("Loading the dataset...")
    dataset_img = np.load(input_data)
    dataset_label = np.load(input_label)
    print("Dataset loaded.")
    print("Dataset_img shape: ", dataset_img.shape)
    print("Dataset_lab shape: ", dataset_label.shape)

    # Check if the dimension of images and labels are the same
    if dataset_img.shape[0] != dataset_label.shape[0]:
        print("Error: the dimension of images and labels are not the same.")
        exit()

    # Remove the images with label 10
    print("Different labels: ", np.unique(dataset_label, return_counts=True))
    print("Removing the images with label 10...")
    print("Dataset_img shape before: ", dataset_img.shape)
    print("Dataset_lab shape before: ", dataset_label.shape)
    index = np.where(dataset_label == 10)
    dataset_img = np.delete(dataset_img, index, axis=0)
    dataset_label = np.delete(dataset_label, index, axis=0)

    print("Dataset_img shape after: ", dataset_img.shape)
    print("Dataset_lab shape after: ", dataset_label.shape)
    print("Images with label 10 removed.")

    # Create more intelligent labels
    print("Creating labels...")
    dataset_label, label_names, n_classes = cnn2d.create_labels(dataset_label)
    print("Labels created.")

    # shuffle the dataset
    print("Shuffling the dataset...")
    index = np.arange(dataset_img.shape[0])
    np.random.shuffle(index)
    dataset_img = dataset_img[index]
    dataset_label = dataset_label[index]
    print("Dataset shuffled.")

    # Save some images
    print("Saving some images...")
    cnn2d.save_samples_from_ds(dataset_img, dataset_label, output_folder+"samples/", name="img", n_samples_per_label=10)
    print("Images saved.")


    # Split the dataset in training and test
    print("Splitting the dataset...")

    split = 0.8

    train_images = dataset_img[:int(dataset_img.shape[0]*split)]
    test_images = dataset_img[int(dataset_img.shape[0]*split):]
    train_labels = dataset_label[:int(dataset_label.shape[0]*split)]
    test_labels = dataset_label[int(dataset_label.shape[0]*split):]

    print("Dataset splitted.")
    print("Train images uniques: ", np.unique(train_labels, return_counts=True))
    print("Test images uniques: ", np.unique(test_labels, return_counts=True))

    # create 1 hot encoding for the labels
    print("Creating 1 hot encoding for the labels...")
    print("Train shape before: ",train_labels.shape)
    train_labels = keras.utils.to_categorical(train_labels)
    test_labels = keras.utils.to_categorical(test_labels)
    print("Train shape after: ",train_labels.shape)
    print("1 hot encoding created.")

    if not load_model or not os.path.exists(output_folder+model_name+".h5"):
        if not os.path.exists(output_folder+model_name+".h5"):
            print("Model not found, building a new one...")
        # Build the model, cnn 2D
        print("Building the model...")
        model = cnn2d.build_model(n_classes=n_classes)


        # Compile the model
        print("Compiling the model...")
        # add learning ratescheduler

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-1,
            decay_steps=10000,
            decay_rate=0.96)

        # Create weights for the loss function to account for the unbalanced dataset
        unique, counts = np.unique(dataset_label, return_counts=True)
        l_weights = []
        for i in range(len(unique)):
            l_weights.append(dataset_label.shape[0]/counts[i]) 
        print("Loss weights: ", l_weights)
            
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_schedule),
                        loss='categorical_crossentropy',
                        loss_weights=l_weights,
                        metrics=['accuracy'])   

        # plot using tf.keras.utils.plot_model
        tf.keras.utils.plot_model(model, to_file=output_folder+model_name+".png", show_shapes=True, show_layer_names=True)

        print("Model compiled.")

        print("Model built.")


        # Train the model
        print("Training the model...")

        callbacks = [
            keras.callbacks.EarlyStopping(
                # Stop training when `val_loss` is no longer improving
                monitor='val_loss',
                # "no longer improving" being further defined as "for at least 2 epochs"
                patience=8,
                verbose=1)
        ]


        history = model.fit(train_images, train_labels, epochs=500, batch_size=32, validation_data=(test_images, test_labels), callbacks=callbacks)
        print("Model trained.")

        # Evaluate the model
        print("Evaluating the model...")
        test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=32)
        print("Model evaluated.")
        print('Test accuracy:', test_acc)
        print('Test loss:', test_loss)

        # Plot the training and validation loss
        print("Plotting the training and validation loss...")
        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        acc_values = history_dict['accuracy']
        val_acc_values = history_dict['val_accuracy']
        epochs = range(1, len(loss_values) + 1)
        plt.figure()
        plt.plot(epochs, loss_values, 'bo', label='Training loss')  # bo = blue dot
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')  # b = "solid blue line"
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(output_folder+model_name+"_loss.png")
        plt.figure()
        plt.plot(epochs, acc_values, 'bo', label='Training accuracy')  # bo = blue dot
        plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')  # b = "solid blue line"
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(output_folder+model_name+"_accuracy.png")
        print("Plot saved.")

        # Save the model
        print("Saving the model...")
        model.save(output_folder+model_name+".h5")
        print("Model saved.")


    else:
        # Load the model
        model = keras.models.load_model(output_folder+model_name+".h5")

    # Do some test
    print("Doing some test...")
    predictions = model.predict(test_images)   
    print("Labels unique: ", np.unique(np.argmax(test_labels, axis=1), return_counts=True))
    print("Predictions unique: ", np.unique(np.argmax(predictions, axis=1), return_counts=True))
    # Calculate metrics
    print("Calculating metrics...")
    cnn2d.log_metrics(test_labels, predictions, label_names=label_names, test=True, output_folder=output_folder, model_name=model_name)
    print("Metrics calculated.")
    print("Test done.")




