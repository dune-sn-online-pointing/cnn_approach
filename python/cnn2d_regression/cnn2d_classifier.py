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
import hyperopt as hp

sys.path.append('/afs/cern.ch/work/d/dapullia/public/dune/online-pointing-utils/python/tps_text_to_image')
import create_images_from_tps_libs as tp2img
sys.path.append('/afs/cern.ch/work/d/dapullia/public/dune/cnn_approach/python/libs/')
import cnn2d_regression_libs as cnn2d

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
parser.add_argument('--hyperopt', action='store_true', help='use hyperopt')
parser.add_argument('--hp_max_evals', help='max number of hyperopt evaluations', default=10, type=int)

args = parser.parse_args()
input_data = args.input_data
input_label = args.input_label
output_folder = args.output_folder
model_name = args.model_name
output_folder = output_folder + model_name + "/"
load_model = args.load_model
hyperopt = args.hyperopt
hp_max_evals = args.hp_max_evals


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
    dataset_img = np.load(input_data, allow_pickle=True)
    dataset_label = np.load(input_label)
    print("Dataset loaded.")
    print("Dataset_img shape: ", dataset_img.shape)
    print("Dataset_lab shape: ", dataset_label.shape)

    print("dataset_img type: ", type(dataset_img))
    print("dataset_lab type: ", type(dataset_label))

    # Check if the dimension of images and labels are the same
    if dataset_img.shape[0] != dataset_label.shape[0]:
        print("Error: the dimension of images and labels are not the same.")
        exit()

    # Remove the images with label 10
    print("Different labels: ", np.unique(dataset_label, return_counts=True))
    print("Removing the images with label 99...")
    print("Dataset_img shape before: ", dataset_img.shape)
    print("Dataset_lab shape before: ", dataset_label.shape)
    index = np.where(dataset_label == 99)
    dataset_img = np.delete(dataset_img, index, axis=0)
    dataset_label = np.delete(dataset_label, index, axis=0)

    print("Dataset_img shape after: ", dataset_img.shape)
    print("Dataset_lab shape after: ", dataset_label.shape)
    print("Images with label 10 removed.")

    # Create more intelligent labels
    print("Labels unique: ", np.unique(dataset_label, return_counts=True))  

    print("Creating labels...")
    dataset_label, label_names, n_classes = cnn2d.create_labels(dataset_label)
    print("Labels created.")
    print("Labels unique: ", np.unique(dataset_label, return_counts=True))  

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
        if not hyperopt:
            # Build the model, cnn 2D
            print("Building the model...")
            # Create weights for the loss function to account for the unbalanced dataset
            unique, counts = np.unique(dataset_label, return_counts=True)
            l_weights = []
            for i in range(len(unique)):
                l_weights.append(dataset_label.shape[0]/counts[i]) 
            print("Loss weights: ", l_weights)

            parameters = {
                'n_conv_layers': 3,
                'n_dense_layers': 2,
                'n_filters': 64,
                'kernel_size': 3,
                'n_dense_units': 128,
                'learning_rate': 0.002,
                'decay_rate': 0.96,
                'pool_size': 5,
                'loss_weights': l_weights
            }
            model, history = cnn2d.build_model(n_classes=n_classes, train_images=train_images, train_labels=train_labels, parameters=parameters)
            # model, history = cnn2d.build_model_sparse_expe(n_classes=n_classes, train_images=train_images, train_labels=train_labels, parameters=parameters)
            print("Model built.")
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
            print("Hyperopt mode...")
            # Hyperopt
            # Define the search space
            unique, counts = np.unique(dataset_label, return_counts=True)
            l_weights = []
            for i in range(len(unique)):
                l_weights.append(dataset_label.shape[0]/counts[i]) 
            print("Loss weights: ", l_weights)
            space_options = {
                'n_conv_layers': [1, 2, 3, 4, 5],
                'n_dense_layers': [2, 3, 4],
                'n_filters': [16, 32, 64, 128],
                'kernel_size': [1, 3, 5],
                'n_dense_units': [32, 64, 128, 256],
                'learning_rate': [0.001, 0.01],
                'decay_rate': [0.90, 0.999],
                # 'pool_size': [3, 5, 10, 30],
                'loss_weights': [l_weights]
            }

            space = {
                'n_conv_layers': hp.hp.choice('n_conv_layers', space_options['n_conv_layers']),
                'n_dense_layers': hp.hp.choice('n_dense_layers', space_options['n_dense_layers']),
                'n_filters': hp.hp.choice('n_filters', space_options['n_filters']),
                'kernel_size': hp.hp.choice('kernel_size', space_options['kernel_size']),
                'n_dense_units': hp.hp.choice('n_dense_units', space_options['n_dense_units']),
                'learning_rate': hp.hp.uniform('learning_rate', space_options['learning_rate'][0], space_options['learning_rate'][1]),
                'decay_rate': hp.hp.uniform('decay_rate', space_options['decay_rate'][0], space_options['decay_rate'][1]),
                # 'pool_size': hp.hp.choice('pool_size', space_options['pool_size']),
                'loss_weights': hp.hp.choice('loss_weights', space_options['loss_weights'])
            }
            trials = hp.Trials()
            # Run the hyperparameter search
            print("Running the hyperparameter search...")
            best = hp.fmin(
                fn=lambda x: cnn2d.hypertest_model( parameters=x, n_classes=n_classes, x_train = train_images, y_train = train_labels, x_test = test_images, y_test = test_labels, output_folder=output_folder),  # objective function   
                space=space,
                algo=hp.tpe.suggest,
                max_evals=hp_max_evals,
                trials=trials
            )
            print("Hyperparameter search done.")
            # Get the best parameters
            print("Getting the best parameters...")
            best_dict = {
                'n_conv_layers': space_options['n_conv_layers'][best['n_conv_layers']],
                'n_dense_layers': space_options['n_dense_layers'][best['n_dense_layers']],
                'n_filters': space_options['n_filters'][best['n_filters']],
                'kernel_size': space_options['kernel_size'][best['kernel_size']],
                'n_dense_units': space_options['n_dense_units'][best['n_dense_units']],
                'learning_rate': best['learning_rate'],
                'decay_rate': best['decay_rate'],
                # 'pool_size': space_options['pool_size'][best['pool_size']],
                'loss_weights': space_options['loss_weights'][best['loss_weights']]
            }

            print("Best parameters: ", best_dict)
            print("Best loss: ", -trials.best_trial['result']['loss'])

            print("Best parameters saved.")

            # Save the trials
            print("Saving the trials...")
            np.save(output_folder+model_name+"_trials.npy", trials)
            print("Trials saved.")
            # Save the best parameters
            print("Saving the best parameters...")
            np.save(output_folder+model_name+"_best.npy", best)
            print("Best parameters saved.")
            # Save the best model
            print("Saving the best model...")
            model, history = cnn2d.build_model(n_classes=n_classes, train_images=train_images, train_labels=train_labels, parameters=best_dict)
            model.save(output_folder+model_name+".h5")
            print("Best model saved.")

            # plot the hyperparameter search
            print("Plotting the hyperparameter search...")
            # remove the trials with impossible loss
            trials = [t for t in trials.trials if t['result']['loss'] != 9999]
            
            plt.figure(figsize=(15, 15))
            plt.suptitle('Hyperparameters tuning')
            plt.subplot(3, 3, 1)
            plt.title('Accuracy vs n_conv_layers')
            plt.scatter([t['misc']['vals']['n_conv_layers'] for t in trials], [-t['result']['loss'] for t in trials], s=20, alpha=0.3, label='loss')
            plt.xlabel('n_conv_layers')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.subplot(3, 3, 2)
            plt.title('Accuracy vs n_dense_layers')
            plt.scatter([t['misc']['vals']['n_dense_layers'] for t in trials], [-t['result']['loss'] for t in trials], s=20, alpha=0.3, label='loss')
            plt.xlabel('n_dense_layers')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.subplot(3, 3, 3)
            plt.title('Accuracy vs n_filters')
            plt.scatter([t['misc']['vals']['n_filters'] for t in trials], [-t['result']['loss'] for t in trials], s=20, alpha=0.3, label='loss')
            plt.xlabel('n_filters')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.subplot(3, 3, 4)
            plt.title('Accuracy vs kernel_size')
            plt.scatter([t['misc']['vals']['kernel_size'] for t in trials], [-t['result']['loss'] for t in trials], s=20, alpha=0.3, label='loss')
            plt.xlabel('kernel_size')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.subplot(3, 3, 5)
            plt.title('Accuracy vs n_dense_units')
            plt.scatter([t['misc']['vals']['n_dense_units'] for t in trials], [-t['result']['loss'] for t in trials], s=20, alpha=0.3, label='loss')
            plt.xlabel('n_dense_units')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.subplot(3, 3, 6)
            plt.title('Accuracy vs learning_rate')
            plt.scatter([t['misc']['vals']['learning_rate'] for t in trials], [-t['result']['loss'] for t in trials], s=20, alpha=0.3, label='loss')
            plt.xlabel('learning_rate')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.subplot(3, 3, 7)
            plt.title('Accuracy vs decay_rate')
            plt.scatter([t['misc']['vals']['decay_rate'] for t in trials], [-t['result']['loss'] for t in trials], s=20, alpha=0.3, label='loss')
            plt.xlabel('decay_rate')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.subplot(3, 3, 8)
            plt.title('Accuracy vs loss_weights')
            plt.scatter([t['misc']['vals']['loss_weights'] for t in trials], [-t['result']['loss'] for t in trials], s=20, alpha=0.3, label='loss')
            plt.xlabel('loss_weights')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.subplot(3, 3, 9)
            plt.title('Accuracy vs Trial ID')
            plt.scatter([t['tid'] for t in trials], [-t['result']['loss'] for t in trials], s=20, alpha=0.3, label='loss')
            plt.xlabel('Trial ID')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(output_folder+model_name+"_hyperopt_evolution.png")
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
    print("Drawing model...")
    keras.utils.plot_model(model, output_folder+model_name+".png", show_shapes=True)
    print("Model drawn.")

    print("Test done.")




