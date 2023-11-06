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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
import sys
import hyperopt as hp

sys.path.append('../../online-pointing-utils/python/tps_text_to_image')
import create_images_from_tps_libs as tp2img


def build_model(n_classes, train_images, train_labels, parameters):

    model = tf.keras.Sequential()

    model.add(layers.Conv2D(32, (30, 3), activation='relu', input_shape=(1000, 70, 1)))

    for i in range(parameters['n_conv_layers']):
        model.add(layers.Conv2D(parameters['n_filters']//(i+1), (parameters['kernel_size'], 1), activation='relu'))
        model.add(layers.LeakyReLU(alpha=0.1))
        model.add(layers.MaxPooling2D((5, 2)))
    
    model.add(layers.Flatten())
    for i in range(parameters['n_dense_layers']):
        model.add(layers.Dense(parameters['n_dense_units']//(i+1), activation='relu'))
        model.add(layers.LeakyReLU(alpha=0.1))
    
    model.add(layers.Dense(parameters['n_dense_units']//parameters['n_dense_layers'], activation='linear'))
    model.add(layers.Dense(n_classes, activation='softmax'))  

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=parameters['learning_rate'],
        decay_steps=10000,
        decay_rate=parameters['decay_rate'])
    
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_schedule),
                loss='categorical_crossentropy',
                loss_weights=parameters['loss_weights'],
                metrics=['accuracy'])   

    callbacks = [
        keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor='val_loss',
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=5,
            verbose=1)
    ]

    train_val_split = 0.8   
    train_val_split_index = int(train_images.shape[0]*train_val_split)
    train_images, val_images = train_images[:train_val_split_index], train_images[train_val_split_index:]
    train_labels, val_labels = train_labels[:train_val_split_index], train_labels[train_val_split_index:]



    history = model.fit(train_images, train_labels, epochs=500, batch_size=32, validation_data=(val_images, val_labels), callbacks=callbacks, verbose=0)

    return model, history

def hypertest_model(parameters, x_train, y_train, x_test, y_test, n_classes):
    model,_=build_model(n_classes=n_classes, train_images=x_train, train_labels=y_train, parameters=parameters)
    loss, accuracy=model.evaluate(x_test, y_test)
    return {'loss': -accuracy, 'status': hp.STATUS_OK}




def calculate_metrics( y_true, y_pred,):
    # calculate the confusion matrix, the accuracy, and the precision and recall 
    y_pred_am = np.argmax(y_pred, axis=1)
    y_true_am = np.argmax(y_true, axis=1)
    cm = confusion_matrix(y_true_am, y_pred_am, normalize='true')
    accuracy = accuracy_score(y_true_am, y_pred_am)
    precision = precision_score(y_true_am, y_pred_am, average='macro')
    recall = recall_score(y_true_am, y_pred_am, average='macro')
    f1 = f1_score(y_true_am, y_pred_am, average='macro')

    return cm, accuracy, precision, recall, f1
    
def log_metrics(y_true, y_pred, label_names=[0,1,2,3,4,5,6,7,8,9], epoch=0, test=False, output_folder="", model_name="model"):
    cm, accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)
    print("Confusion Matrix")
    print(cm)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    # save to file
    if test:
        if not os.path.exists(output_folder+f"log/test/"):
            os.makedirs(output_folder+f"log/test/")
        with open(output_folder+f"log/test/{model_name}_metrics_test.txt", "a") as f:
            f.write("Confusion Matrix\n")
            f.write(str(cm)+"\n")
            f.write("Accuracy: "+str(accuracy)+"\n")
            f.write("Precision: "+str(precision)+"\n")
            f.write("Recall: "+str(recall)+"\n")
            f.write("F1: "+str(f1)+"\n")
    else:
        if not os.path.exists(output_folder+f"log/train/"):
            os.makedirs(output_folder+f"log/train/")
        with open(output_folder+f"log/train/{model_name}_metrics_train.txt", "a") as f:
            f.write("Confusion Matrix\n")
            f.write(str(cm)+"\n")
            f.write("Accuracy: "+str(accuracy)+"\n")
            f.write("Precision: "+str(precision)+"\n")
            f.write("Recall: "+str(recall)+"\n")
            f.write("F1: "+str(f1)+"\n")

    # save confusion matrix 
    plt.figure(figsize=(10,10))
    plt.title("Confusion matrix")
    sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels=label_names, yticklabels=label_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if test:
        if not os.path.exists(output_folder+f"log/test/"):
            os.makedirs(output_folder+f"log/test/")
        plt.savefig(output_folder+f"log/test/{model_name}_confusion_matrix_test_{epoch}.png")
    else:
        if not os.path.exists(output_folder+f"log/train/"):
            os.makedirs(output_folder+f"log/train/")
        plt.savefig(output_folder+f"log/train/{model_name}_confusion_matrix_train_{epoch}.png")
    plt.clf()
    # Compute ROC curve and ROC area for each class
    # Binarize the output
    y_test = label_binarize(y_true, classes=np.arange(len(label_names)))
    n_classes = y_test.shape[1]
    
    fpr = dict()
    tpr = dict()

    plt.figure()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(label_names[i], roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    if test:
        if not os.path.exists(output_folder+f"log/test/"):
            os.makedirs(output_folder+f"log/test/")
        plt.savefig(output_folder+f"log/test/{model_name}_roc_curve_test_{epoch}.png")
    else:
        if not os.path.exists(output_folder+f"log/train/"):
            os.makedirs(output_folder+f"log/train/")
        plt.savefig(output_folder+f"log/train/{model_name}_roc_curve_train_{epoch}.png")

def save_sample_img(ds_item, output_folder, img_name):
    if ds_item.shape[2] == 1:
        img = ds_item[:,:,0]
        plt.figure(figsize=(10, 26))
        plt.title(img_name)
        plt.imshow(img)
        # plt.colorbar()
        # add x and y labels
        plt.xlabel("Channel")
        plt.ylabel("Time (ticks)")
        # save the image, with a bbox in inches smaller than the default but bigger than tight
        plt.savefig(output_folder+img_name+".png", bbox_inches='tight', pad_inches=1)
        plt.close()

    else:
        img_u = ds_item[:,:,0]
        img_v = ds_item[:,:,1]
        img_x = ds_item[:,:,2]
        fig = plt.figure(figsize=(8, 20))
        grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                        nrows_ncols=(1,3),
                        axes_pad=0.5,
                        share_all=True,
                        cbar_location="right",
                        cbar_mode="single",
                        cbar_size="30%",
                        cbar_pad=0.25,
                        )   


        if img_u[0, 0] != -1:
            im = grid[0].imshow(img_u)
            grid[0].set_title('U plane')
        if img_v[0, 0] != -1:
            im = grid[1].imshow(img_v)
            grid[1].set_title('V plane')
        if img_x[0, 0] != -1:
            im = grid[2].imshow(img_x)
            grid[2].set_title('X plane')
        grid.cbar_axes[0].colorbar(im)
        grid.axes_llc.set_yticks(np.arange(0, img_u.shape[0], 100))
        # save the image
        plt.savefig(output_folder+ 'multiview_' + img_name + '.png')
        plt.close()

def save_samples_from_ds(dataset, labels, output_folder, name="img", n_samples_per_label=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # get the labels
    labels_unique = np.unique(labels, return_counts=True)
    # get the samples
    for label in labels_unique[0]:
        # get the indices
        indices = np.where(labels == label)[0]
        indices = indices[:np.minimum(n_samples_per_label, indices.shape[0])]
        samples = dataset[indices]
        # save the images
        for i, sample in enumerate(samples):
            save_sample_img(sample, output_folder, name+"_"+str(label)+"_"+str(i))
    # make one image for each label containing n_samples_per_label images, using plt suplot
    fig= plt.figure(figsize=(20, 20))
    for i, label in enumerate(labels_unique[0]):
        grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                nrows_ncols=(1,10),
                axes_pad=0.5,
                share_all=True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="30%",
                cbar_pad=0.25,
                )   
        indices = np.where(labels == label)[0]
        indices = indices[:np.minimum(n_samples_per_label, indices.shape[0])]
        samples = dataset[indices]
        # save the images
        plt.suptitle("Label: "+str(label), fontsize=25)

        for j, sample in enumerate(samples):
            im = grid[j].imshow(sample[:,:,0])
            grid[j].set_title(j)

        grid.cbar_axes[0].colorbar(im)
        grid.axes_llc.set_yticks(np.arange(0, sample.shape[0], 100))
        plt.savefig(output_folder+ 'all_'+str(label)+'.png')
        plt.clf()

def create_labels(dataset_label):
    # create more intelligent labels
    effective_labels = np.zeros(dataset_label.shape)
    label_names = []
    n_classes = np.unique(dataset_label).shape[0]
    for i, label in enumerate(np.unique(dataset_label)):
        label_names.append(label)
        effective_labels[np.where(dataset_label == label)] = i
    return effective_labels, label_names, n_classes
