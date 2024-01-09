import numpy as np
import argparse

seed = 42
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", help="input data file", default="/afs/cern.ch/work/d/dapullia/public/dune/cnn_approach/lists/maint-vs-all.txt", type=str)
parser.add_argument("--input_label", help="input label file", default="/afs/cern.ch/work/d/dapullia/public/dune/cnn_approach/lists/maint-vs-all-lab.txt", type=str)
parser.add_argument("--output_folder", help="save path", default="/eos/user/d/dapullia/tp_dataset/", type=str)
parser.add_argument("--remove_labels", help="remove labels", default=[], nargs='+', type=int)
parser.add_argument("--shuffle", help="shuffle data", default=1, type=int)
parser.add_argument("--balance", help="balance data", default=1, type=int)

args = parser.parse_args()
input_data = args.input_data
input_label = args.input_label
output_folder = args.output_folder
shuffle = args.shuffle
balance = args.balance
remove_labels = args.remove_labels

# read data and labels
data = np.loadtxt(input_data, dtype=str)
labels = np.loadtxt(input_label, dtype=str)

print("data shape: ", data)
print("labels shape: ", labels)

if len(data) != len(labels):
    print("data and labels have different lengths!")
    exit()
datasets = []
labelsets = []
for i in range(len(data)):
    print(i)
    print("loading dataset: ", data[i])
    dataset = np.load(data[i], allow_pickle=True)
    # check if labels[i] is a number or a string
    if labels[i].isdigit():
        print("label is a number: ", labels[i])
        lab_value = int(labels[i])
        labelset = np.full((len(dataset),1), (lab_value))
    else:
        labelset = np.load(labels[i])
    print("labelset shape: ", labelset.shape)
    datasets.append(dataset)
    labelsets.append(labelset)

# concatenate all datasets and labels
dataset = np.concatenate(datasets)
labelset = np.concatenate(labelsets)

# remove labels
mask = np.isin(labelset, remove_labels)
mask = np.reshape(mask, (len(mask)))
print(dataset.shape) # (193987, 300, 70, 1)
print(labelset.shape) # (193987,1)
print(mask.shape) # (193987,1)
labelset = labelset[~mask]
dataset = dataset[~mask]

# balance data, i.e. make sure that the number of events for each class is the same
if balance == 1:
    print("balancing data")
    # get all unique labels
    unique_labels, counts = np.unique(labelset, return_counts=True)
    print("unique labels: ", unique_labels)
    print("counts: ", counts)
    # get the minimum number of events for a class
    min_count = np.min(counts)
    print("min count: ", min_count)
    # loop over all unique labels
    for label in unique_labels:
        # get the indices of the events with this label
        indices = np.where(labelset == label)[0]
        # shuffle the indices
        np.random.shuffle(indices)
        # keep only the first min_count indices
        indices = indices[:min_count]
        reduced_dataset = dataset[indices]
        reduced_labelset = labelset[indices]
        # concatenate the reduced datasets and labelsets
        if label == unique_labels[0]:
            balanced_dataset = reduced_dataset
            balanced_labelset = reduced_labelset
        else:
            balanced_dataset = np.concatenate((balanced_dataset, reduced_dataset))
            balanced_labelset = np.concatenate((balanced_labelset, reduced_labelset))
    dataset = balanced_dataset
    labelset = balanced_labelset

    print("balanced dataset shape: ", dataset.shape)
    print("balanced labelset shape: ", labelset.shape)

# shuffle data
if shuffle == 1:
    print("shuffling data")
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    dataset = dataset[indices]
    labelset = labelset[indices]


print(np.unique(labelset, return_counts=True))

# save dataset and labelset
np.save(output_folder + "dataset_img.npy", dataset)
np.save(output_folder + "dataset_label.npy", labelset)
