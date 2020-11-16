# deactivate all conda environments,
# use the default base conda environment

"""
This script calculates the anomaly threshold after the DCGAN training,
through comparison of the real MFCCs from the dataset and the generated
MFCCs after the training.
Finally, this script loads files from the test dataset and creates a
ROC-curve to find a range of optimal thresholds.
"""

from skimage import io, color
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse

print("Imports are ready!")

parser = argparse.ArgumentParser(description = "Evaluate DCGAN.")
parser.add_argument('--train_real', help = "Path to the MFCC used for training of the DCGAN.", required = True)
parser.add_argument('--train_generated', help = "Path to the MFCC generated after the training of the DCGAN.", required = True)
parser.add_argument('--good_real', help = "Path to the directory with MFCCs recordings of a good bearing.", required = True)
parser.add_argument('--bad_real', help = "Path to the directory with MFCCs recordings with unexpected stops.", required = True)
parser.add_argument('--defect_real', help = "Path to the directory with MFCCs redordings of a defect bearing.", required = True)
parser.add_argument('--generated_mfccs', help = "Path to the directory with generated MFCCs.", required = True)
args = parser.parse_args()

def convert_to_gray(img_path):
    """
    This function converts the color image to black-white.
    :param img_path: the path to the color image
    :returns: a black-white image as a 2D array
    """
    color_img = io.imread(img_path)
    color_gray = color.rgb2gray(color_img)
    return color_gray

train_real_gray = convert_to_gray(args.train_real) #"../../mfccs/img/1200_200_0.jpg")
train_generated_gray = convert_to_gray(args.train_generated) #"./dcgan_mfccs_output/samples_745_0.png")

# check the shape of gray images
print(train_generated_gray.shape) #(20, 44)

# calculate the anomaly threshold as a maximum of differences
differences = np.abs(train_real_gray.flatten() - train_generated_gray.flatten())
threshold = np.max(differences)
print("The anomaly threshold is: ", threshold) #0.17

def read_files(directory):
    """
    This function reads the MFCCs in JPG format from the given directory.
    :param directory: the directory to read the files from
    :returns: a list of path to the MFCC files
    """
    files = []
    for f in os.listdir(directory):
        if f.endswith(".jpg"):
            files.append(os.path.join(directory, os.path.basename(f)))

    return files

# set the directories where we read the MFCCs for testing
good_real = read_files(args.good_real) #"../../mfccs/test/good")
bad_real = read_files(args.bad_real) #"../../mfccs/test/bad")
defect_real = read_files(args.defect_real) #"../../mfccs/test/defect")

good_generated = read_files(args.generated_mfccs) #"./dcgan_mfccs_output/test/good")
bad_generated = good_generated #read_files("./dcgan_mfccs_output/test/bad")
defect_generated = good_generated #read_files("./dcgan_mfccs_output/test/defect")

# create a range of thresholds in regard to the calculated threshold
thresholds = [threshold - 0.1, threshold - 0.05, threshold, threshold + 0.05, threshold + 0.1, threshold + 0.15, threshold + 0.2, threshold + 0.25]

# calculate the values of TPR and FPR
true_positives = []
false_positives = []
true_negatives = []
false_negatives = []
tprs = []
fprs = []

for threshold in thresholds:
    print(threshold)
    false_negative = 0
    true_positive = 0
    for i in range(min(len(good_real), len(good_generated))):
        real = convert_to_gray(good_real[i])
        generated = convert_to_gray(good_generated[i])
        differences = np.abs(real.flatten() - generated.flatten())
        if max(differences) > threshold:
            print("false negative")
            false_negative += 1
        else:
            print("true positive")
            true_positive += 1

    false_positive = 0
    true_negative = 0
    for i in range(min(len(defect_real), len(defect_generated))):
        real = convert_to_gray(defect_real[i])
        generated = convert_to_gray(defect_generated[i])
        differences = np.abs(real.flatten() - generated.flatten())
        if max(differences) > threshold:
            print("true negative")
            true_negative += 1
        else:
            print("false positive")
            false_positive += 1

    tpr = true_positive / (true_positive + false_negative)
    fpr = false_positive / (true_negative + false_positive)
    true_positives.append(true_positive)
    true_negatives.append(true_negative)
    false_positives.append(false_positive)
    false_negatives.append(false_negative)
    tprs.append(tpr)
    fprs.append(fpr)

# check the values of TPR and FPR for different thresholds
print(tprs)
print(fprs)

# now plot the ROC-curve
fig, ax = plt.subplots(figsize = (26, 13), dpi = 180, facecolor = 'w', edgecolor = 'k')

roc = ax.plot(fprs, tprs, marker = 'o', label = "Trained for 750 epochs DCGAN model")

d = ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1))
d[0].set_linestyle('dashed')

plt.text(0, 0, "Thresholds {:.3f} to {:.3f}".format(thresholds[0], thresholds[2]))
plt.text(0, 0.95, "Thresholds {:.3f} to {:.3f}".format(thresholds[3], thresholds[5]))
plt.text(0.55, 0.95, "Threshold {:.3f}".format(thresholds[6]))
plt.text(0.85, 0.95, "Threshold {:.3f}".format(thresholds[7]))


ax.set_ylabel("TPR")
ax.set_xlabel("FPR")
ax.set_title("ROC curve for thresholds from {} to {}.".format(round(min(thresholds), 3), round(max(thresholds), 3)))
ax.legend(loc = 'lower right')
plt.show()
