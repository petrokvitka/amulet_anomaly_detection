# deactivate all conda environments,
# use the default base conda environment
from skimage import io, color
import pandas as pd
import numpy as np

def convert_to_gray(img_path):
    color_img = io.imread(img_path)
    color_gray = color.rgb2gray(color_img)
    return color_gray

train_real_grey = convert_to_gray("../../mfccs/img/1200_200_0.jpg")
train_generated_grey = convert_to_gray("./dcgan_mfccs_output/samples_745_0.png")

# check the shape of grey images
print(train_generated_grey.shape) #(20, 44)

differences = np.abs(train_real_grey.flatten() - train_generated_grey.flatten())
threshold = np.max(differences)
print(threshold) #0.17


# now test this threshold with the test data
bad_real_color = io.imread("../../mfccs/test/bad_22050.jpg")
bad_real_grey = color.rgb2gray(bad_real_color)

defect_real_color = io.imread("../../mfccs/test/defect_22050.jpg")
defect_real_grey = color.rgb2gray(defect_real_color)

good_real_color = io.imread("../../mfccs/test/good_22050.jpg")
good_real_grey = color.rgb2gray(good_real_color)



bad_generated_grey = convert_to_gray("./dcgan_mfccs_output/test/bad/samples_0_1.jpg")
good_generated_grey = convert_to_gray("./dcgan_mfccs_output/test/good/samples_0_1.jpg")
defect_generated_grey = convert_to_gray("./dcgan_mfccs_output/test/defect/samples_0_1.jpg")


bad_differences = np.abs(bad_real_grey.flatten() - bad_generated_grey.flatten())
good_differences = np.abs(good_real_grey.flatten() - good_generated_grey.flatten())
defect_differences = np.abs(defect_real_grey.flatten() - defect_generated_grey.flatten())

"""
if max(bad_differences) > threshold:
    #print("defect")
    print("true negative")
else:
    print("false positive")
"""

if max(good_differences) > threshold:
    print("true positive")
else:
    print("false negative")

if max(defect_differences) > threshold:
    print("true negative")
else:
    print("false positive")
