# Importing the libraries
import os
import random
from shutil import copyfile
from os import listdir, makedirs
from os.path import isfile, join, exists
import re

# Define the path to the data
folder = 'Converted_Project_Data'
source_dir = 'Dataset/'

training_dir = source_dir + 'Training/'
validation_dir = source_dir + 'Validation/'
test_dir = source_dir + 'Test/'

# Define the split ratios
train_ratio = 0.7
val_ratio = 0.15
# test_ratio is the remaining part

onlyfiles = [f for f in listdir(source_dir + folder) if isfile(join(source_dir + folder, f))]

# Shuffle the list of filenames
random.shuffle(onlyfiles)

# Calculate the size of each subset
train_size = int(len(onlyfiles) * train_ratio)
val_size = int(len(onlyfiles) * val_ratio)

# Split the filenames
train_files = onlyfiles[:train_size]
val_files = onlyfiles[train_size:train_size + val_size]
test_files = onlyfiles[train_size + val_size:]

# Create directories if they don't exist
if not exists(training_dir):
    makedirs(training_dir)
if not exists(validation_dir):
    makedirs(validation_dir)
if not exists(test_dir):
    makedirs(test_dir)

for filename_num, filename in enumerate(train_files):
    copyfile(source_dir + folder + '/' + filename, training_dir + str(folder.replace('/', '_')) + str(filename_num).zfill(4) + '.jpg')

for filename_num, filename in enumerate(val_files):
    copyfile(source_dir + folder + '/' + filename, validation_dir + str(folder.replace('/', '_')) + str(filename_num).zfill(4) + '.jpg')

for filename_num, filename in enumerate(test_files):
    copyfile(source_dir + folder + '/' + filename, test_dir + str(folder.replace('/', '_')) + str(filename_num).zfill(4) + '.jpg')

print('Done')