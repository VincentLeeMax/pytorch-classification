import glob
import os
from shutil import move, rmtree

base_path = '/data/dataset/1/tiny-imagenet-200'
target_folder = base_path + '/val/'
test_folder   = base_path + '/test/'

if os.path.exists(test_folder):
    rmtree(test_folder)
val_dict = {}
with open(base_path + '/val/val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]

paths = glob.glob(base_path + '/val/images/*')
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    if not os.path.exists(target_folder + str(folder)):
        os.makedirs(target_folder + str(folder))
        os.makedirs(target_folder + str(folder) + '/images')
    if not os.path.exists(test_folder + str(folder)):
        os.makedirs(test_folder + str(folder))
        os.makedirs(test_folder + str(folder) + '/images')

for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    if len(glob.glob(target_folder + str(folder) + '/images/*')) < 25:
        dest = target_folder + str(folder) + '/images/' + str(file)
    else:
        dest = test_folder + str(folder) + '/images/' + str(file)
    move(path, dest)

rmtree(base_path + '/val/images')