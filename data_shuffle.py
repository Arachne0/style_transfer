import os
import shutil
from tqdm import tqdm

source_path = "/media/hail/HDD/style_transfer/datasets/x-ray/"
trainA_path = os.path.join(source_path, "trainA")
trainB_path = os.path.join(source_path, "trainB")
testA_path = os.path.join(source_path, "testA")
testB_path = os.path.join(source_path, "testB")

train_path = os.path.join(source_path, "train")
test_path = os.path.join(source_path, "test")

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

def copy_files(src_folder, dest_folder):
    for file_name in tqdm(os.listdir(src_folder), desc=f"Copying files from {src_folder}"):
        src_file = os.path.join(src_folder, file_name)
        dest_file = os.path.join(dest_folder, file_name)
        if os.path.isfile(src_file):
            shutil.copy(src_file, dest_file)

copy_files(trainA_path, train_path)
copy_files(trainB_path, train_path)

copy_files(testA_path, test_path)
copy_files(testB_path, test_path)

print("Files have been successfully copied and combined into 'train' and 'test' folders!")
