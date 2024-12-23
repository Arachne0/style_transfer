import os
import shutil

# source_dir = "/media/hail/HDD/style_transfer/results/x-ray_cyclegan/test_latest/images/"
source_dir = "/results/maps_cyclegan_vanilla/test_latest/images/"

real_dir = os.path.join(source_dir, "real")
fake_dir = os.path.join(source_dir, "fake")

os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

for filename in os.listdir(source_dir):
    if filename.endswith(".png"):
        if "_real" in filename:
            shutil.move(os.path.join(source_dir, filename), os.path.join(real_dir, filename))
        elif "_fake" in filename:
            shutil.move(os.path.join(source_dir, filename), os.path.join(fake_dir, filename))

