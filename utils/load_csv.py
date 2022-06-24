import os
import shutil
import pandas as pd
from tqdm import tqdm

data_dir = "/media/agnieszka/Data/data/skin-lesion/kaggle/archive/train"

df = pd.read_csv(os.path.join(data_dir, 'real_val.csv'))
print(df.head())
for path, category in tqdm(zip(df['image_name'], df['target'])):
    category = "val/benign" if category == 0 else "val/malignant"
    path = path.split("/")[-1]
    source = os.path.join(data_dir, 'train', path)
    dest = os.path.join(data_dir, category, path)
    if not os.path.exists(os.path.join(data_dir, category)):
        os.makedirs(os.path.join(data_dir, category))
    shutil.move(source, dest)