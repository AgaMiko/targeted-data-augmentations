import os
import shutil
import pandas as pd
from tqdm import tqdm

data_dir = "/media/agnieszka/Data/data/isic_data/"
mal = 116
ben = 6508
files_to_move = dict()
files_to_move['malignant'] = mal
files_to_move['benign'] = ben


df = pd.read_csv(os.path.join(data_dir, 'ISIC_2020_Training_GroundTruth.csv'))
for sample in files_to_move:
    df_temp = df.loc[df['benign_malignant'] == sample]
    df_part = df_temp.sample(n = files_to_move[sample])
    for path, category in tqdm(zip(df_part['image_name'], df_part['benign_malignant'])):
        source = os.path.join(data_dir, 'train', sample, path + '.jpg')
        dest = os.path.join(data_dir, 'test', category, path + '.jpg')
        if not os.path.exists(os.path.join(data_dir, 'test', category)):
            os.makedirs(os.path.join(data_dir, 'test', category))
        shutil.move(source, dest)