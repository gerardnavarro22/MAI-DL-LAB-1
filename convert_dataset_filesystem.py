import os
import pandas as pd
import shutil

DATASET_PATH = fr'./datasets/raw/mame/'
OUT_PATH = fr'./datasets/processed/mame/'

if __name__ == '__main__':
    labels = pd.read_csv(os.path.join(DATASET_PATH, 'MAMe_dataset.csv'), header=0, index_col=0)
    labels_translation = pd.read_csv(os.path.join(DATASET_PATH, 'MAMe_labels.csv'), header=None)
    
    labels_to_num_dict = {}
    for _, label in labels_translation.iterrows():
        if label[1] == 'Woven fabric':
            label[1] = 'Woven fabric '
        labels_to_num_dict[label[1]] = label[0]
    
    train_idx = []
    train_labels = []
    test_idx = []
    test_labels = []
    val_idx = []
    val_labels = []
    for index, row in labels.iterrows():
        img_subset = row['Subset']
        if img_subset == 'train':
            shutil.move(os.path.join(DATASET_PATH, 'data_256', index), os.path.join(OUT_PATH, 'train/', index))
            train_idx.append(index)
            train_labels.append(labels_to_num_dict[row['Medium']])
        elif img_subset == 'test':
            shutil.move(os.path.join(DATASET_PATH, 'data_256', index), os.path.join(OUT_PATH, 'test/', index))
            test_idx.append(index)
            test_labels.append(labels_to_num_dict[row['Medium']])
        elif img_subset == 'val':
            shutil.move(os.path.join(DATASET_PATH, 'data_256', index), os.path.join(OUT_PATH, 'val/', index))
            val_idx.append(index)
            val_labels.append(labels_to_num_dict[row['Medium']])
        else:
            continue
        
    d = {'col1': train_idx, 'col2': train_labels}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(OUT_PATH, 'train/', 'labels.csv'), index=False, header=False)
    
    d = {'col1': test_idx, 'col2': test_labels}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(OUT_PATH, 'test/', 'labels.csv'), index=False, header=False)
    
    d = {'col1': val_idx, 'col2': val_labels}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(OUT_PATH, 'val/', 'labels.csv'), index=False, header=False)