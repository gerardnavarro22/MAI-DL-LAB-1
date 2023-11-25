import os
import gc
import torch
import json
from sklearn import svm
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from MAMEDataset import MAMEDataset
from utils import select_model, extract_features_resnet
from plotters.plotters import plot_confusion_matrix, plot_classification_report

EPOCHS = 50
BATCH_SIZE = 1024
NUM_CLASSES = 29
SAVE_PATH = [
    r'./output_fe/resnet18',
]
MODELS = ['resnet18']
DATASET_PATH = fr'./datasets/raw/mame'

labels_translation = pd.read_csv(os.path.join(DATASET_PATH, 'MAMe_labels.csv'), header=None)

num_to_labels_dict = {}
target_names = []
for _, label in labels_translation.iterrows():
    if label[1] == 'Woven fabric':
        label[1] = 'Woven fabric '
    target_names.append(label[1])
    num_to_labels_dict[label[0]] = label[1]

transforms = transforms.Compose(
    [
        transforms.CenterCrop((int(256 * 0.875), int(256 * 0.875))),
        transforms.ToTensor(),
        transforms.Normalize([0.53116883, 0.57740572, 0.6089572], [0.26368123, 0.2632309, 0.26533898])
    ])

train_dataset = MAMEDataset(fr'./datasets/processed/mame/train/labels.csv',
                            fr'./datasets/processed/mame/train', header=None, transform=transforms)
val_dataset = MAMEDataset(fr'./datasets/processed/mame/val/labels.csv',
                          fr'./datasets/processed/mame/val', header=None, transform=transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = MAMEDataset(fr'./datasets/processed/mame/test/labels.csv',
                           fr'./datasets/processed/mame/test', header=None, transform=transforms)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

for model_name, path in zip(MODELS, SAVE_PATH):
    if not os.path.exists(path):
        os.makedirs(path)

    model, BATCH_SIZE = select_model(model_name, False)

    x_train, y_train, x_val, y_val, x_test, y_test = extract_features_resnet(
        model, train_loader, val_loader, test_loader
    )

    clf = svm.LinearSVC()
    clf.fit(X=x_train, y=y_train)
    y_pred = clf.predict(x_test)
    accuracy = (y_pred == y_test).sum() / len(y_test)

    metrics = {'accuracy': accuracy}

    true_labels = [num_to_labels_dict[number] for number in y_test]
    pred_labels = [num_to_labels_dict[number] for number in y_pred]

    plot_confusion_matrix(pred_labels, true_labels, target_names, path)
    plot_classification_report(pred_labels, true_labels, target_names, path)

    with open(os.path.join(path, 'metrics.json'), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)

    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
