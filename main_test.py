import os
import torch
import gc
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from MAMEDataset import MAMEDataset
from utils import test
from plotters.plotters import plot_confusion_matrix

from models.FCNN import FCNN1Layers, FCNN2Layers, FCNN3Layers
from models.CNN import CNN1Conv, CNN2Conv, CNN3Conv

DATASET_PATH = fr'datasets/raw/mame'
TRAINED_MODELS_PATH = r'./output/'
SAVE_PATH = r'./output_test/'
NUM_CLASSES = 29

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

labels_translation = pd.read_csv(os.path.join(DATASET_PATH, 'MAMe_labels.csv'), header=None)

num_to_labels_dict = {}
for _, label in labels_translation.iterrows():
    if label[1] == 'Woven fabric':
        label[1] = 'Woven fabric '
    num_to_labels_dict[label[0]] = label[1]

for trained_model in os.listdir(TRAINED_MODELS_PATH):
    path = os.path.join(TRAINED_MODELS_PATH, trained_model)
    if 'FCNN1Layers'.lower() in trained_model.lower():
        model = FCNN1Layers(NUM_CLASSES)
        model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))
    elif 'FCNN2Layers'.lower() in trained_model.lower():
        p = float(trained_model.split('drop')[-1])
        model = FCNN2Layers(NUM_CLASSES, p)
        model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))
    elif 'FCNN3Layers'.lower() in trained_model.lower():
        p = float(trained_model.split('drop')[-1])
        model = FCNN3Layers(NUM_CLASSES, p)
        model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))
    elif 'CNN1Conv'.lower() in trained_model.lower():
        p = float(trained_model.split('drop')[-1].split('_dropconv')[0])
        p_conv = float(trained_model.split('drop')[-1].split('_dropconv')[-1])
        model = CNN1Conv(NUM_CLASSES, p, p_conv)
        model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))
    elif 'CNN2Conv'.lower() in trained_model.lower():
        p = float(trained_model.split('drop')[-1].split('_dropconv')[0])
        p_conv = float(trained_model.split('drop')[-1].split('_dropconv')[-1])
        model = CNN2Conv(NUM_CLASSES, p, p_conv)
        model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))
    elif 'CNN3Conv'.lower() in trained_model.lower():
        p = float(trained_model.split('drop')[-1].split('_dropconv')[0])
        p_conv = float(trained_model.split('drop')[-1].split('_dropconv')[-1])
        model = CNN3Conv(NUM_CLASSES, p, p_conv)
        model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))
    else:
        print(f'{trained_model} NOT IMPLEMENTED')
        continue

    gc.collect()
    torch.cuda.empty_cache()
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.7,max_split_size_mb:128"

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.53116883, 0.57740572, 0.6089572], [0.26368123, 0.2632309, 0.26533898])])

    test_dataset = MAMEDataset(fr'./datasets/processed/mame/test/labels.csv',
                               fr'./datasets/processed/mame/test', header=None, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    accuracy, true_labels, pred_labels = test(test_loader, model)

    true_labels = [num_to_labels_dict[number] for number in true_labels]
    pred_labels = [num_to_labels_dict[number] for number in pred_labels]

    plot_confusion_matrix(pred_labels, true_labels, SAVE_PATH)
