import os
import torch
import gc
import json
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from MAMEDataset import MAMEDataset
from utils import test
from plotters.plotters import plot_confusion_matrix, plot_classification_report

from models.FCNN import FCNN1Layers, FCNN2Layers, FCNN3Layers
from models.CNN import CNN1Conv, CNN2Conv, CNN3Conv, CNN3ConvNoBatchNorm, ComplexCNN

DATASET_PATH = fr'../datasets/raw/mame'
TRAINED_MODELS_PATH = r'../output/'
SAVE_PATH = r'../output_test/'
NUM_CLASSES = 29
BATCH_SIZE = 256

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

labels_translation = pd.read_csv(os.path.join(DATASET_PATH, 'MAMe_labels.csv'), header=None)

num_to_labels_dict = {}
target_names = []
for _, label in labels_translation.iterrows():
    if label[1] == 'Woven fabric':
        label[1] = 'Woven fabric '
    target_names.append(label[1])
    num_to_labels_dict[label[0]] = label[1]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.53116883, 0.57740572, 0.6089572], [0.26368123, 0.2632309, 0.26533898])])

test_dataset = MAMEDataset(fr'./datasets/processed/mame/test/labels.csv',
                           fr'../datasets/processed/mame/test', header=None, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

for trained_model in sorted(os.listdir(TRAINED_MODELS_PATH)):
    if trained_model != 'ComplexCNN_drop0.2_dropconv0.2':
        continue
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
        p = float(trained_model.split('drop')[1][:-1])
        p_conv = float(trained_model.split('drop')[-1].split('conv')[-1])
        model = CNN1Conv(NUM_CLASSES, p, p_conv)
        model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))
    elif 'CNN2Conv'.lower() in trained_model.lower():
        p = float(trained_model.split('drop')[1][:-1])
        p_conv = float(trained_model.split('drop')[-1].split('conv')[-1])
        model = CNN2Conv(NUM_CLASSES, p, p_conv)
        model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))
    elif 'CNN3Conv'.lower() in trained_model.lower() and 'CNN3ConvNoBatchNorm'.lower() not in trained_model.lower():
        p = float(trained_model.split('drop')[1][:-1])
        p_conv = float(trained_model.split('drop')[-1].split('conv')[-1])
        model = CNN3Conv(NUM_CLASSES, p, p_conv)
        model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))
    elif 'CNN3ConvNoBatchNorm'.lower() in trained_model.lower():
        p = float(trained_model.split('drop')[1][:-1])
        p_conv = float(trained_model.split('drop')[-1].split('conv')[-1])
        model = CNN3ConvNoBatchNorm(NUM_CLASSES, p, p_conv)
        model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))
    elif 'ComplexCNN'.lower() in trained_model.lower():
        p = float(trained_model.split('drop')[1][:-1])
        p_conv = float(trained_model.split('drop')[-1].split('conv')[-1])
        model = ComplexCNN(NUM_CLASSES, p, p_conv)
        model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))
    else:
        print(f'{trained_model} NOT IMPLEMENTED')
        continue

    path = os.path.join(SAVE_PATH, trained_model)
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        continue

    print(f'Evaluating {trained_model}')

    model = model.cuda()

    accuracy, true_labels, pred_labels, auroc = test(test_loader, model)

    metrics = {'accuracy': accuracy}

    true_labels = [num_to_labels_dict[number] for number in true_labels]
    pred_labels = [num_to_labels_dict[number] for number in pred_labels]

    plot_confusion_matrix(pred_labels, true_labels, target_names, path)
    plot_classification_report(pred_labels, true_labels, target_names, path)

    metrics['auroc'] = auroc

    with open(os.path.join(path, 'metrics.json'), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)

    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
