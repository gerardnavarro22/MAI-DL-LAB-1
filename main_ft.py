import os
import gc
import torch
import pandas as pd
import json
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from MAMEDataset import MAMEDataset
from torchinfo import summary
from train import train
from utils import test, select_model
from plotters.plotters import plot_confusion_matrix, plot_classification_report

EPOCHS = 50
BATCH_SIZE = 64
NUM_CLASSES = 29
SAVE_PATH = [
    r'./output_ft/vgg11_no-freeze_lr-0.001',
    r'./output_ft/vgg11_freeze-conv_lr-0.001',
    r'./output_ft/vgg16_no-freeze_lr-0.001',
    r'./output_ft/vgg16_freeze-conv_lr-0.001',
    r'./output_ft/efficientnet-b4_no-freeze_lr-0.001',
    r'./output_ft/resnet18_no-freeze_lr-0.001',
    r'./output_ft/resnet18_freeze-conv_lr-0.001',
    r'./output_ft/resnet50_no-freeze_lr-0.001',
    r'./output_ft/resnet50_freeze-conv_lr-0.001',
    r'./output_ft/efficientnet-v2-s_freeze-conv_lr-0.001',
    r'./output_ft/efficientnet-v2-m_freeze-conv_lr-0.001',
    r'./output_ft/efficientnet-v2-l_freeze-conv_lr-0.001'
]
MODELS = ['vgg11', 'vgg11', 'vgg16', 'vgg16', 'efficientnet-b4', 'resnet18', 'resnet18', 'resnet50',
          'resnet50', 'efficientnet-v2-s', 'efficientnet-v2-m', 'efficientnet-v2-l']
FREEZES = [False, True, False, True, False, False, True, False, True, True, True, True]
DATASET_PATH = fr'./datasets/raw/mame'

labels_translation = pd.read_csv(os.path.join(DATASET_PATH, 'MAMe_labels.csv'), header=None)

num_to_labels_dict = {}
target_names = []
for _, label in labels_translation.iterrows():
    if label[1] == 'Woven fabric':
        label[1] = 'Woven fabric '
    target_names.append(label[1])
    num_to_labels_dict[label[0]] = label[1]

transform_test = transforms.Compose(
    [
        transforms.CenterCrop((int(256 * 0.875), int(256 * 0.875))),
        transforms.ToTensor(),
        transforms.Normalize([0.53116883, 0.57740572, 0.6089572], [0.26368123, 0.2632309, 0.26533898])
    ])

test_dataset = MAMEDataset(fr'./datasets/processed/mame/test/labels.csv',
                           fr'./datasets/processed/mame/test', header=None, transform=transform_test)

test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

for model_name, path, freeze in zip(MODELS, SAVE_PATH, FREEZES):

    model, BATCH_SIZE = select_model(model_name, freeze)

    if freeze:
        BATCH_SIZE = 128

    if not os.path.exists(path):
        os.makedirs(path)

    model_stats = summary(model, input_size=(BATCH_SIZE, 3, 256, 256), verbose=0)
    summary_str = str(model_stats)

    with open(os.path.join(path, 'summary.txt'), 'w+') as f:
        f.writelines(summary_str)

    train(model, EPOCHS, BATCH_SIZE, NUM_CLASSES, path, False, lr=0.001, eps=0.1)

    model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))
    model.cuda()

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

    BATCH_SIZE = 64
