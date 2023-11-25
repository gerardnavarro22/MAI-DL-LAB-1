import os
import gc
import torch
import pandas as pd
from torchinfo import summary
import numpy as np
from torchviz import make_dot
from torchview import draw_graph

from models.FCNN import FCNN1Layers, FCNN2Layers, FCNN3Layers
from models.CNN import CNN1Conv, CNN2Conv, CNN3Conv, CNN3ConvNoBatchNorm, ComplexCNN

DATASET_PATH = fr'../datasets/raw/mame'
TRAINED_MODELS_PATH = r'../output/'
TESTED_MODELS_PATH = r'../output_test/'
SAVE_PATH = r'../output_test_plots/'
NUM_CLASSES = 29
BATCH_SIZE = 128

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


for tested_model, trained_model in zip(sorted(os.listdir(TESTED_MODELS_PATH)), sorted(os.listdir(TRAINED_MODELS_PATH))):
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
        BATCH_SIZE = 64
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

    print(f'Plotting tested model: {trained_model}')
    model_stats = summary(model, input_size=(BATCH_SIZE, 3, 256, 256), verbose=0)
    summary_str = str(model_stats)

    with open(os.path.join(path, 'summary.txt'), 'w+') as f:
        f.writelines(summary_str)

    y = model(torch.Tensor(np.random.rand(BATCH_SIZE, 3, 256, 256)).cuda())
    make_dot(y, params=dict(list(model.named_parameters()))).render(os.path.join(path, 'architecture1'), format="png")
    model_graph = draw_graph(model, input_size=(BATCH_SIZE, 3, 256, 256), device='cuda')
    model_graph.visual_graph.render(os.path.join(path, 'architecture2'), format="png")

    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    BATCH_SIZE = 128
