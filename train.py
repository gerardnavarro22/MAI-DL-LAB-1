import os
import json
import torch
import gc
from typing import Type
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils import validate, train_one_epoch
from MAMEDataset import MAMEDataset
from plotters.plotters import plot_auroc, plot_loss, plot_acc


def train(Net: Type[torch.nn.Module], epochs: int, batch_size: int, num_classes: int, out_path: str, p=0.2, p_conv=0.1):
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.7,max_split_size_mb:128"

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.53116883, 0.57740572, 0.6089572], [0.26368123, 0.2632309, 0.26533898])])

    train_dataset = MAMEDataset(fr'./datasets/processed/mame/train/labels.csv',
                                fr'./datasets/processed/mame/train', header=None, transform=transform)
    val_dataset = MAMEDataset(fr'./datasets/processed/mame/val/labels.csv',
                              fr'./datasets/processed/mame/val', header=None, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Net(num_classes, p, p_conv)
    model.cuda()

    out_path = os.path.join(out_path, model.__str__())

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=0.1, amsgrad=True)

    epoch_iters = int(train_dataset.__len__() / batch_size)

    best_loss = 999999999

    history = {'train_loss': [], 'train_acc': [], 'train_auroc': [], 'val_loss': [], 'val_acc': [], 'val_auroc': []}
    for epoch in range(epochs):

        train_loss, train_auroc, train_acc = train_one_epoch(epoch, epochs, epoch_iters, train_loader, criterion,
                                                             optimizer, model)

        mean_loss, val_auroc, val_acc = validate(val_loader, criterion, model)

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(),
                       os.path.join(out_path, 'best.pt'))

        msg = 'VALIDATION METRICS: Loss: {:.3f} AUROC: {:.3f} Accuracy: {:.3f}'.format(mean_loss, val_auroc, val_acc)
        print(msg)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_auroc'].append(train_auroc)
        history['val_loss'].append(mean_loss)
        history['val_acc'].append(val_acc)
        history['val_auroc'].append(val_auroc)

        if epoch > 0:
            plot_loss(history['train_loss'], history['val_loss'], out_path)
            plot_acc(history['train_acc'], history['val_acc'], out_path)
            plot_auroc(history['train_auroc'], history['val_auroc'], out_path)

        with open(os.path.join(out_path, 'history.json'), 'w') as outfile:
            json.dump(history, outfile, indent=4)

    torch.save(model.state_dict(), os.path.join(out_path, 'final_state.pt'))

    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
