import os
import json
import torch
import gc
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils import validate, train
from MAMEDataset import MAMEDataset
from plotters.plotters import *

from models.CNN import Net

gc.collect()
torch.cuda.empty_cache()

EPOCHS = 50
BATCH_SIZE = 128
NUM_CLASSES = 29
OUT_PATH = r'./output/'
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.7,max_split_size_mb:128"

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.53116883, 0.57740572, 0.6089572], [0.26368123, 0.2632309, 0.26533898])])

target_transform = transforms.Compose(
    [transforms.ToTensor()])

train_dataset = MAMEDataset(fr'./datasets/processed/mame/train/labels.csv',
                            fr'./datasets/processed/mame/train', header=None, transform=transform)
test_dataset = MAMEDataset(fr'./datasets/processed/mame/test/labels.csv',
                           fr'./datasets/processed/mame/test', header=None, transform=transform)
val_dataset = MAMEDataset(fr'./datasets/processed/mame/val/labels.csv',
                          fr'./datasets/processed/mame/val', header=None, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = Net(NUM_CLASSES)
model.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=0.1, amsgrad=True)

epoch_iters = int(train_dataset.__len__() / BATCH_SIZE)

best_loss = 999999999

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
for epoch in range(EPOCHS):

    train_loss, train_acc = train(epoch, EPOCHS, epoch_iters, train_loader, criterion, optimizer, model)

    mean_loss, val_acc = validate(val_loader, criterion, model)

    if mean_loss < best_loss:
        print('=> saving checkpoint to {}'.format(
            OUT_PATH + 'checkpoint.pth.tar'))
        torch.save({
            'epoch': epoch + 1,
            'best_mIoU': best_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(OUT_PATH, 'checkpoint.pth.tar'))

        best_loss = mean_loss
        torch.save(model.state_dict(),
                   os.path.join(OUT_PATH, 'best.pt'))

    msg = 'VALIDATION METRICS: Loss: {:.3f} Accuracy: {:.3f}'.format(mean_loss, val_acc)
    print(msg)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(mean_loss)
    history['val_acc'].append(val_acc)

    if epoch > 0:
        plot_loss(history['train_loss'], history['val_loss'], OUT_PATH)
        plot_acc(history['train_acc'], history['val_acc'], OUT_PATH)

    with open(os.path.join(OUT_PATH, 'history.json'), 'w') as outfile:
        json.dump(history, outfile, indent=4)


torch.save(model.state_dict(), os.path.join(OUT_PATH, 'final_state.pt'))

