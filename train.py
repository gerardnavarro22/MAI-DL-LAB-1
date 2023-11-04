import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils import validate, train
from MAMEDataset import MAMEDataset

from models.CNN import Net

EPOCHS = 50
BATCH_SIZE = 128
NUM_CLASSES = 29
OUT_PATH = r'./output/'

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
for epoch in range(EPOCHS):

    train(epoch, EPOCHS, epoch_iters, train_loader, criterion, optimizer, model)

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

torch.save(model.state_dict(), os.path.join(OUT_PATH, 'final_state.pt'))
