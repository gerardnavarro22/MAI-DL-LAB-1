import numpy as np
import torch
import torch.nn as nn
import time
from torchmetrics.classification import MulticlassAUROC, MulticlassConfusionMatrix
from torchvision.models.vgg import *
from torchvision.models.resnet import *
from torchvision.models.efficientnet import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def test(test_loader, model: torch.nn.Module):
    model.eval()
    ave_acc = AverageMeter()
    true_labels = []
    pred_labels = []
    n_batches = len(test_loader)
    ave_auroc = AverageMeter()
    auroc_metric = MulticlassAUROC(num_classes=29, average="macro", thresholds=None)
    with torch.no_grad():
        for idx, batch in enumerate(test_loader, 0):
            images, labels = batch
            images = images.cuda()
            labels = labels.cuda()
            labels = labels.squeeze()

            outputs = model(images)
            current_pred_labels = torch.argmax(outputs, dim=1)
            acc = torch.mean((current_pred_labels == labels).float())
            pred_labels.extend(current_pred_labels.tolist())
            true_labels.extend(labels.tolist())

            if idx == 0:
                all_outputs = outputs
                all_labels = labels
            else:
                all_outputs = torch.cat((all_outputs, outputs))
                all_labels = torch.cat((all_labels, labels))

            ave_acc.update(acc.item())
            if idx % 5 == 0:
                msg = 'Iter:[{}/{}], Acc:{:.6f}'.format(idx, n_batches, ave_acc.average())
                print(msg)
    auroc = auroc_metric(all_outputs, all_labels)
    ave_auroc.update(auroc.item())
    return ave_acc.average(), true_labels, pred_labels, ave_auroc.average()


def validate(val_loader, loss_fn, model):
    model.eval()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    with torch.no_grad():
        for idx, batch in enumerate(val_loader, 0):
            images, labels = batch
            images = images.cuda()
            labels = labels.cuda()
            labels = labels.squeeze()

            outputs = model(images)
            pred_labels = torch.argmax(outputs, dim=1)
            losses = loss_fn(outputs, labels)
            loss = losses.mean()
            ave_loss.update(loss.item())
            if idx == 0:
                all_outputs = outputs
                all_labels = labels
            else:
                all_outputs = torch.cat((all_outputs, outputs))
                all_labels = torch.cat((all_labels, labels))

            acc = torch.mean((pred_labels == labels).float())
            ave_acc.update(acc.item())

    return ave_loss.average(), ave_acc.average()


def train_one_epoch(epoch, num_epoch, epoch_iters, train_loader, loss_fn, optimizer, model):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    tic = time.time()

    for i_iter, batch in enumerate(train_loader, 0):
        images, labels = batch
        images = images.cuda()
        labels = labels.cuda()
        labels = labels.squeeze()

        optimizer.zero_grad()

        outputs = model(images)
        pred_labels = torch.argmax(outputs, dim=1)
        losses = loss_fn(outputs, labels)
        loss = losses.mean()
        acc = torch.mean((pred_labels == labels).float())

        losses.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())

        if i_iter % 5 == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}'.format(epoch + 1, num_epoch, i_iter, epoch_iters,
                                                            batch_time.value(),
                                                            [x['lr'] for x in optimizer.param_groups],
                                                            ave_loss.average(),
                                                            ave_acc.average())
            print(msg)

    return ave_loss.average(), ave_acc.average()


def select_model(model_name, freeze, num_classes=29):
    batch_size = 64
    if 'efficientnet' in model_name and 'efficientnet-v2' not in model_name:
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        if 'efficientnet-b4' in model_name:
            batch_size = 32
    elif 'vgg11' in model_name:
        model = vgg11_bn(weights='IMAGENET1K_V1')
        model.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes)
    elif 'vgg16' in model_name:
        model = vgg16_bn(weights='IMAGENET1K_V1')
        model.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes)
    elif 'resnet18' in model_name:
        batch_size = 128
        model = resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    elif 'resnet50' in model_name:
        batch_size = 128
        model = resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    elif 'efficientnet-v2-s' in model_name:
        model = efficientnet_v2_s(weights='IMAGENET1K_V1')
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes),
        )
    elif 'efficientnet-v2-m' in model_name:
        batch_size = 32
        model = efficientnet_v2_m(weights='IMAGENET1K_V1')
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes),
        )
    elif 'efficientnet-v2-l' in model_name:
        batch_size = 16
        model = efficientnet_v2_l(weights='IMAGENET1K_V1')
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes),
        )
    else:
        raise NotImplementedError

    if 'efficientnet' in model_name and freeze and 'efficientnet-v2' not in model_name:
        for param in model._conv_stem.parameters():
            param.requires_grad_(False)
        for param in model._bn0.parameters():
            param.requires_grad_(False)
        for param in model._blocks.parameters():
            param.requires_grad_(False)
    elif 'vgg' in model_name and freeze:
        for param in model.features.parameters():
            param.requires_grad_(False)
    elif 'resnet' in model_name and freeze:
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.fc.parameters():
            param.requires_grad_(True)
    elif 'efficientnet-v2' in model_name and freeze:
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.classifier.parameters():
            param.requires_grad_(True)

    return model, batch_size


def extract_features_resnet(model: ResNet, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
    model.fc = torch.nn.Sequential()
    model.cuda()
    model.eval()
    n_batches = len(train_loader) + len(val_loader) + len(test_loader)
    with torch.no_grad():
        for idx, batch in enumerate(train_loader, 0):
            images, labels = batch
            images = images.cuda()
            labels = labels.squeeze().cpu().numpy()
            outputs = model(images).cpu().numpy()

            if idx == 0:
                x_train = np.array(outputs)
                y_train = np.array(labels)
            else:
                x_train = np.vstack((x_train, outputs))
                y_train = np.append(y_train, labels)

            if idx % 1 == 0:
                msg = 'Iter:[{}/{}]'.format(idx, n_batches)
                print(msg)

        torch.cuda.empty_cache()
        for idx, batch in enumerate(val_loader, 0):
            images, labels = batch
            images = images.cuda()
            labels = labels.squeeze().cpu().numpy()
            outputs = model(images).cpu().numpy()

            if idx == 0:
                x_val = np.array(outputs)
                y_val = np.array(labels)
            else:
                x_val = np.vstack((x_val, outputs))
                y_val = np.append(y_val, labels)

            if idx % 1 == 0:
                msg = 'Iter:[{}/{}]'.format(idx+len(train_loader), n_batches)
                print(msg)

        torch.cuda.empty_cache()
        for idx, batch in enumerate(test_loader, 0):
            images, labels = batch
            images = images.cuda()
            labels = labels.squeeze().cpu().numpy()
            outputs = model(images).cpu().numpy()

            if idx == 0:
                x_test = np.array(outputs)
                y_test = np.array(labels)
            else:
                x_test = np.vstack((x_test, outputs))
                y_test = np.append(y_test, labels)

            if idx % 1 == 0:
                msg = 'Iter:[{}/{}]'.format(idx+len(train_loader)+len(val_loader), n_batches)
                print(msg)

    return x_train, y_train, x_val, y_val, x_test, y_test
