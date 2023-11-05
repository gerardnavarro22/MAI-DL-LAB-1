import numpy as np
import torch
import time
from torchmetrics.classification import MulticlassAUROC, MulticlassConfusionMatrix


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

            ave_acc.update(acc.item())
            if idx % 5 == 0:
                msg = 'Iter:[{}/{}], Acc:{:.6f}'.format(idx, n_batches, ave_acc.average())
                print(msg)

    return ave_acc.average(), true_labels, pred_labels


def validate(val_loader, loss_fn, model):
    model.eval()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    ave_auroc = AverageMeter()
    auroc_metric = MulticlassAUROC(num_classes=29, average="macro", thresholds=None)
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

    auroc = auroc_metric(all_outputs, all_labels)
    ave_auroc.update(auroc.item())
    return ave_loss.average(), ave_auroc.average(), ave_acc.average()


def train_one_epoch(epoch, num_epoch, epoch_iters, train_loader, loss_fn, optimizer, model):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    ave_auroc = AverageMeter()
    auroc_metric = MulticlassAUROC(num_classes=29, average="macro", thresholds=None)
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

        auroc = auroc_metric(outputs, labels)

        losses.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        ave_auroc.update(auroc.item())

        if i_iter % 5 == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, AUROC: {:.6f}, Acc:{:.6f}'.format(epoch + 1, num_epoch, i_iter, epoch_iters,
                                                                           batch_time.average(),
                                                                           [x['lr'] for x in optimizer.param_groups],
                                                                           ave_loss.average(), ave_auroc.average(),
                                                                           ave_acc.average())
            print(msg)

    return ave_loss.average(), ave_auroc.average(), ave_acc.average()
