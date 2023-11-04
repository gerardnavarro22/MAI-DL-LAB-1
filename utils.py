import numpy as np
import torch
import time


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


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
        label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                i_pred] = label_count[cur_index]
    return confusion_matrix


def validate(testloader, loss_fn, model):
    model.eval()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, labels = batch
            image = image.cuda()
            labels = labels.cuda()
            labels = labels.squeeze()

            outputs = model(image)
            pred_labels = torch.argmax(outputs, dim=1)
            losses = loss_fn(outputs, labels)
            loss = losses.mean()
            ave_loss.update(loss.item())

            acc = torch.mean((pred_labels == labels).float())
            ave_acc.update(acc.item())

    return ave_loss.average(), ave_acc.average()


def train(epoch, num_epoch, epoch_iters, train_loader, loss_fn, optimizer, model):
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
        if epoch == 7:
            pass

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
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}'.format(epoch+1, num_epoch, i_iter, epoch_iters,
                                                            batch_time.average(),
                                                            [x['lr'] for x in optimizer.param_groups],
                                                            ave_loss.average(), ave_acc.average())
            print(msg)

