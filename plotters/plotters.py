from matplotlib import pyplot as plt
import os
from torchmetrics.classification import MulticlassConfusionMatrix



def plot_loss(train_loss, val_loss, path):
    plt.title('Training and validation loss')
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.savefig(os.path.join(path, 'loss.png'))
    plt.clf()
    plt.close()


def plot_acc(train_acc, val_acc, path):
    plt.title('Training and validation accuracy')
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='val')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.savefig(os.path.join(path, 'accuracy.png'))
    plt.clf()
    plt.close()


def plot_auroc(train_auroc, val_auroc, path):
    plt.title('Training and validation AUROC')
    plt.plot(train_auroc, label='train')
    plt.plot(val_auroc, label='val')
    plt.ylabel('AUROC')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.savefig(os.path.join(path, 'auroc.png'))
    plt.clf()
    plt.close()


def plot_confusion_matrix(preds, target, path):
    metric = MulticlassConfusionMatrix(num_classes=29)
    metric.update(preds, target)
    fig, ax = metric.plot()
    fig.savefig(os.path.join(path, 'confusion_matrix.png'))
    plt.clf()
    plt.close()
