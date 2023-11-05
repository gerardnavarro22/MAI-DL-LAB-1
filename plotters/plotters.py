from matplotlib import pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


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


def plot_confusion_matrix(preds, target, target_names, path):
    cf_matrix = confusion_matrix(np.array(preds), np.array(target), normalize='true')
    cf_matrix = np.around(cf_matrix, 2)
    fig, ax = plt.subplots(figsize=(13, 13))
    sns.heatmap(cf_matrix, annot=True, cmap='PuRd', cbar=False, square=True, xticklabels=target_names,
                yticklabels=target_names)
    fig.tight_layout()
    fig.savefig(os.path.join(path, 'confusion_matrix.png'), dpi=300)
    plt.clf()
    plt.close()
