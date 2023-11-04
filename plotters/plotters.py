from matplotlib import pyplot as plt
import os


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
