from train import train
from models.FCNN import *
from models.CNN import *
import warnings

warnings.filterwarnings('ignore')

models = [CNN3Conv, CNN3ConvNoBatchNorm, ComplexCNN]
p_list = [0.5]
p_conv_list = [0.1, 0.2, 0.5]

EPOCHS = 20
BATCH_SIZE = 128
NUM_CLASSES = 29

for model in models:
    if str(model) == "<class 'models.FCNN.FCNN1Layers'>" or str(model) == "<class 'models.FCNN.FCNN2Layers'>" or str(model) == "<class 'models.CNN.CNN1Conv'>" or str(model) == "<class 'models.CNN.CNN2Conv'>":
        train(model, EPOCHS, BATCH_SIZE, NUM_CLASSES, r'../output/')
    elif str(model) == "<class 'models.FCNN.FCNN3Layers'>":
        for p in p_list:
            train(model, EPOCHS, BATCH_SIZE, NUM_CLASSES, r'../output/', p)
    elif str(model) == "<class 'models.CNN.CNN3Conv'>" or str(model) == "<class 'models.CNN.CNN3ConvNoBatchNorm'>":
        for p in p_list:
            for p_conv in p_conv_list:
                train(model, EPOCHS, BATCH_SIZE, NUM_CLASSES, r'../output/', p, p_conv)
    elif str(model) == "<class 'models.CNN.ComplexCNN'>":
        p_list = [0.2, 0.5]
        p_conv_list = [0.1, 0.2]
        for p in p_list:
            for p_conv in p_conv_list:
                train(model, EPOCHS, 64, NUM_CLASSES, r'../output/', p, p_conv)
    else:
        print(f'{str(model)} not implemented')
        continue
