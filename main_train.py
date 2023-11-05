from train import train
from models.FCNN import *
from models.CNN import *
import warnings
warnings.filterwarnings('ignore')

models = [CNN2Conv, CNN3Conv]

EPOCHS = 20
BATCH_SIZE = 128
NUM_CLASSES = 29

for model in models:
    train(model, EPOCHS, BATCH_SIZE, NUM_CLASSES, r'./output/')
