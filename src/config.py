import os

###########################
# Path Configurations
###########################
BASE_DIR = os.path.join('..', 'data')
SAVE_DIR = os.path.join('..', 'result')

INPUT_PATH = os.path.join(BASE_DIR, 'sales_sell_out.csv')

###########################
# Data Configurations
###########################
FILTER_MINUS_YN = False    # Remove minus values or not

COL_TARGET = 'amt'

# window configuration
INPUT_WIDTH = 28
LABEL_WIDTH = 7
SHIFT = 7

# Outlier handling configuration
SMOOTH_YN = True    # Smoothing or not (True / False)
SMOOTH_METHOD = 'quantile'    # Methods: quantile / sigma
SMOOTH_RATE = 0.05    # Quantile rate

###########################
# Model Hyper-parameters
###########################
N_HIDDEN = 64
BATCH_SIZE = 16
EPOCHS = 50
