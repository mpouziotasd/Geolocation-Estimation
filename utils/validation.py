import os
from utils.colors import colors 

def VALIDATE_PROJECT_FILES():
    if not os.listdir('models'):
        print(f"{colors.FAIL}Empty 'models' directory.\n{colors.WARNING}Please add YOLO model weights of your choice or run train_model.py to train a new model.{colors.ENDC}")
        exit(-1)
    
