import pickle
import os
import datetime

import pickle

def save_model(model, file_path):
    """
    모델 저장
    
    Parameters:
        - model: 저장할 모델 객체
        - file_path (str): 저장할 파일 경로 (.pkl)
    """
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)