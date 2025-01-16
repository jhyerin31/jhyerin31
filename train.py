import sys
import os
import datetime

import numpy as np
from sklearn import metrics

from src.data_loader import load_data, preprocess_data, split_data
from src.model import train_random_forest, evaluate_model
from sklearn.metrics import mean_squared_error
from src.utils import project_path
from src.model_save import save_model

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def main():
    # 1. 데이터 로드 및 전처리
    df = load_data(os.path.join(project_path(), "data", "imdb_top_1000.csv"))
    #print(df)

    df = preprocess_data(df)
    model_params = {'n_estimators': [10, 20, 30]}  # n_estimators 하이퍼파라미터
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S") # 저장 순서대로
    

    # 2. 훈련 데이터와 테스트 데이터 분할
    X_train, X_test, y_train, y_test = split_data(df, target_column='IMDB_Rating')

    # 3. 특성 스케일링
    #X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # 4. 모델 훈련
    for n_est in model_params['n_estimators']:
        model = train_random_forest(X_train, y_train, n_estimators=n_est)
        
        # 5. 모델 평가
        y_pred = evaluate_model(model, X_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        mse = mean_squared_error(y_test, y_pred)
        print(f'Root Mean Squared Error: {rmse}')
        print(f'Mean Squared Error: {mse}')

        save_path_template = os.path.join(project_path(),'models',f'n_{n_est}_t_{current_time}.pkl')
        save_path = save_path_template.format(n_estimators=n_est, current_time = current_time)
        save_model(model, save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()

