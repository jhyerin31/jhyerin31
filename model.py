from sklearn.ensemble import RandomForestRegressor

    
def train_random_forest(X_train, y_train, n_estimators, random_state=42):
    """랜덤 포레스트 모델 훈련"""
    
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test):
    """모델 평가"""
    y_pred = model.predict(X_test)
    return y_pred