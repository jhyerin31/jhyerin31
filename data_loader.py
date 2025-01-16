import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

    
def load_data(file_path):
    """CSV 파일을 로드하는 함수"""
    df = pd.read_csv(file_path)
    columns = ['Released_Year', 'Certificate','Runtime', 'IMDB_Rating', 'Meta_score','No_of_Votes', 'Gross']
    df = df[columns]

    return df
    
def preprocess_data(df):
    """데이터 전처리 함수"""
    if df is None:
        print("No data to preprocess.")
        return None
    
    # 결측값 제거
    df = df.dropna()
    df = df.reset_index(drop = True)
    df = df[df['Released_Year'] != 'PG']

    df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')
    df['Runtime'] = df['Runtime'].str.replace(' min', '').astype(int)
    df['Gross'] = df['Gross'].str.replace(',', '').astype(int)
    
    # Certificate 범주 재 정의
    df['Certificate'] = df['Certificate'].replace(['U', 'G', 'Passed', 'Approved', 'GP'], 'G')
    df['Certificate'] = df['Certificate'].replace('U/A', 'UA')
    df['Certificate'] = df['Certificate'].replace('TV-PG', 'PG')

    categorical_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    # Label encoding for categorical columns
    label_encoders = {}
    for col in categorical_columns:
        lbl = LabelEncoder()
        df[col] = lbl.fit_transform(df[col])
        label_encoders[col] = lbl
    
    print("Data preprocessed.")
    return df

def split_data(df, target_column, test_size=0.2):
    """훈련 데이터와 테스트 데이터로 분할"""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test