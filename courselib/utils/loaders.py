import os
import pandas as pd

def load_or_download_csv(file_name, url, column_names=None, encoding='utf-8'):
    if os.path.exists(file_name):
        print(f"Loading from local `{file_name}`...")
        return pd.read_csv(file_name, index_col=0, encoding=encoding)
    else:
        print(f"Downloading from `{url}`...")
        df = pd.read_csv(url, names=column_names, encoding=encoding)
        df.to_csv(file_name, encoding=encoding)
        print("Saved to local file.")
        return df
    

def load_irises():
    file_name = 'iris_data_set.csv'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    column_names = ['sepal length [cm]', 'sepal width [cm]',
                'petal length [cm]', 'petal width [cm]', 'iris type']
    
    df = load_or_download_csv(file_name, url, column_names)

    return df

def load_digits():
    file_name = 'letter-recognition.data'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
    column_names = ['letter','x-box','y-box','width','high','onpix','x-bar','y-bar','x2bar','y2bar','xybar',
                'x2ybr','xy2br','x-ege','xegvy','y-ege','yegvx']


    df = load_or_download_csv(file_name, url, column_names)
    return df 

def load_wine():
    file_name = 'wine.csv'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
    column_names = ['Class','Alcohol', 'Malic acid','Ash', 'Alcalinity of ash', 'Magnesium',
               'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
                'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

    df = load_or_download_csv(file_name, url, column_names)
    return df 

