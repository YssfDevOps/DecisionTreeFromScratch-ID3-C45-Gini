__authors__ = ['1638618, 1636517, 1633311']
__group__ = 'GM08:30_3'

from copy import copy, deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import make_scorer, f1_score


# Funció per eliminar files amb valors buits.
def remove_missing_values(df):
    return df.dropna()


# Funció per discretitzar valors contínues.
def discretize(df, columns):
    for column in columns:
        df[column] = pd.qcut(df[column], q=4, labels=False)
    return df



# Funció per visualitzar l'arbre
def print_tree(node, depth=0):
    if node.is_leaf_node():
        print(f"{depth * '  '}[{node.value}]")
        return

    print(f"{depth * '  '}[{node.feature} <= {node.threshold}]")
    print_tree(node.left, depth + 1)
    print_tree(node.right, depth + 1)


# Funció principal
def main():
    # Carregar les dades
    df = pd.read_csv('train.csv')
    nan_count = df.isnull().sum()
    nan_percentage = df.isnull().mean() * 100
    print("Número de NaNs por columna:")
    print(nan_count)
    print("\nPorcentaje de NaNs por columna:")
    print(nan_percentage)



    # Tractament de valors buits APARTAT C.
    df = remove_missing_values(df)

    # Tractament de valors buits APARTAT B.
    imputer = SimpleImputer(strategy='mean')
    # Ajustar y transformar los datos.
    df['CUALQUIER ATRIBUTO'] = imputer.fit_transform(df['CUALQUIER ATRIBUTO'].values.reshape(-1, 1))


    # Discretització de valors contínues
    ''' Nota: Llista de columnes amb valors continus que s'haurien de discretitzar
    battery_power
    fc (Front Camera Megapixels)
    int_memory (Internal Memory in GB)
    mobile_wt (Mobile Weight in grams)
    pc (Primary Camera Megapixels)
    px_height (Pixel Height)
    px_width (Pixel Width)
    ram (RAM capacity in MB)
    sc_h (Screen Height in cm)
    sc_w (Screen Width in cm)
    talk_time (Talk Time in hours)
    '''

    ''' DISCRETICACIÓN 
    columnas_continuas = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc',
                          'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']
    df = discretize(df, columnas_continuas)
    '''


    # Preparar les dades per a l'entrenament
    S = df.values  # El conjunto de entrenamiento completo
    X = df.drop(columns=['price_range']).values
    y = df['price_range'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Crear i entrenar el model
    model = ID3(max_depth=10)
    model.fit(X_train, y_train)

    # Visualització de l'arbre
    print_tree(model.root)

    ''' VALIDACIO DE BASE DE DADES METRICA ACCURACY PARA CROSS VALIDATION & LOOCV '''
    # Cross-validation
    scores = cross_val_score(model, X, y, cv=10)
    print(f'Average accuracy: {np.mean(scores)}')

    # Leave one out
    loocv = LeaveOneOut()
    loocv_scores = cross_val_score(model, X, y, cv=loocv)
    print(f'Average accuracy: {np.mean(loocv_scores)}')


if __name__ == "__main__":
    main()
