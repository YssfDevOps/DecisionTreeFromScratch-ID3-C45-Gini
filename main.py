__authors__ = ['1638618, 1636517, 1633311']
__group__ = 'GM08:30_3'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, LeaveOneOut
from DecisionTreeClassifier import DecisionTreeClassifier
import sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support



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

def calculate_accuracy(estimator, X, y):
    res = []
    for index, value in enumerate(X):
        y_pred = estimator.predict(value)
        res.append(y_pred == y[index])

    return(sum(res) / len(res))

def calculate_metrics(y_pred, y_test, average='weighted'):
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Precision, Recall, and F1 Score
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average=average)

    return accuracy, precision, recall, f1_score

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
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Ajustar y transformar los datos.
    discretize(df, ['battery_power', 'clock_speed', 'fc', 'ram', 'talk_time', 'int_memory', 'm_dep', 'mobile_wt',
                    'pc', 'px_height', 'px_width', 'sc_h', 'sc_w'])

    # Separate target from predictors
    X = np.array(df.drop('price_range', axis=1).copy())
    y = np.array(df['price_range'].copy())
    feature_names = list(df.columns)[:-1]  # Assuming the last column is the target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tipo = 'Gini'
    # Create and fit the decision tree classifier using ID3 criterion
    tree_clf = DecisionTreeClassifier(criterion=tipo, nombres_atributos=feature_names)  # 'entropy' corresponds to ID3 criterion

    tree_clf.fit(X=X_train, y=y_train)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Visualització de l'arbre
    tree_clf.print_tree()
    original_stdout = sys.stdout  # Guarda la salida estándar original

    with open(tipo+'.txt', 'w') as f:
        sys.stdout = f  # Cambia la salida estándar al archivo que estamos escribiendo
        tree_clf.print_tree()  # Llama a la función
        sys.stdout = original_stdout  # Restaura la salida estándar original

    y_pred = tree_clf.predict(X_test)
    accuracy, precision, recall, f1_score = calculate_metrics(y_pred, y_test)

    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("f1_score: {:.4f}".format(f1_score))

    ''' VALIDACIO DE BASE DE DADES METRICA ACCURACY PARA CROSS VALIDATION & LOOCV '''
    # Cross-validation
    scores = cross_val_score(tree_clf, X_train, y_train, cv=10, scoring='f1')
    print(f'Average accuracy: {np.mean(scores)}')

    # Leave one out
    loocv = LeaveOneOut()
    loocv_scores = cross_val_score(tree_clf, X_train, y_train, cv=loocv, scoring='f1')
    print(f'Average accuracy: {np.mean(loocv_scores)}')


if __name__ == "__main__":
    main()
