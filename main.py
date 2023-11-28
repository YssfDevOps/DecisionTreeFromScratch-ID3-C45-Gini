__authors__ = ['1638618, 1636517, 1633311']
__group__ = 'GM08:30_3'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from DecisionTreeClassifier import DecisionTreeClassifier
import sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
import graphviz


def parse_tree(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def build_graph(tree_lines, graph, parent=None):

    value = 0
    for line in tree_lines:
        node = line.strip().split()
        if node.startswith('('):
            # Leaf node
            graph.node(value, label=f"{node}\n{value}")
            if parent is not None:
                graph.edge(parent, value)
        else:
            # Non-leaf node
            graph.node(value, label=node)
            if parent is not None:
                graph.edge(parent, value)
            sub_lines = tree_lines[tree_lines.index(line) + 1:]
            sub_lines = [sub_line for sub_line in sub_lines if sub_line.startswith('\t')]
            build_graph(sub_lines, graph, parent=value)

def printSection(title):
    print("====================================================")
    print(title)
    print("====================================================")

# Funció per eliminar files amb valors buits.
def remove_missing_values(df):
    return df.dropna()


# Funció per discretitzar valors contínues.
def discretize(df, columns):
    for column in columns:
        df[column] = pd.qcut(df[column], q=4, labels=False)
    return df

def calculate_metrics(y_pred, y_test, average='weighted'):
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Precision, Recall, and F1 Score
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average=average)

    return accuracy, precision, recall, f1_score

def save_tree(tree_clf, tipo='ID3'):
    original_stdout = sys.stdout  # Guarda la salida estándar original

    with open(tipo + '.txt', 'w') as f:
        sys.stdout = f  # Cambia la salida estándar al archivo que estamos escribiendo
        tree_clf.print_tree()  # Llama a la función
        sys.stdout = original_stdout  # Restaura la salida estándar original


def LOOCV_f1(estimator, X_train, y_train):
    f1_score_total = 0
    for i, X_val in enumerate(X_train):
        # Prepare data
        y_val = y_train[i]
        y_train_val = np.delete(y_train, i)
        X_train_val = np.delete(X_train, i, axis=0)
        # Train and predict
        estimator.fit(X_train_val, y_train_val)
        y_pred = estimator.predict([X_val])
        # Calculate metric
        _, _, _, f1 = calculate_metrics(y_pred, [y_val])
        f1_score_total += f1
    # Return Mean
    return f1_score_total / len(X_train)


def evaluar(model, x_test_val, y_test_val):
    for i in range(len(x_test_val)):
        y_pred = model.predict(x_test_val)
        _, _, _, resultat = calculate_metrics(y_pred, y_test_val)
    return resultat

def k_fold(cv, X, y, estimator):
    subset_size = len(X) // cv
    cv_scores = []

    for i in range(cv):
        x_test_val = X[i * subset_size: (i + 1) * subset_size]
        y_test_val = y[i * subset_size: (i + 1) * subset_size]
        x_train_val = np.concatenate((X[:i * subset_size], X[(i + 1) * subset_size:]))
        y_train_val = np.concatenate((y[:i * subset_size], y[(i + 1) * subset_size:]))

        estimator.fit(X=x_train_val, y=y_train_val)

        score = evaluar(estimator, x_test_val, y_test_val)

        cv_scores.append(score)

    return sum(cv_scores) / len(cv_scores)

def encode_titanic(train):
    # Initial encode
    train['hasCabin'] = train['Cabin'].notna()
    train['hasFamiliar'] = train['SibSp'] != 0
    # Drop unique values
    train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    train.dropna(inplace=True)
    # Get dummies of other encodable columns
    sex = pd.get_dummies(train['Sex'])
    embark = pd.get_dummies(train['Embarked'])
    train.drop(['Sex', 'Embarked'], axis=1, inplace=True)
    train = pd.concat([train, sex, embark], axis=1)
    return train

# Funció principal
def main():
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    printSection("TRAIN LOAD")
    """
    ====================================================
    Preparar datos de nuestro dataset
    ====================================================
    """
    # Carregar les dades
    df = pd.read_csv('train.csv')
    nan_count = df.isnull().sum()
    nan_percentage = df.isnull().mean() * 100
    print("Número de NaNs por columna:")
    print(nan_count)
    print("\nPorcentaje de NaNs por columna:")
    print(nan_percentage)

    # Tractament de valors buits APARTAT C (Està comentat, ja que utilitzem el tractament de valors buit de l'apartat B)
    # df = remove_missing_values(df)

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

    """
    ====================================================
    Evaluación del modelo y las metricas     
    ====================================================
    """
    printSection("METRIC SHOWCASE")
    tipo = ['ID3', 'C45', 'Gini']
    for t in tipo:
        print("Criteri {}:".format(t))
        # Create and fit the decision tree classifier using ID3 criterion
        tree_clf = DecisionTreeClassifier(criterion=t, nombres_atributos=feature_names)  # 'entropy' corresponds to ID3 criterion

        tree_clf.fit(X=X_train, y=y_train)

        # Visualització de l'arbre
        #tree_clf.print_tree()
        #save_tree(tree_clf, t)

        y_pred = tree_clf.predict(X_test)
        accuracy, precision, recall, f1_score = calculate_metrics(y_pred, y_test)

        print("Accuracy: {:.4f}".format(accuracy))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("f1_score: {:.4f}".format(f1_score))
        print("")

    """
    ====================================================
    Cross-Validation 
    ====================================================
    """
    printSection("CROSS VALIDATION")
    # K_Fold
    tipo = ['ID3', 'C45', 'Gini']
    mean_kfold_f1 = []
    # Cross Validation k-fold
    for t in tipo:
        tree_clf = DecisionTreeClassifier(criterion=t, nombres_atributos=feature_names)
        mean_kfold_f1.append(k_fold(10, X_train, y_train, tree_clf))
    print(mean_kfold_f1)
    # El criteri ID3 ens hauria de donar un major f1-score de promitg
    print("Best criterion ID3")

    # LOOCV (NOTA: ESTA PARTE DEL CODIGO TARDA BASTANTE YA QUE EL DATASET ES UN POCO GRANDE)
    #mean_loocv_f1 = []
    #for t in tipo:
    #    dtc = DecisionTreeClassifier(nombres_atributos=feature_names, criterion=t)
    #    mean_loocv_f1.append(LOOCV_f1(dtc, X_train, y_train))
    #print(mean_loocv_f1)

    """
    ====================================================
    Evaluacion del mejor modelo 
    ====================================================
    """
    printSection("PART C RESULTS")
    best_criterion = 'ID3'
    tree_clf = DecisionTreeClassifier(criterion=best_criterion, nombres_atributos=feature_names)  # 'entropy' corresponds to ID3 criterion

    tree_clf.fit(X=X_train, y=y_train)

    # Visualització de l'arbre
    tree_clf.print_tree_graph(output_file_path='mobile_dataset_ID3')
    #tree_clf.print_tree()
    # save_tree(tree_clf, t)

    y_pred = tree_clf.predict(X_test)
    accuracy, precision, recall, f1_score = calculate_metrics(y_pred, y_test)

    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("f1_score: {:.4f}".format(f1_score))

def main_titanic():
    """
    ====================================================
    B: Simple imputer in titanic dataset
    ====================================================
    """
    printSection("PART B: TITANIC LOAD")
    # Carregar les dades
    dft = pd.read_csv('train_titanic.csv')
    nan_count = dft.isnull().sum()
    nan_percentage = dft.isnull().mean() * 100
    print("Número de NaNs por columna:")
    print(nan_count)
    print("\nPorcentaje de NaNs por columna:")
    print(nan_percentage)

    # Codificar datos
    encode_titanic(dft)

    # Tractament de valors buits APARTAT B.
    imputer = SimpleImputer(strategy='mean')
    dft = pd.DataFrame(imputer.fit_transform(dft), columns=dft.columns)

    # Preprocessing
    # Discretizar todos los datos que sean numericos
    discretize(dft, ['Age', 'Fare'])

    # Separate target from predictors
    X = np.array(dft.drop('Survived', axis=1).copy())
    y = np.array(dft['Survived'].copy())
    feature_names = list(dft.columns)[1:]  # Assuming the last column is the target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    """
    ====================================================
    Cross-Validation Titanic
    ====================================================
    """
    printSection("CROSS VALIDATION")
    # K_Fold
    tipo = ['ID3', 'C45', 'Gini']
    mean_kfold_f1 = []
    # Cross Validation k-fold
    for t in tipo:
        tree_clf = DecisionTreeClassifier(criterion=t, nombres_atributos=feature_names)
        mean_kfold_f1.append(k_fold(10, X_train, y_train, tree_clf))
    print(mean_kfold_f1)

    # El criteri C45 ens hauria de donar un major f1-score de promitg
    print("Best criterion C45")
    """
    ====================================================
    Evaluacion del mejor modelo 
    ====================================================
    """
    printSection("TITANIC RESULTS")
    best_criterion = 'C45'
    tree_clf = DecisionTreeClassifier(criterion=best_criterion,
                                      nombres_atributos=feature_names)  # 'entropy' corresponds to ID3 criterion
    tree_clf.fit(X=X_train, y=y_train)

    # Visualització de l'arbre
    tree_clf.print_tree_graph(output_file_path='titanic_dataset_C45')
    #tree_clf.print_tree()
    # save_tree(tree_clf, t)

    y_pred = tree_clf.predict(X_test)
    accuracy, precision, recall, f1_score = calculate_metrics(y_pred, y_test)

    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("f1_score: {:.4f}".format(f1_score))


if __name__ == "__main__":
    # C
    main()
    # B
    main_titanic()
