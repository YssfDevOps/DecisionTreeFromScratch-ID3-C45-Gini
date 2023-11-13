__authors__ = ['1638618, 1636517, 1633311']
__group__ = 'GM08:30_3'

from copy import copy, deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# Classe Node
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


# Funció per calcular l'entropia total.
def entropy_total(S, X_train, label):
    total_row = S.shape[0]
    entropy = 0

    for c in X_train:
        class_count = S[S[label] == c].shape[0]
        class_entropy = - (class_count / total_row) * np.log2(class_count / total_row)
        entropy += class_entropy

    return entropy

# Funció per calcular l'entropia d'una clase especifica.
def entropy_specific(X_train, feature, label):
    class_count = feature.shape[0]
    entropy = 0

    for c in X_train:
        label_class_count = feature[feature[label] == c].shape[0]
        class_entropy = 0
        if label_class_count != 0:
            class_probability = label_class_count / class_count
            class_entropy = - class_probability * np.log2(class_probability)
        entropy += class_entropy
    return entropy

def gainID3(S, X_train, label, feature):
    feature_value_list = S[feature].unique()  # unqiue values of the feature
    total_row = S.shape[0]
    feature_info = 0.0

    for feature_value in feature_value_list:
        feature_value_data = S[S[feature] == feature_value]  # filtering rows with that feature_value
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = entropy_specific(X_train, label, feature_value_data)
        feature_value_probability = feature_value_count / total_row
        feature_info += feature_value_probability * feature_value_entropy

    return entropy_total(S, X_train, label) - feature_info

# Funció per eliminar files amb valors buits.
def remove_missing_values(df):
    return df.dropna()


# Funció per discretitzar valors contínues.
def discretize(df, columns):
    for column in columns:
        df[column] = pd.qcut(df[column], q=4, labels=False)
    return df

def ID3(S, X_train, label):
    feature_list = S.columns.drop(label)

    max_info_gain = -1
    max_info_feature = None

    for feature in feature_list:  # for each feature in the dataset
        feature_info_gain = gainID3(S, X_train, label, feature)
        if max_info_gain < feature_info_gain:  # selecting feature name with the highest information gain
            max_info_gain = feature_info_gain
            max_info_feature = feature

    return max_info_feature

def C45(S, X_train, label):
    pass

def TreePruning(S, T, y):
    # Seleccionar un nodo t en T de manera que al podarlo se mejora máximamente algún criterio de evaluación.
    t = select_node_to_prune(T, S, y)

    # Podar el nodo t si existe.
    while t is not None:
        T = pruned(T, t)
        t = select_node_to_prune(T, S, y)

    return T


def StoppingCriterion(S):
    # Cuando todos los ejemplos que quedan pertenecen a la misma clase.
    if np.unique(S).size == 1:
        return True

    # Cuando no quedan atributos por los que ramificar.
    if S.shape[1] == 0:
        return True

    # Cuando no quedan datos para clasificar.
    if S.shape[0] == 0:
        return True

    return False


def SplitCriterion(S, X_train, label, tipus):
    if tipus == 0:  # ID 3
        return ID3(S, X_train, label)
    elif tipus == 1:  # C4.5
        return C45()

    return None


def treeGrowing(S, X_train, y_train, tipusSplit):
    # Crear un nuevo árbol T con un solo nodo raíz.
    T = Node()

    if StoppingCriterion(S):
        T.value = np.bincount(y_train).argmax()  # Etiqueta con el valor más común de y en S.
    else:
        # Encontrar el atributo a que obtiene el mejor SplitCriterion.
        a = SplitCriterion(S, X_train, y_train, tipusSplit)

        # Etiquetar t con a.
        T.feature = a

        # Para cada valor vi de a.
        for vi in np.unique(X_train[:, a]):
            # Crear un subárbol para cada valor único de a.
            sub_S = S[X_train[:, a] == vi]
            usb_y = y_train[X_train[:, a] == vi]
            sub_X = X_train[X_train[:, a] == vi]

            # Crear un subárbol con las instancias que tienen a = vi.
            subtree = treeGrowing(sub_S, sub_X, usb_y, SplitCriterion)

            # Conectar el nodo raíz de T a Subtreei con una arista etiquetada como vi.
            if vi <= T.threshold:
                T.left = subtree
            else:
                T.right = subtree

        # Devolver el árbol podado.
    return TreePruning(S, T, y_train)


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

    # Tractament de valors buits.
    df = remove_missing_values(df)

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
    #df = discretize(df, ['column1', 'column2'])

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

    # Validació creuada
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation accuracy: {np.mean(scores)}")


if __name__ == "__main__":
    main()
