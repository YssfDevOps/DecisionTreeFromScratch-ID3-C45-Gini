import numpy as np


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

class DecisionTreeCasifier:
    def __init__(self, criterion='ID3'):
        self.root_node = Node()
        self.criterion = criterion # {'ID3_Entropy', 'C45_Entropy', 'ID3_Gini', 'C45_Gini'}

    def __str__(self): # Print tree
        if self.root_node.is_leaf_node():
            return f"There is no tree to be printed!"
        self.print_tree(self.root_node)

    def print_tree(self, node, depth=0):
        if node.is_leaf_node():
            print(f"{depth * '  '}[{node.value}]")
            return

        print(f"{depth * '  '}[{node.feature} <= {node.threshold}]")
        self.print_tree(node.left, depth + 1)
        self.print_tree(node.right, depth + 1)

    # Funció per calcular l'entropia total.
    def entropy_total(self, S, X_train, label):
        total_row = S.shape[0]
        entropy = 0

        for c in X_train:
            class_count = S[S[label] == c].shape[0]
            class_entropy = - (class_count / total_row) * np.log2(class_count / total_row)
            entropy += class_entropy

        return entropy

    # Funció per calcular l'entropia d'una clase especifica.
    def entropy_specific(self, X_train, feature, label):
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

    def gainID3(self, S, X_train, label, feature):
        feature_value_list = S[feature].unique()  # unqiue values of the feature
        total_row = S.shape[0]
        feature_info = 0.0

        for feature_value in feature_value_list:
            feature_value_data = S[S[feature] == feature_value]  # filtering rows with that feature_value
            feature_value_count = feature_value_data.shape[0]
            feature_value_entropy = self.entropy_specific(X_train, label, feature_value_data)
            feature_value_probability = feature_value_count / total_row
            feature_info += feature_value_probability * feature_value_entropy

        return self.entropy_total(S, X_train, label) - feature_info

    def SplitInfo(self, S, X_train, label, feature):
        feature_value_list = S[feature].unique()  # unqiue values of the feature
        total_row = S.shape[0]
        feature_split_info = 0.0

        for feature_value in feature_value_list:
            feature_value_data = S[S[feature] == feature_value]  # filtering rows with that feature_value
            feature_value_count = feature_value_data.shape[0]
            feature_value_probability = feature_value_count / total_row
            feature_split_info += feature_value_probability * np.log2(feature_value_probability)

        return feature_split_info

    def gainRatio(self, S, X_train, label, feature):
        return self.gainID3(S, X_train, label, feature) / self.SplitInfo(S, X_train, label, feature)

    def gini(self, S, X_train, label, feature):
        # Extrae las etiquetas y características para las muestras en S
        y = X_train.loc[S, label]

        total_samples = len(S)

        if total_samples == 0:
            return 0  # Manejar el caso cuando el conjunto de datos está vacío

        # Contar la frecuencia de cada clase en el conjunto de datos
        class_counts = y.value_counts()

        # Calcular el índice de Gini
        gini_index = 1.0
        for count in class_counts:
            label_probability = count / total_samples
            gini_index -= label_probability ** 2

        return gini_index

    def gainGini(self, S, X_train, label, feature):
        gini_general = self.gini(S, X_train, label, feature)

        feature_value_list = S[feature].unique()  # unqiue values of the feature
        total_row = S.shape[0]
        feature_split_info = 0.0

        for feature_value in feature_value_list:
            feature_value_data = S[S[feature] == feature_value]  # filtering rows with that feature_value
            feature_value_count = feature_value_data.shape[0]
            feature_value_probability = feature_value_count / total_row
            feature_gini_info = self.gini(feature_value_count, feature_value_data, label, feature_value)
            feature_split_info += feature_value_probability * feature_gini_info

        return feature_split_info

    def giniRatio(self, S, X_train, label, feature):
        return self.gainGini(S, X_train, label, feature) / self.SplitInfo(S, X_train, label, feature)


    def StoppingCriterion(self, S):
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

    def ID3(self, S, X_train, label, entropy):
        feature_list = S.columns.drop(label)

        max_info_gain = -1
        max_info_feature = None

        for feature in feature_list:  # for each feature in the dataset
            feature_info_gain = self.gainID3(S, X_train, label, feature) if entropy else self.gini(S, X_train, label, feature)
            if max_info_gain < feature_info_gain: # selecting feature name with the highest information gain
                max_info_gain = feature_info_gain
                max_info_feature = feature

        return max_info_feature
    def C45(self, S, X_train, label, entropy):
        feature_list = S.columns.drop(label)

        max_info_gain = -1
        max_info_feature = None

        for feature in feature_list:  # for each feature in the dataset
            feature_info_gain = self.gainRatio(S, X_train, label, feature) if entropy else self.giniRatio(S, X_train, label, feature)
            if max_info_gain < feature_info_gain:  # selecting feature name with the highest information gain
                max_info_gain = feature_info_gain
                max_info_feature = feature

        return max_info_feature

    def TreePruning(self, S, T, y):
        # Seleccionar un nodo t en T de manera que al podarlo se mejora máximamente algún criterio de evaluación.
        t = self.select_node_to_prune(T, S, y)

        # Podar el nodo t si existe.
        while t is not None:
            T = self.pruned(T, t)
            t = self.select_node_to_prune(T, S, y)

        return T




    def SplitCriterion(self, S, X_train, label):# {'ID3_Entropy', 'C45_Entropy', 'ID3_Gini', 'C45_Gini'}
        if self.criterion == 'ID3_Gini':  # ID 3 Gini
            return self.ID3(S, X_train, label, 0)
        elif self.criterion == 'C45_Entropy':  # C4.5 Entropy
            return self.C45(S, X_train, label, 1)
        elif self.criterion == 'C45_Gini':  # C4.5 Gini
            return self.C45(S, X_train, label, 0)
        else: # ID3_Entropy by default
            return self.ID3(S, X_train, label, 1) # ID 3 Entropy


    def tree_growing(self, S, X_train, y_train):
        # Crear un nuevo árbol T con un solo nodo raíz.
        T = self.root_node

        if self.StoppingCriterion(S):
            T.value = np.bincount(y_train).argmax()  # Etiqueta con el valor más común de y en S.
        else:
            # Encontrar el atributo a que obtiene el mejor SplitCriterion.
            a = self.SplitCriterion(S, X_train, y_train)

            # Etiquetar t con a.
            T.feature = a

            # Para cada valor vi de a.
            for vi in np.unique(X_train[:, a]):
                # Crear un subárbol para cada valor único de a.
                sub_S = S[X_train[:, a] == vi]
                usb_y = y_train[X_train[:, a] == vi]
                sub_X = X_train[X_train[:, a] == vi]

                # Crear un subárbol con las instancias que tienen a = vi.
                subtree = self.tree_growing(sub_S, sub_X, usb_y)

                # Conectar el nodo raíz de T a Subtreei con una arista etiquetada como vi.
                if vi <= T.threshold:
                    T.left = subtree
                else:
                    T.right = subtree

            # Devolver el árbol podado.
        return self.TreePruning(S, T, y_train)

    def predict_rec(self, X, node):
        if node.is_leaf_node():
            return node.value
        elif X[node.feature] <= node.threshold:
            return self.predict_rec(X, node.left)
        else:
            return self.predict_rec(X, node.right)

    def predict(self, X):
        return self.predict_rec(X, self.root_node)

    def fit(self, X_train, y_train):
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same number of samples")

            # Use NumPy's concatenate function to combine the data along the appropriate axis
        S = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
        return self.tree_growing(S, X_train, y_train)
