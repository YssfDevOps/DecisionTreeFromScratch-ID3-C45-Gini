import numpy as np


# Classe Node
class Node:
    def __init__(self, feature=None, left=None, right=None, value=None):
        self.feature = feature
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeCasifier:
    def __init__(self, criterion='ID3'):
        self.root = None
        self.root_node = Node()
        self.criterion = criterion  # {'ID3_Entropy', 'C45_Entropy', 'ID3_Gini', 'C45_Gini'}

    def __str__(self):  # Print tree
        if self.root_node.is_leaf_node():
            return f"There is no tree to be printed!"
        self.print_tree(self.root_node)

    def calculo_entropia_total(self, X_train, clases, atributo):
        entropia_total = 0
        dim_fila = X_train.shape[0]

        for clase in clases:
            num_total_clases = X_train[X_train[atributo] == clase].shape[0]
            entropia_total_clase = - (num_total_clases / dim_fila) * np.log2(num_total_clases / dim_fila)
            entropia_total += entropia_total_clase

        return entropia_total

    def calculo_entropia_label(self, datos_atributo, clases, atributo):
        entropia_total = 0
        dim_fila = datos_atributo.shape[0]

        for clase in clases:
            num_total_atributo = datos_atributo[datos_atributo[atributo] == clase].shape[0]
            entropia_atributo = (- (num_total_atributo/dim_fila)*np.log2(num_total_atributo/dim_fila)) if num_total_atributo != 0 else 0
            entropia_total += entropia_atributo

        return entropia_total

    def gainID3(self, X_train, clases, atributo, caracteristica):
        valor_info = 0
        dim_fila = X_train.shape[0]
        valors_caracteristica = X_train[caracteristica].unique()

        for carac in valors_caracteristica:
            carac_datos = X_train[X_train[caracteristica] == carac]
            carac_dim = carac_datos.shape[0]
            carac_probabilidad = carac_dim / dim_fila
            carac_entropia = self.calculo_entropia_label(carac_datos, clases, atributo)
            valor_info += carac_probabilidad * carac_entropia

        gain = self.calculo_entropia_total(X_train, clases, atributo) - valor_info
        return gain

    def ID3(self, X_train, clases, atributo):
        gain_info = -1
        caracteristica_info = None
        caracteristicas = X_train.columns.drop(atributo)

        for carac in caracteristicas:
            gain = self.gainID3(X_train, clases, atributo, carac)
            if gain_info < gain:
                gain_info = gain
                caracteristica_info = carac

        return caracteristica_info

    def calculo_ganancia_info(self, X_train, clases, atributo):
        ganancia_info_total = 0
        dim_fila = X_train.shape[0]

        for clase in clases:
            num_total_clases = X_train[X_train[atributo] == clase].shape[0]
            if num_total_clases != 0:
                ganancia_info_total_clase = - (num_total_clases / dim_fila) * np.log2(num_total_clases / dim_fila)
                ganancia_info_total += ganancia_info_total_clase

        return ganancia_info_total

    def calculo_gini(self, datos_atributo, clases, atributo):
        gini_total = 1
        dim_fila = datos_atributo.shape[0]

        for clase in clases:
            num_total_atributo = datos_atributo[datos_atributo[atributo] == clase].shape[0]
            prob = num_total_atributo / dim_fila
            gini_total -= prob ** 2

        return gini_total

    def gain_ratio_guany(self, X_train, clases, atributo, caracteristica):
        split_info = 0
        dim_fila = X_train.shape[0]
        valors_caracteristica = X_train[caracteristica].unique()

        for carac in valors_caracteristica:
            carac_datos = X_train[X_train[caracteristica] == carac]
            carac_dim = carac_datos.shape[0]
            carac_probabilidad = carac_dim / dim_fila
            carac_ganancia_info = self.calculo_ganancia_info(carac_datos, clases, atributo)
            if carac_probabilidad > 0:
                split_info -= carac_probabilidad * np.log2(carac_probabilidad)

        if split_info != 0:
            gain_ratio = (self.calculo_ganancia_info(X_train, clases, atributo) - carac_ganancia_info) / split_info
        else:
            gain_ratio = 0

        return gain_ratio

    def gain_ratio_gini(self, X_train, clases, atributo, caracteristica):
        split_info = 0
        dim_fila = X_train.shape[0]
        valors_caracteristica = X_train[caracteristica].unique()

        for carac in valors_caracteristica:
            carac_datos = X_train[X_train[caracteristica] == carac]
            carac_dim = carac_datos.shape[0]
            carac_probabilidad = carac_dim / dim_fila
            carac_gini = self.calculo_gini(carac_datos, clases, atributo)
            if carac_probabilidad > 0:
                split_info -= carac_probabilidad * np.log2(carac_probabilidad)

        if split_info != 0:
            gain_ratio = (1 - self.calculo_gini(X_train, clases, atributo)) / split_info
        else:
            gain_ratio = 0

        return gain_ratio

    def C45(self, X_train, clases, atributo, tipo=1):
        gain_ratio_max = -1
        caracteristica_max = None
        caracteristicas = X_train.columns.drop(atributo)

        for carac in caracteristicas:
            if tipo == 1:
                gain_ratio = self.gain_ratio_guany(X_train, clases, atributo, carac)
            else:  # asumimos que cualquier otro valor para 'tipo' debería usar 'gini'
                gain_ratio = self.gain_ratio_gini(X_train, clases, atributo, carac)

            if gain_ratio_max < gain_ratio:
                gain_ratio_max = gain_ratio
                caracteristica_max = carac

        return caracteristica_max

    def SplitCriterion(self, X_train, clases, atributo):  # {'ID3_Entropy', 'C45_Entropy', 'C45_Gini'}
        if self.criterion == 'ID3_Gini':  # ID 3 Gini
            return self.ID3(X_train, clases, atributo)
        elif self.criterion == 'C45_Entropy':  # C4.5 Entropy
            return self.C45(X_train, clases, atributo, 1)
        elif self.criterion == 'C45_Gini':  # C4.5 Gini
            return self.C45(X_train, clases, atributo, 0)
        else:  # ID3_Entropy by default
            return self.ID3(X_train, clases, atributo)  # ID 3 Entropy

    def StoppingCriterion(self, X_train):
        # Cuando todos los ejemplos que quedan pertenecen a la misma clase.
        if np.unique(X_train).size == 1:
            return True

        # Cuando no quedan atributos por los que ramificar.
        if X_train.shape[1] == 0:
            return True

        # Cuando no quedan datos para clasificar.
        if X_train.shape[0] == 0:
            return True

        return False

    def tree_growing(self, X_train, objectiu, clases, tipus):
        self.criterion = tipus
        if self.StoppingCriterion(X_train) is False:
            caracteristica = self.SplitCriterion(X_train, clases, objectiu)

            # Aquí es donde se genera el subárbol
            feature_value_count_dict = X_train[caracteristica].value_counts(sort=False)
            tree = {}

            for feature_value, count in feature_value_count_dict.iteritems():
                feature_value_data = X_train[X_train[caracteristica] == feature_value]

                assigned_to_node = False
                for c in clases:
                    class_count = feature_value_data[feature_value_data[objectiu] == c].shape[0]

                    if class_count == count:
                        tree[feature_value] = Node(value=c)
                        X_train = X_train[X_train[caracteristica] != feature_value]
                        assigned_to_node = True
                if not assigned_to_node:
                    tree[feature_value] = Node(value="?")

            next_root = None

            if self.root != None:
                self.root.feature[caracteristica] = dict()
                self.root.feature[caracteristica][caracteristica] = tree
                next_root = self.root.feature[caracteristica][caracteristica]
            else:
                self.root = Node(feature=caracteristica)
                self.root.feature[caracteristica] = tree
                next_root = self.root.feature[caracteristica]

            for node, branch in list(next_root.items()):
                if branch.value == "?":
                    feature_value_data = X_train[X_train[caracteristica] == node]
                    self.tree_growing(feature_value_data, objectiu, clases, tipus)

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


    def print_tree(self, node, depth=0):
        if node.is_leaf_node():
            print(f"{depth * '  '}[{node.value}]")
            return

        print(f"{depth * '  '}[{node.feature} <= {node.threshold}]")
        self.print_tree(node.left, depth + 1)
        self.print_tree(node.right, depth + 1)
