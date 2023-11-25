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
            entropia_atributo = (- (num_total_atributo/dim_fila)*np.log2(num_total_atributo/dim_fila)) if num_total_atributo > 0 else 0
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

    



    def SplitCriterion(self, dataset, X_train, label):  # {'ID3_Entropy', 'C45_Entropy', 'ID3_Gini', 'C45_Gini'}
        if self.criterion == 'ID3_Gini':  # ID 3 Gini
            return self.ID3(dataset, X_train, label, 0)
        elif self.criterion == 'C45_Entropy':  # C4.5 Entropy
            return self.C45(dataset, X_train, label, 1)
        elif self.criterion == 'C45_Gini':  # C4.5 Gini
            return self.C45(dataset, X_train, label, 0)
        else: # ID3_Entropy by default
            return self.ID3(dataset, X_train, label, 1)  # ID 3 Entropy

    def StoppingCriterion(self, dataset):
        # Cuando todos los ejemplos que quedan pertenecen a la misma clase.
        if np.unique(dataset).size == 1:
            return True

        # Cuando no quedan atributos por los que ramificar.
        if dataset.shape[1] == 0:
            return True

        # Cuando no quedan datos para clasificar.
        if dataset.shape[0] == 0:
            return True

        return False

    def tree_growing(self, dataset, X_train, y_train):
        # Crear un nuevo árbol T con un solo nodo raíz.
        T = self.root_node

        if self.StoppingCriterion(dataset):
            T.value = np.bincount(y_train).argmax()  # Etiqueta con el valor más común de y en S.
        else:
            # Encontrar el atributo a que obtiene el mejor SplitCriterion.
            a = self.SplitCriterion(dataset, X_train, y_train)

            # Etiquetar t con a.
            T.feature = a

            # Para cada valor vi de a.
            for vi in np.unique(X_train[:, a]):
                # Crear un subárbol para cada valor único de a.
                sub_S = dataset[X_train[:, a] == vi]
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
        return self.TreePruning(dataset, T, y_train)

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
