import numpy as np
from collections import deque

# Classe Node
class Node:
    """Contains the information of the node and another nodes of the Decision Tree."""
    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None


class DecisionTreeClassifier:
    def __init__(self, nombres_atributos, criterion='ID3'):
        self.nodo = None
        self.criterion = criterion  # {'ID3_Entropy', 'C45_Entropy', 'ID3_Gini', 'C45_Gini'}
        self.X = None
        self.nombres_atributos = nombres_atributos
        self.etiquetas = None
        self.categoriasEtiquetas = None
        self.conteo = None
        self.entropia = None

    def __str__(self):  # Print tree
        if self.root_node.is_leaf_node():
            return f"There is no tree to be printed!"
        self.print_tree(self.root_node)

    def calcular_entropy(self, lista_attr):
        etiquetas = [self.etiquetas[i] for i in lista_attr]
        conteo_etiquetas = [etiquetas.count(x) for x in self.categoriasEtiquetas]
        entropia = sum([-conteo / len(lista_attr) * np.log2(conteo / len(lista_attr)) if conteo else 0 for conteo in
                        conteo_etiquetas])
        return entropia

    def calcular_ganancia_informacion(self, lista_attr, id_caracteristica):
        ganancia_info = self.calcular_entropy(lista_attr)
        caracteristicas = [self.X[x][id_caracteristica] for x in lista_attr]
        valuees_caracteristicas = list(set(caracteristicas))
        conteo_valuees_caracteristicas = [caracteristicas.count(x) for x in valuees_caracteristicas]
        attr_valuees_caracteristicas = [[lista_attr[i] for i, x in enumerate(caracteristicas) if x == y] for y in valuees_caracteristicas]
        ganancia_info = ganancia_info - sum([conteo_valuees / len(lista_attr) * self.calcular_entropy(ids_valuees)
                                             for conteo_valuees, ids_valuees in
                                             zip(conteo_valuees_caracteristicas, attr_valuees_caracteristicas)])
        return ganancia_info

    def obtener_maxima_ganancia_informacion(self, attr_muestras, attr_atributos):
        entropia_atributos = [self.calcular_ganancia_informacion(attr_muestras, id_atributo) for id_atributo in attr_atributos]
        id_maximo = attr_atributos[entropia_atributos.index(max(entropia_atributos))]
        return self.nombres_atributos[id_maximo], id_maximo

    def ID3(self):
        lista_attr = [x for x in range(len(self.X))]
        caracteristicas_attr = [x for x in range(len(self.nombres_atributos))]
        self.nodo = self.ID3Recursive(lista_attr, caracteristicas_attr, self.nodo)
        print('')

    def ID3Recursive(self, lista_attr, caracteristicas_attr, nodo):
        """ID3 algorithm. It is called recursively until some criteria is met.
                Parameters
                __________
                :param x_ids: list, list containing the samples ID's
                :param feature_ids: list, List containing the feature ID's
                :param node: object, An instance of the class Nodes
                __________
                :returns: An instance of the class Node containing all the information of the nodes in the Decision Tree
                """
        if not nodo:
            nodo = Node()  # initialize nodes

        attribute = [self.etiquetas[x] for x in lista_attr]

        # SPLIT CRITERION
        if len(set(attribute)) == 1:
            nodo.value = self.etiquetas[lista_attr[0]]
            return nodo
        if len(caracteristicas_attr) == 0:
            nodo.value = max(set(attribute), key=attribute.count)  # compute mode
            return nodo

        best_feature_name, best_feature_id = self.obtener_maxima_ganancia_informacion(lista_attr, caracteristicas_attr)
        nodo.value = best_feature_name
        nodo.childs = []
        # value of the chosen feature for each instance
        feature_values = list(set([self.X[x][best_feature_id] for x in lista_attr]))
        # loop through all the values
        for value in feature_values:
            child = Node()
            child.value = value  # add a branch from the node to each feature value in our feature
            nodo.childs.append(child)  # append new child node to current node
            child_x_ids = [x for x in lista_attr if self.X[x][best_feature_id] == value]
            if not child_x_ids:
                child.next = max(set(attribute), key=attribute.count)
                print('')
            else:
                if caracteristicas_attr and best_feature_id in caracteristicas_attr:
                    to_remove = caracteristicas_attr.index(best_feature_id)
                    caracteristicas_attr.pop(to_remove)
                # recursively call the algorithm
                child.next = self.ID3Recursive(child_x_ids, caracteristicas_attr, child.next)
        return nodo


    def C45(self, X_train, clases, atributo, tipo=1):
        pass

    def SplitCriterion(self, X_train, clases, atributo):  # {'ID3', 'C45_Entropy', 'C45_Guany'}
        if self.criterion == 'C45_Guany':  # C4.5 Entropy
            return self.C45(X_train, clases, atributo, 1)
        elif self.criterion == 'C45_Gini':  # C4.5 Gini
            return self.C45(X_train, clases, atributo, 0)
        else:  # ID3_Entropy by default
            return self.ID3(X_train, clases, atributo)  # ID 3 Entropy

    def predict_rec(self, X, node):
        if node.is_leaf_node():
            return node.value
        elif X[node.feature] <= node.value:
            return self.predict_rec(X, node.left)
        else:
            return self.predict_rec(X, node.right)

    def predict(self, X):
        return self.predict_rec(X, self.root_node)

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError("X_train and y_train must have the same number of samples")
        # Inicializar variables en base a los datos
        self.X = X
        self.etiquetas = y
        self.categoriasEtiquetas = list(set(y))
        self.conteo = [list(y).count(x) for x in self.categoriasEtiquetas]
        self.entropia = self.calcular_entropy([x for x in range(len(self.etiquetas))])  # calculates the initial entropy

        self.ID3() # Aqui va lo de decidir el criterio


    def print_tree(self):
        if not self.nodo:
            return
        nodes = deque()
        nodes.append(self.nodo)
        while len(nodes) > 0:
            node = nodes.popleft()
            print(node.value)
            if node.childs:
                for child in node.childs:
                    print('({})'.format(child.value))
                    nodes.append(child.next)
            elif node.next:
                print(node.next)

