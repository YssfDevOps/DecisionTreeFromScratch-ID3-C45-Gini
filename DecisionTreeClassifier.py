import numpy as np
import graphviz


# Classe Node
class Node:
    """Contains the information of the node and another nodes of the Decision Tree."""
    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None


class DecisionTreeClassifier():
    def __init__(self, nombres_atributos, criterion='ID3'):
        self.nodo = None
        self.criterion = criterion  # {'ID3', 'C45', 'Gini'}
        self.X = None
        self.nombres_atributos = nombres_atributos
        self.etiquetas = None
        self.categoriasEtiquetas = None

    def __str__(self):  # Print tree
        pass
        #self.print_tree()

    def SplitCriterion(self):  # {'ID3', 'C45', 'Gini'}
        if self.criterion == 'C45':  # C4.5 Entropy
            return self.C45()
        elif self.criterion == 'Gini':  # C4.5 Gini
            return self.Gini()
        else:  # ID3_Entropy by default
            return self.ID3()  # ID 3 Entropy

    def calcular_entropia_atr(self, y):
        clases, conteo = np.unique(y, return_counts=True)
        probabilidades = conteo / len(y)
        entropia = -np.sum(probabilidades * np.log2(probabilidades))
        return entropia

    def particion_binaria(self, X, y):
        # X: Atributo continuo
        # y: Etiquetas de clase

        # Ordenar los valores del atributo en orden ascendente
        indices_ordenados = np.argsort(X)
        X_ordenado = X[indices_ordenados]
        y_ordenado = y[indices_ordenados]

        mejor_ganancia = 0
        mejor_punto_particion = None

        for i in range(1, len(X_ordenado)):
            # Calcular punto medio
            punto_medio = (X_ordenado[i - 1] + X_ordenado[i]) / 2

            # Particionar los datos
            izquierda = y_ordenado[X_ordenado <= punto_medio]
            derecha = y_ordenado[X_ordenado > punto_medio]

            # Calcular la ganancia de información (en este caso, la reducción de entropía)
            ganancia = self.calcular_entropia_atr(y_ordenado) - (
                    (len(izquierda) / len(y_ordenado)) * self.calcular_entropia_atr(izquierda) +
                    (len(derecha) / len(y_ordenado)) * self.calcular_entropia_atr(derecha)
            )

            # Actualizar si encontramos una ganancia mejor
            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_punto_particion = punto_medio

        return mejor_punto_particion, mejor_ganancia

    def tracta_atributs_continus(self, X_train, y_train):
        mejor_punto, ganancia = self.particion_binaria(X_train, y_train)

        

    def predict_rec(self, X, node):
        if node.childs is not None:
            # Find best node
            value = X[self.nombres_atributos.index(node.value)]
            diff_list = [pow(value - n.value, 2) for n in node.childs]
            return self.predict_rec(X, node.childs[diff_list.index(min(diff_list))])
        elif node.next is not None:
            return self.predict_rec(X, node.next)
        else:
            return node.value

    def predict(self, X):
        return [self.predict_rec(x, self.nodo) for x in X]

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError("X_train and y_train must have the same number of samples")
        # Inicializar variables en base a los datos
        self.X = X
        self.etiquetas = y
        self.categoriasEtiquetas = list(set(y))

        self.SplitCriterion()


    def print_tree(self, nodo=None, nivel=0):
        if not nodo:
            nodo = self.nodo

        indent = ' ' * nivel * 4
        print(f'{indent}{nodo.value}')

        if nodo.childs:
            for child in nodo.childs:
                print(f'{indent}({child.value})')
                self.print_tree(child.next, nivel + 1)
        elif nodo.next:
            print(f'{indent}{nodo.next}')

    def build_graph(self, graph, nodo=None, path=""):
        if not nodo:
            nodo = self.nodo
            path = nodo.value
            graph.node(path)

        # Comprovar que l'arbre no esta buit
        if nodo.childs:
            for child in nodo.childs:
                if child.next.childs:
                    # Crida recursiva
                    graph.node(path+child.next.value, label=nodo.value+"="f'{child.value}'+"\n"+child.next.value)
                    graph.edge(path, path+child.next.value)
                    self.build_graph(graph, child.next, path+child.next.value)
                else:
                    # Cas fulla
                    graph.node(path+f'{child.value}'+f'{child.next.value}', label=nodo.value+"="f'{child.value}'+"\n"+f'{child.value}')
                    graph.edge(path, path+f'{child.value}'+f'{child.next.value}')

    def print_tree_graph(self, output_file_path='output_graph'):
        graph = graphviz.Digraph(format='png', engine='dot')
        self.build_graph(graph)
        # Save the graph to a file
        graph.render(output_file_path, format='png', cleanup=True)


    """
    ====================================================
    CODIGO ID3
    ====================================================
    """

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
        #print('')

    def ID3Recursive(self, lista_attr, caracteristicas_attr, nodo):
        if not nodo:
            nodo = Node()  # initialize nodes

        attribute = [self.etiquetas[x] for x in lista_attr]

        # StoppingCriterion
        if len(set(attribute)) == 1:
            nodo.value = self.etiquetas[lista_attr[0]]
            return nodo
        if len(caracteristicas_attr) == 0:
            nodo.value = max(set(attribute), key=attribute.count)  # compute mode
            return nodo

        best_feature_name, best_feature_id = self.obtener_maxima_ganancia_informacion(lista_attr, caracteristicas_attr)
        nodo.value = best_feature_name
        nodo.childs = []

        valors_caracteristiques = list(set([self.X[x][best_feature_id] for x in lista_attr]))

        for value in valors_caracteristiques:
            child = Node()
            child.value = value
            nodo.childs.append(child)
            child_x_ids = [x for x in lista_attr if self.X[x][best_feature_id] == value]
            if not child_x_ids:
                child.next = max(set(attribute), key=attribute.count)
                #print('')
            else:
                if caracteristicas_attr and best_feature_id in caracteristicas_attr:
                    to_remove = caracteristicas_attr.index(best_feature_id)
                    caracteristicas_attr.pop(to_remove)
                # recursively call the algorithm
                child.next = self.ID3Recursive(child_x_ids, caracteristicas_attr, child.next)
        return nodo

    """
    ====================================================
    CODIGO C4.5
    ====================================================
    """
    def calcular_ratio_ganancia(self, lista_attr, id_caracteristica):
        ganancia_info = self.calcular_ganancia_informacion(lista_attr, id_caracteristica)
        caracteristicas = [self.X[x][id_caracteristica] for x in lista_attr]
        valuees_caracteristicas = list(set(caracteristicas))
        conteo_valuees_caracteristicas = [caracteristicas.count(x) for x in valuees_caracteristicas]
        attr_valuees_caracteristicas = [[lista_attr[i] for i, x in enumerate(caracteristicas) if x == y] for y in
                                        valuees_caracteristicas]
        split_info = -sum(
            [conteo_valuees / len(lista_attr) * np.log2(conteo_valuees / len(lista_attr)) if conteo_valuees else 0
             for conteo_valuees in conteo_valuees_caracteristicas])
        if split_info == 0:
            return 0
        else:
            return ganancia_info / split_info

    def obtener_maxima_ratio_ganancia(self, attr_muestras, attr_atributos):
        ratio_ganancia_atributos = [self.calcular_ratio_ganancia(attr_muestras, id_atributo) for id_atributo in
                                    attr_atributos]
        id_maximo = attr_atributos[ratio_ganancia_atributos.index(max(ratio_ganancia_atributos))]
        return self.nombres_atributos[id_maximo], id_maximo

    def C45(self):
        lista_attr = [x for x in range(len(self.X))]
        caracteristicas_attr = [x for x in range(len(self.nombres_atributos))]
        self.nodo = self.C45Recursive(lista_attr, caracteristicas_attr, self.nodo)
        #print('')

    def C45Recursive(self, lista_attr, caracteristicas_attr, nodo):
        if not nodo:
            nodo = Node()  # initialize nodes

        attribute = [self.etiquetas[x] for x in lista_attr]

        # StoppingCriterion
        if len(set(attribute)) == 1:
            nodo.value = self.etiquetas[lista_attr[0]]
            return nodo
        if len(caracteristicas_attr) == 0:
            nodo.value = max(set(attribute), key=attribute.count)  # compute mode
            return nodo

        best_feature_name, best_feature_id = self.obtener_maxima_ratio_ganancia(lista_attr, caracteristicas_attr)
        nodo.value = best_feature_name
        nodo.childs = []

        valors_caracteristiques = list(set([self.X[x][best_feature_id] for x in lista_attr]))

        for value in valors_caracteristiques:
            child = Node()
            child.value = value
            nodo.childs.append(child)
            child_x_ids = [x for x in lista_attr if self.X[x][best_feature_id] == value]
            if not child_x_ids:
                child.next = max(set(attribute), key=attribute.count)
                #print('')
            else:
                if caracteristicas_attr and best_feature_id in caracteristicas_attr:
                    to_remove = caracteristicas_attr.index(best_feature_id)
                    caracteristicas_attr.pop(to_remove)
                # recursively call the algorithm
                child.next = self.C45Recursive(child_x_ids, caracteristicas_attr, child.next)
        return nodo

    """
    ====================================================
    CODIGO GINI
    ====================================================
    """

    def calcular_gini(self, lista_attr):
        etiquetas = [self.etiquetas[i] for i in lista_attr]
        conteo_etiquetas = [etiquetas.count(x) / len(lista_attr) for x in set(etiquetas)]
        gini = 1 - sum([(conteo ** 2) for conteo in conteo_etiquetas])
        return gini

    def calcular_ganancia_gini(self, lista_attr, id_caracteristica):
        gini = self.calcular_gini(lista_attr)
        caracteristicas = [self.X[x][id_caracteristica] for x in lista_attr]
        valores_caracteristicas = list(set(caracteristicas))
        conteo_valores_caracteristicas = [caracteristicas.count(x) for x in valores_caracteristicas]
        attr_valores_caracteristicas = [[lista_attr[i] for i, x in enumerate(caracteristicas) if x == y] for y in
                                        valores_caracteristicas]
        ganancia_gini = gini - sum(
            [(conteo_valores / len(lista_attr)) * self.calcular_gini(ids_valores) for conteo_valores, ids_valores in
             zip(conteo_valores_caracteristicas, attr_valores_caracteristicas)])
        return ganancia_gini

    def obtener_maxima_ganancia_gini(self, attr_muestras, attr_atributos):
        gini_atributos = [self.calcular_ganancia_gini(attr_muestras, id_atributo) for id_atributo in attr_atributos]
        id_maximo = attr_atributos[gini_atributos.index(max(gini_atributos))]
        return self.nombres_atributos[id_maximo], id_maximo

    def Gini(self):
        lista_attr = [x for x in range(len(self.X))]
        caracteristicas_attr = [x for x in range(len(self.nombres_atributos))]
        self.nodo = self.GiniRecursive(lista_attr, caracteristicas_attr, self.nodo)
        #print('')

    def GiniRecursive(self, lista_attr, caracteristicas_attr, nodo):
        if not nodo:
            nodo = Node()  # initialize nodes

        attribute = [self.etiquetas[x] for x in lista_attr]

        # StoppingCriterion
        if len(set(attribute)) == 1:
            nodo.value = self.etiquetas[lista_attr[0]]
            return nodo
        if len(caracteristicas_attr) == 0:
            nodo.value = max(set(attribute), key=attribute.count)  # compute mode
            return nodo

        best_feature_name, best_feature_id = self.obtener_maxima_ganancia_gini(lista_attr, caracteristicas_attr)
        nodo.value = best_feature_name
        nodo.childs = []

        valors_caracteristiques = list(set([self.X[x][best_feature_id] for x in lista_attr]))

        for value in valors_caracteristiques:
            child = Node()
            child.value = value
            nodo.childs.append(child)
            child_x_ids = [x for x in lista_attr if self.X[x][best_feature_id] == value]
            if not child_x_ids:
                child.next = max(set(attribute), key=attribute.count)
                #print('')
            else:
                if caracteristicas_attr and best_feature_id in caracteristicas_attr:
                    to_remove = caracteristicas_attr.index(best_feature_id)
                    caracteristicas_attr.pop(to_remove)
                # recursively call the algorithm
                child.next = self.GiniRecursive(child_x_ids, caracteristicas_attr, child.next)
        return nodo


