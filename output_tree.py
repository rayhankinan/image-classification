import os
from sklearn.tree import export_graphviz
from pydot import graph_from_dot_data
from train_test import *

def print_tree(tree):
    train_labels = os.listdir(train_path)

    train_labels.sort()

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    graph = graph_from_dot_data(export_graphviz(tree, class_names=train_labels, rounded=True, filled=True))[0]
    graph.write_jpg('tree.jpg')