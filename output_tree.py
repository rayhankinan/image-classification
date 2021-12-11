import os
from sklearn.tree import export_graphviz
from train_test import *

def print_tree(tree):
    # get the training labels
    train_labels = os.listdir(train_path)

    # sort the training labels
    train_labels.sort()

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    export_graphviz(
        tree,
        out_file="tree.dot",
        class_names=train_labels,
        rounded=True,
        filled=True
    )

if __name__ == "__main__":
    tree  = DecisionTreeClassifier(random_state=seed)
    train_model(tree)
    print_tree(tree)