import os
from sklearn.tree import export_graphviz
from train_test import *

def print_tree(bintree):
    # get the training labels
    train_labels = os.listdir(train_path)

    # sort the training labels
    train_labels.sort()

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    export_graphviz(
        bintree,
        out_file="tree.dot",
        class_names=train_labels,
        rounded=True,
        filled=True
    )

if __name__ == "__main__":
    bintree  = DecisionTreeClassifier(random_state=seed)
    train_model(bintree)
    print_tree(bintree)