from sklearn.tree import DecisionTreeClassifier
from global_var import *
from output_tree import *
from train_test import *

tree = None

if __name__ == "__main__":
    print("Available Command: ")
    print("1. PREPARE")
    print("2. TRAIN")
    print("3. TEST")
    print("4. PRINT")
    print("5. QUIT")
    print()

    while True:
        try:
            command = input("-> ")

            if command == "PREPARE":
                prepare_data()
            elif command == "TRAIN":
                tree  = DecisionTreeClassifier(random_state=seed)
                train_model(tree)
            elif command == "TEST":
                test_model(tree)
            elif command == "PRINT":
                print_tree(tree)
            elif command == "QUIT":
                break
            else:
                print("INVALID COMMAND")
                
        except:
            print("COMMAND ERROR")