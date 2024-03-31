import pickle

# Utility function to load data from pickle files
def load_data(): 
    with open('../data/X_train.pkl', 'rb') as file: X_train = pickle.load(file)
    with open('../data/y_train.pkl', 'rb') as file: y_train = pickle.load(file)
    with open('../data/X_test.pkl', 'rb') as file: X_test = pickle.load(file)
    with open('../data/y_test.pkl', 'rb') as file: y_test = pickle.load(file)
    return X_train, y_train, X_test, y_test
