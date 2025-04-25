

import pickle

def testing_with_svm_best_model( df, model  ):
    
    # Load the model
    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)