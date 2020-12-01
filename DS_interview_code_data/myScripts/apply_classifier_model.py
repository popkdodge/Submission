
import itertools
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

from matplotlib import pyplot as plt

def apply_train_test_split(feature_df, test_size):
    '''
    Args: 
        feature_df dataframe to be used for classification
        testsize float value in the range (0, 1), proportion of test dataset
        
    Output:
        splits feature_df into training and testing features and target variables
        (X_train, X_test, y_train, y_test)
    '''
    return train_test_split(feature_df.drop('class', axis = 1), 
                            feature_df['class'], test_size=test_size)

def apply_RF_classifier(X_train, y_train, model_path):
    '''
    Args: 
        X_train dataframe with all the features to be used for training
        y_train series containing labels for each row of X_train
        model_path path where trained random forest model is to be saved
        
    Output:
        trained random forest model
    '''

    RF_model = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)    
    # Fit the training data
    RF_model.fit(X_train, y_train)
    
    pickle_models(RF_model, model_path)
    
    return RF_model

def apply_balanced_RF_classifier(X_train, y_train, model_path):
    '''
    Args: 
        X_train dataframe with all the features to be used for training
        y_train series containing labels for each row of X_train
        model_path path where trained balanced random forest model is to be saved
        
    Output:
        trained balanced random forest model
    '''
    BRF_model = BalancedRandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)    
    # Fit the training data
    BRF_model.fit(X_train, y_train)
    
    pickle_models(BRF_model, model_path)
    
    return BRF_model

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
### Calculate performance in dollar value
def calculate_loss(row):
    '''
    Args: 
        row of data containing actual and predicted labels and purchase value for each transaction
        
    Output:
        loss value for each prediction
    '''
    if (row['actual'] == 0) & (row['predicted']==1):
        row['loss'] = 8.0
    elif (row['actual'] == 1) & (row['predicted']==0):
        row['loss'] = row['purchase_value']
    else:
        row['loss'] = 0
    
    return row

def calculate_total_loss(X_test, y_test, y_pred):
    '''
    Args: 
        X_test test datset
        y_test test labels
        y_pred predicted labels
        
    Output:
        total_loss for the entire dataset
    '''
    loss_df = X_test
    loss_df['actual'] = y_test
    loss_df['predicted'] = y_pred
    loss_df = loss_df.apply(calculate_loss, axis = 1)
    total_loss = loss_df.loss.sum()
    return total_loss

def pickle_models(model, filename):
    '''pickles saved models in the specified path'''
    joblib.dump(model, filename)

def load_models(filename):
    '''loads trained model specified path'''
    return joblib.load(filename)