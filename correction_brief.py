# -*- coding: utf-8 -*-
"""
Example script

Script to perform some corrections in the brief audio project

Created on Fri Jan 27 09:08:40 2023

@author: ValBaron10
"""

# Import
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from features_functions import compute_features

from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlxtend.classifier import StackingClassifier
from sklearn.neural_network import MLPClassifier


Loading = True

if Loading == False :
    # Set the paths to the files 
    data_path = "Data/"

    # Names of the classes
    classes_paths = ["Cars/", "Trucks/"]
    classes_names = ["car", "truck"]
    cars_list = [4,5,7,9,10,15,20,21,23,26,30,38,39,44,46,48,51,52,53,57]
    trucks_list = [2,4,10,11,13,20,22,25,27,30,31,32,33,35,36,39,40,45,47,48]
    nbr_of_sigs = 20 # Nbr of sigs in each class
    seq_length = 0.2 # Nbr of second of signal for one sequence
    nbr_of_obs = int(nbr_of_sigs*10/seq_length) # Each signal is 10 s long

    # Go to search for the files
    learning_labels = []
    for i in range(2*nbr_of_sigs):
        if i < nbr_of_sigs:
            name = f"{classes_names[0]}{cars_list[i]}.wav"
            class_path = classes_paths[0]
        else:
            name = f"{classes_names[1]}{trucks_list[i - nbr_of_sigs]}.wav"
            class_path = classes_paths[1]

        # Read the data and scale them between -1 and 1
        fs, data = sio.wavfile.read(data_path + class_path + name)
        data = data.astype(float)
        data = data/32768

        # Cut the data into sequences (we take off the last bits)
        data_length = data.shape[0]
        nbr_blocks = int((data_length/fs)/seq_length)
        seqs = data[:int(nbr_blocks*seq_length*fs)].reshape((nbr_blocks, int(seq_length*fs)))

        for k_seq, seq in enumerate(seqs):
            # Compute the signal in three domains
            sig_sq = seq**2
            sig_t = seq / np.sqrt(sig_sq.sum())
            sig_f = np.absolute(np.fft.fft(sig_t))
            sig_c = np.absolute(np.fft.fft(sig_f))

            # Compute the features and store them
            features_list = []
            N_feat, features_list = compute_features(sig_t, sig_f[:sig_t.shape[0]//2], sig_c[:sig_t.shape[0]//2], fs)
            features_vector = np.array(features_list)[np.newaxis,:]

            if k_seq == 0 and i == 0:
                learning_features = features_vector
                learning_labels.append(classes_names[0])
            elif i < nbr_of_sigs:
                learning_features = np.vstack((learning_features, features_vector))
                learning_labels.append(classes_names[0])
            else:
                learning_features = np.vstack((learning_features, features_vector))
                learning_labels.append(classes_names[1])

    print(learning_features.shape)
    print(len(learning_labels))
    
    pickle.dump(learning_features, open("learning_features", "wb"))
    pickle.dump(learning_labels, open("learning_labels","wb"))
     
else :
    learning_features = pickle.load(open("learning_features", "rb"))
    learning_labels = pickle.load(open("learning_labels", "rb"))
    # Separate data in train and test
    X_train, X_test, y_train, y_test = train_test_split(learning_features, learning_labels, test_size=0.2, random_state=42)

    # Standardize the labels
    labelEncoder = preprocessing.LabelEncoder().fit(y_train)
    learningLabelsStd = labelEncoder.transform(y_train)
    testLabelsStd = labelEncoder.transform(y_test)

    #mlpclassifer

    model = svm.SVC(C=100, kernel='rbf', class_weight=None, probability=False)
    scaler = preprocessing.StandardScaler(with_mean=True).fit(X_train)
    learningFeatures_scaled = scaler.transform(X_train)

    model.fit(learningFeatures_scaled, learningLabelsStd)

    # Obtenez les scores de performance pour les données d'entraînement et de validation
    train_sizes, train_scores, test_scores = learning_curve(model, learningFeatures_scaled, learningLabelsStd, cv=10)

    # Tracer les courbes de performance
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Performance sur les données d\'entraînement')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='Performance sur les données de validation')

    # Ajouter des étiquettes et un titre au graphique
    plt.xlabel("Taille de l'ensemble d'entraînement")
    plt.ylabel("Score")
    plt.title("Courbes de performance")

    # Afficher la légende
    plt.legend(loc='best')

    # Afficher le graphique
    plt.show()
    
    testFeatures_scaled = scaler.transform(X_test)
    
    # Compute predictions
    y_pred = model.predict(testFeatures_scaled)

    # Compute and print the classification report
    cr = classification_report(testLabelsStd, y_pred)
    print(cr)
    
    # # Test the model
    # testFeatures_scaled = scaler.transform(X_test)

    # # Compute predictions
    # y_pred = model.predict(testFeatures_scaled)

    # # Compute and print the classification report
    # cr = classification_report(testLabelsStd, y_pred)
    # print(cr)

    # # Matrix confusion
    # plot_confusion_matrix(model, testFeatures_scaled, testLabelsStd) 
    # plt.show()

    # Learn the model

    #!--------------------------------------------------------------------!#

    # Kneighbor Classifer

    #GridSearch

    """
    model = KNeighborsClassifier(n_neighbors=5)
    scaler = preprocessing.StandardScaler(with_mean=True).fit(X_train)
    learningFeatures_scaled = scaler.transform(X_train)

    param_grid = {'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance']}

    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(learningFeatures_scaled, y_train)

    print("Meilleurs paramètres trouvés : ", grid_search.best_params_)
    print("Meilleure précision : ", grid_search.best_score_)

    """

    #!--------------------------------------------------------------------!#

    #RandomSearch

    """
    model = svm.SVC()
    scaler = preprocessing.StandardScaler(with_mean=True).fit(X_train)
    learningFeatures_scaled = scaler.transform(X_train)

    param_dist = {'C': uniform(loc=1, scale=100),
                'kernel': ['linear', 'rbf', 'sigmoïd'],
                'class_weight': [None, 'balanced']}

    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, cv=5)
    random_search.fit(learningFeatures_scaled, y_train)

    print("Meilleurs paramètres trouvés : ", random_search.best_params_)
    print("Meilleure précision : ", random_search.best_score_)

    """

    #!--------------------------------------------------------------------!#

    #GridSearch
    """
    model = svm.SVC()
    scaler = preprocessing.StandardScaler(with_mean=True).fit(X_train)
    learningFeatures_scaled = scaler.transform(X_train)

    param_grid = {'C': [50, 100],
                'kernel': ['rbf', 'sigmoïd'],
                'class_weight': [None, 'balanced']}


    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(learningFeatures_scaled, y_train)

    print("Meilleurs paramètres trouvés : ", grid_search.best_params_)
    print("Meilleure précision : ", grid_search.best_score_)
    """

    #!--------------------------------------------------------------------!#

    # Model de base
    """
    model = svm.SVC(C=10, kernel='linear', class_weight=None, probability=False)
    scaler = preprocessing.StandardScaler(with_mean=True).fit(X_train)
    learningFeatures_scaled = scaler.transform(X_train)

    model.fit(learningFeatures_scaled, learningLabelsStd)

    # Test the model
    testFeatures_scaled = scaler.transform(X_test)

    # Compute predictions
    y_pred = model.predict(testFeatures_scaled)

    # Compute and print the classification report
    cr = classification_report(testLabelsStd, y_pred)
    print(cr)

    # Matrix confusion
    plot_confusion_matrix(model, testFeatures_scaled, testLabelsStd) 
    plt.show()
    """
    #!--------------------------------------------------------------------!#
    """
    model = svm.SVC(C=37, kernel='rbf', class_weight=None, probability=False)
    scaler = preprocessing.StandardScaler(with_mean=True).fit(X_train)
    learningFeatures_scaled = scaler.transform(X_train)

    model.fit(learningFeatures_scaled, learningLabelsStd)

    # Test the model
    testFeatures_scaled = scaler.transform(X_test)

    # Compute predictions
    y_pred = model.predict(testFeatures_scaled)

    # Compute and print the classification report
    cr = classification_report(testLabelsStd, y_pred)
    print(cr)

    # Matrix confusion
    plot_confusion_matrix(model, testFeatures_scaled, testLabelsStd) 
    plt.show()
    """
    #!--------------------------------------------------------------------!#

    """
    model = RandomForestClassifier(n_estimators=100)
    scaler = preprocessing.StandardScaler(with_mean=True).fit(X_train)
    learningFeatures_scaled = scaler.transform(X_train)

    model.fit(learningFeatures_scaled, learningLabelsStd)

    # Test the model
    testFeatures_scaled = scaler.transform(X_test)

    # Compute predictions
    y_pred = model.predict(testFeatures_scaled)

    # Compute and print the classification report
    cr = classification_report(testLabelsStd, y_pred)
    print(cr)

    # Matrix confusion
    plot_confusion_matrix(model, testFeatures_scaled, testLabelsStd) 
    plt.show()

    """

    # Initialize XGBoost model

    """

    model = xgb.XGBClassifier()

    scaler = preprocessing.StandardScaler(with_mean=True).fit(X_train)
    learningFeatures_scaled = scaler.transform(X_train)

    # Fit the model on the training data
    model.fit(learningFeatures_scaled, learningLabelsStd)
    testFeatures_scaled = scaler.transform(X_test)
    # Make predictions on the test data
    y_pred = model.predict(testFeatures_scaled)

    # Evaluate the model using accuracy score
    print("Accuracy: %.2f%%" % (accuracy_score(testLabelsStd, y_pred) * 100.0))


    print("Confusion Matrix:")
    print(confusion_matrix(testLabelsStd, y_pred))

    """
    """
    # Define base models
    model1 = svm.SVC(C=37, kernel='rbf', class_weight=None, probability=False)
    model2 = RandomForestClassifier(n_estimators=100)
    model3 = KNeighborsClassifier(n_neighbors=5)


    scaler = preprocessing.StandardScaler(with_mean=True).fit(X_train)
    learningFeatures_scaled = scaler.transform(X_train)
    testFeatures_scaled = scaler.transform(X_test)

    # Initialize Stacking Classifier
    sclf = StackingClassifier(classifiers=[model1, model2, model3], 
                            meta_classifier=LogisticRegression())

    # Fit the classifier
    sclf.fit(learningFeatures_scaled, learningLabelsStd)

    # Predict on the test data
    y_pred = sclf.predict(testFeatures_scaled)

    cr = classification_report(testLabelsStd, y_pred)
    print(cr)

    print(confusion_matrix(testLabelsStd, y_pred))
    """