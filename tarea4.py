# Tarea 4 IA
# ppaste - segleisn

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import sys
import time

from multilayer_perceptron.multilayer_perceptron import MultilayerPerceptronClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def read_mat(file_url):
    if not os.path.exists(file_url):
        print "Archivo", file_url, "no existe"
        return None
    return scipy.io.loadmat(file_url)

def generate_matrix(categories, set_type='Train'):
    all_data = []
    all_categories = []
    folder_prefix = 'FC6'
    for category in categories:
        if not os.path.exists(os.path.join(folder_prefix+set_type, category+set_type)):
            print "No existe", os.path.join(folder_prefix+set_type, category+set_type)
        for mat_file in glob.glob(os.path.join(folder_prefix+set_type, category+set_type)+'/*.mat'):
            mat_row = read_mat(mat_file)['stored'][0]
            all_data.append(mat_row)
            all_categories.append(category)

    all_data_array = np.array(all_data)
    all_categories_array = np.array(all_categories)

    return (all_data_array,all_categories_array)

if __name__ == '__main__':
    categories = ['Auditorium', 'bar', 'classroom', 'closet', 'movietheater',
        'restaurant']

    data_train, category_train = generate_matrix(categories, 'Train')
    data_test, category_test = generate_matrix(categories, 'Test')

    perceptron = MultilayerPerceptronClassifier( \
          hidden_layer_sizes = (500,), \
          max_iter = 6000, \
        #   random_state = 123,
          algorithm = 'sgd',
          batch_size = 200,
          shuffle = True,
          alpha = 0.00002,\
          learning_rate_init = 0.0001, \
          verbose = True,)

    category_predicted = perceptron.fit(data_train, category_train).predict(data_test)

    accuracy = accuracy_score(category_test, category_predicted)

    cm = confusion_matrix(category_test, category_predicted)
    # cm_title = 'Confusion matrix, without normalization ('+selected_set_name+', '+perceptr_name+')'
    # print(cm_title)
    # print(cm)
    # plt.figure()
    # plot_confusion_matrix(cm, cm_title, ticks=tuple(actions))

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # cm_normalized_title = 'Normalized confusion matrix ('+selected_set_name+', '+perceptr_name+') '+accuracy
    # print cm_normalized_title
    # print(cm_normalized)
    # plt.figure()
    # plot_confusion_matrix(cm_normalized, cm_normalized_title, ticks=tuple(actions))

    print 'Accuracy: ' + str(accuracy)

    # plt.savefig('cm.png', bbox_inches='tight')
    # plt.close()
