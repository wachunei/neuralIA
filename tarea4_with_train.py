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

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, ticks=None):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if ticks:
        plt.yticks(np.arange(0,7), ticks, fontsize=12)
        plt.xticks(np.arange(0,7), ticks, rotation=70, fontsize=9)
    width = len(cm)
    height = len(cm[0])
    for x in xrange(width):
        for y in xrange(height):
            plt.annotate(str(cm[x][y])[:4], xy=(y, x), horizontalalignment='center', verticalalignment='center', fontsize=10)



def read_mat(file_url):
    if not os.path.exists(file_url):
        print "Archivo", file_url, "no existe"
        return None
    return scipy.io.loadmat(file_url)

def category_code(category, no_code = True):
    if no_code:
        return category
    codes = {'Auditorium': [0,0,0],
            'bar': [0,0,1],
            'classroom': [0,1,0],
            'closet': [0,1,1],
            'movietheater': [1,0,0],
            'restaurant': [1,0,1],}
    return codes[category];

def category_decode(code):
    if (code == np.array([0,0,0])).all():
        return 'Auditorium'
    elif (code == np.array([0,0,1])).all():
        return 'bar'
    elif (code == np.array([0,1,0])).all():
        return 'classroom'
    elif (code == np.array([0,1,1])).all():
        return 'closet'
    elif (code == np.array([1,0,0])).all():
        return 'movietheater'
    elif (code == np.array([1,0,1])).all():
        return 'restaurant'
    else:
        return 'other'

def matrix_category_decode(matrix):
    new_matrix = []
    for item in matrix:
        new_matrix.append(category_decode(item))
    return np.array(new_matrix)

def normalize_matrix(matrix):
    new_matrix = []
    for row in matrix:
        row_sum = sum(row)
        if row_sum != 0:
            new_row = [float(item)/row_sum for item in row]
            new_matrix.append(new_row)
        else:
            new_matrix.append(row)
    return new_matrix

def generate_matrix(categories, set_type='Train', no_code=False):
    all_data = []
    all_categories = []
    folder_prefix = 'FC6'
    for category in categories:
        if not os.path.exists(os.path.join(folder_prefix+set_type, category+set_type)):
            print "No existe", os.path.join(folder_prefix+set_type, category+set_type)
        for mat_file in glob.glob(os.path.join(folder_prefix+set_type, category+set_type)+'/*.mat'):
            mat_row = read_mat(mat_file)['stored'][0]
            all_data.append(mat_row)
            all_categories.append(category_code(category, no_code))

    all_data_array = np.array(all_data)
    all_categories_array = np.array(all_categories)

    return (all_data_array,all_categories_array)

if __name__ == '__main__':
    categories = ['Auditorium', 'bar', 'classroom', 'closet', 'movietheater',
        'restaurant', 'other']

    no_code = False

    data_train, category_train = generate_matrix(categories, 'Train', no_code)
    data_test, category_test = generate_matrix(categories, 'Test', no_code = True)

    perceptron = MultilayerPerceptronClassifier( \
          hidden_layer_sizes = (12,), \
          max_iter = 500, \
          algorithm = 'sgd',
          batch_size = 200,
          shuffle = True,
          alpha = 0.00002,\
          learning_rate_init = 0.0001,
          verbose=True)        

    perceptron_model = perceptron.fit(data_train, category_train)
    category_predicted = perceptron_model.predict(data_test)
    # Probamos en el set de entrenamiento para medir sobreajuste
    category_predicted_train = perceptron_model.predict(data_train)
    if not no_code:
        category_predicted = matrix_category_decode(category_predicted)
        category_predicted_train = matrix_category_decode(category_predicted_train)
        category_train = matrix_category_decode(category_train)

    accuracy = accuracy_score(category_test, category_predicted)
    accuracy_train = accuracy_score(category_train, category_predicted_train)

    cm = confusion_matrix(category_test, category_predicted, categories)    
    cm_train = confusion_matrix(category_train, category_predicted_train, categories)

    cm_normalized = normalize_matrix(cm)
    cm_train_normalized = normalize_matrix(cm_train)

    coded_title = 'String coded' if no_code else 'List coded'
    outputs = perceptron.n_outputs_
    cm_normalized_title_base = coded_title + ' outputs: ' +str(outputs) 
    cm_normalized_title = cm_normalized_title_base + ' Acc. sobre test: '+str(accuracy)
    cm_train_normalized_title = cm_normalized_title_base +' Acc. sobre train '+str(accuracy_train)

    plt.figure()
    plot_confusion_matrix(cm_normalized, cm_normalized_title, ticks=(categories))

    print 'Accuracy test: ' + str(accuracy)

    plt.savefig('cm_'+coded_title.lower().replace(" ", "_")+'_'+str(time.time()).replace('.','')+'.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    plot_confusion_matrix(cm_train_normalized, cm_train_normalized_title, ticks=(categories))

    print 'Accuracy train: ' + str(accuracy_train)

    plt.savefig('cm_train_'+coded_title.lower().replace(" ", "_")+'_'+str(time.time()).replace('.','')+'.png', bbox_inches='tight')
    plt.close()
