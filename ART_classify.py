# -*- coding: utf-8 -*-
import numpy as np
from train_ART import train_ART

def ART_classify(image, y, y_number, classes, n, w, v, prog_czujnosci):
    
    #zamiana obrazu na wektor wejsciowy
    X = image.ravel()
    
    #suma x razy w
    suma_x_w = np.zeros([y_number])
    
    for j in range(0, y_number):
        for i in range(0, n):
            suma_x_w[j] += X[i]*w[i][j]
    
    #przyporzadkowanie do klasy
    condition = True
    counter = 0
    assigned = False
    while condition and counter<10 :
    
        #wybranie neuronu o najwiekszej sumie
        win_neuron = suma_x_w.argmax()
        y[win_neuron] = 1
        
        #sprawdzenie poziomu podobienstwa do klasy
        suma_x_v = 0
        suma_x =0
        
        for i in range(0, n):
            suma_x_v += X[i]*v[i][win_neuron]
            suma_x += X[i]
            
        D = suma_x_v/suma_x
        print(D)
        print()
        if D > prog_czujnosci:
            print('przyporzadkowano do klasy %d', win_neuron)
            condition = False
            assigned = True
            
            v, w = train_ART(v, w, win_neuron, n, X)
            
        else:
            y[win_neuron] = 0
            np.delete(suma_x_w, win_neuron)
            counter += 1
    
    #sprawdzenie, czy wszystkie klasy są zajete
    if assigned == False and classes.all():
        print('Nie rozpoznano')
    
    #sprawdzenie, czy jest nowa klasa oraz tworzenie nowej klasy
    if assigned == False and not classes.all():
        free_classes = np.argwhere(classes == 0)
        new_class = free_classes[0]
        classes[new_class] = 1
        
        v, w = train_ART(v, w, new_class, n, X)
        
        print('stworzono nowa klase')
    
    return v, w, classes