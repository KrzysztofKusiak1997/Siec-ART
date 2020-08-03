# -*- coding: utf-8 -*-
import numpy as np

def train_ART(v, w, win_neuron, n, X):

    #inicjalizacja zmiennych tymczasowych
    previous_v = np.zeros(n)
    previous_w = np.zeros(n)
    sum_new_v = 0
    
    
    
    for i in range(0, n):
        steps_v = 1
        while steps_v > 0.000001:
            
            previous_v[i] = v[i][win_neuron]
            
            #aktualizacja wag
            v[i][win_neuron] = v[i][win_neuron]*X[i]
            sum_new_v += v[i][win_neuron]*X[i]
                    
            #obliczenie, jaki krok wykonano
            steps_v = abs(v[i][win_neuron]-previous_v[i])
            
                          
    print('zaktualizowano wagi v')
        
    for i in range(0, n):
        steps_w= 1
        while steps_w > 0.000001:
            
            previous_w[i] = w[i][win_neuron]
            
            #aktualizacja wag
            w[i][win_neuron] = v[i][win_neuron]/(0.5+sum_new_v)
            
            #obliczenie, jaki krok wykonano
            steps_w = abs(w[i][win_neuron]-previous_w[i])
         
    print('zaktualizowano wagi w')
    
    return v, w