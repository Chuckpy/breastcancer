import pandas as pd

import keras 

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv("entradas_breast.csv")
classes = pd.read_csv("saidas_breast.csv")

def criarRede(): 

    classificador = Sequential()

    classificador.add(Dense(units = 16, activation= "relu",
                        kernel_initializer = "random_uniform", input_dim = 30 ))
    
    #Dropout para zerar aleatoriamente uma porcentagem de entrada 
    classificador.add(Dropout(0.15))
    #segunda camada oculta
    classificador.add(Dense(units = 16, activation= "relu",
                        kernel_initializer = "random_uniform" ))
    #segundo Dropout feito em camada oculta
    
    classificador.add(Dropout(0.15))
    #terceira camada oculta
    classificador.add(Dense(units = 16, activation= "relu",
                        kernel_initializer = "random_uniform" ))
    #output camada de saida
    classificador.add(Dense(units= 1, activation ="sigmoid"))

    otimizador = keras.optimizers.Adam(lr = 0.0001, decay = 0.00001, clipvalue = 0.5)

    classificador.compile(optimizer= "adam", loss = "binary_crossentropy",
                    metrics = ["binary_accuracy"])
    return classificador

classificador = KerasClassifier(build_fn= criarRede,
                                epochs = 100,
                                batch_size = 10)

resultados = cross_val_score(estimator=classificador,
                             X=previsores, y= classes,
                             cv=10, scoring= "accuracy") 

media = resultados.mean()
desvios = resultados.std()
print(resultados)