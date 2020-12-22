import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv("entradas_breast.csv")
classes = pd.read_csv("saidas_breast.csv")

                        #estou usando loos escrito de maneira diferente para alcancar no compile
def criarRede(optimizer, loos, kernel_initializer, activation, neurons):
    
    classificador = Sequential()
    #1 primeira camada oculta e 30 entradas
    classificador.add(Dense(units= neurons, activation = activation,
                            kernel_initializer = kernel_initializer, input_dim = 30))
    #Dropout de primeira camada oculta
    classificador.add(Dropout(0.2))
    
    #2 segunda camada oculta
    classificador.add(Dense(units = neurons, activation = activation,
                            kernel_initializer = kernel_initializer))
    #Dropout de segunda camada oculta
    classificador.add(Dropout(0.2))
    #3 terceira camada oculta
    classificador.add(Dense(units = neurons, activation = activation,
                            kernel_initializer = kernel_initializer))
    #Dropout de terceira camada oculta
    classificador.add(Dropout(0.2))
    
    #ultima camada de saida 1 output              saida binaria
    classificador.add(Dense(units=1, activation = "sigmoid"))
    
                                               #aqui esta a saida loos"
    classificador.compile(optimizer = optimizer, loss = loos,
                          metrics=["binary_accuracy"])
    return classificador


classificador = KerasClassifier(build_fn = criarRede )

parametros = {"batch_size": [10,30],
              "epochs": [100, 150],
              "optimizer": ["adam","sgd" ],
              #mais uma vez referenciando loos errado que seria da camada inicial
              "loos" : ["binary_crosentropy", "hinge"],
              "kernel_initializer" : ["random_uniform","normal"],
              "activation": ["relu","tanh"],
              "neurons" : [16, 8]             
              }

grid_search = GridSearchCV(estimator= classificador, 
                           param_grid= parametros,
                           scoring = "accuracy",
                           cv = 7)

grid_search = grid_search.fit(previsores,classes)

melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_



