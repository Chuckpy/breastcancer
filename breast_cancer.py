import pandas as pd

previsores = pd.read_csv("entradas_breast.csv")
classe = pd.read_csv("saidas_breast.csv")

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,classe, test_size= 0.25)

import keras

from keras.models import Sequential
from keras.layers import Dense

#sequential que da a propriedade de sequencias de camadas de neuronios
classificador = Sequential()
#add - primeira camada oculta com 16 neuronios e 30 entradas, que são os 30 inputs iniciais
classificador.add(Dense(units = 16, activation= "relu",
                        kernel_initializer = "random_uniform", input_dim = 30 ))
#add - segunda camada oculta, ambas com função relu contanto que a segunda não tem input
classificador.add(Dense(units = 16, activation= "relu",
                        kernel_initializer = "random_uniform" ))
#add - neuronio de saida que tem função sigmoid para resposta binaria
classificador.add(Dense(units= 1, activation ="sigmoid"))

otimizador = keras.optimizers.Adam(lr = 0.00001, decay = 0.000001, clipvalue = 0.3)

classificador.compile(optimizer = otimizador, loss = "binary_crossentropy",
                    metrics = ["binary_accuracy"])

#otimizador "adam" que responde melhor aos testes


#classificador.compile(optimizer= "adam", loss = "binary_crossentropy",
#                    metrics = ["binary_accuracy"])

classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 10, epochs= 100)

pesos0 = classificador.layers[0].get_weights()

pesos1 = classificador.layers[1].get_weights()

pesos2 = classificador.layers[2].get_weights()


print(pesos0)

print(len(pesos0))  





previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score

precisao = accuracy_score (classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

resultado = classificador.evaluate(previsores_teste, classe_teste)


    