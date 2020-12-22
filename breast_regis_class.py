import pandas as pd
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Dropout


previsores = pd.read_csv("entradas_breast.csv")
classe =  pd.read_csv("saidas_breast.csv")

classificador = Sequential()

    #primeira camada oculta
classificador.add(Dense(units = 8, activation= "relu",
                        kernel_initializer = "normal", input_dim = 30 ))    
 #Dropout para zerar aleatoriamente uma porcentagem de entrada 
classificador.add(Dropout(0.2))

    #segunda camada oculta
classificador.add(Dense(units = 8, activation= "relu",
                        kernel_initializer = "normal" ))
    #segundo Dropout feito em camada oculta
classificador.add(Dropout(0.2))

    #terceira camada oculta
classificador.add(Dense(units = 8, activation= "relu",
                        kernel_initializer = "normal" ))
    #output camada de saida
classificador.add(Dense(units= 1, activation ="sigmoid"))

classificador.compile(optimizer= "adam", loss = "binary_crossentropy",
                    metrics = ["binary_accuracy"])

classificador.fit(previsores,classe, batch_size=(10), epochs= 150)

novo = np.array([[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,
                  0.1471,0.2419,0.07871,1095,0.9053,8589,153.4,
                  0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,
                  25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,
                  0.4601,0.1189]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.5)
