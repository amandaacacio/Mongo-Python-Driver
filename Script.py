from datetime import datetime, tzinfo, timezone
from pymongo import MongoClient
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import os

# Requires the PyMongo package.
# https://api.mongodb.com/python/current

#consulta para recuperar os dados do brasil quando a imunização iniciou (data base) até o último registro
client = MongoClient('mongodb://bigdata-1:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false')
filter={
    'iso_code': 'BRA', 
    'date': {
        '$gte': datetime(2021, 2, 6, 0, 0, 0, tzinfo=timezone.utc)
    }, 
    'people_fully_vaccinated': {
        '$gt': 0
    }
}
project={
    'date': 1, 
    'people_fully_vaccinated': 1, 
    'population': 1
}

result = client['bigdata']['owid-covid-data'].find(
  filter=filter,
  projection=project
)

lista = []

for doc in result:
    lista.append(doc)

df = pd.DataFrame(lista)
print(df)

#primeiro dia de imunização
x = df['date'].loc[~df['people_fully_vaccinated'].isnull()].iloc[0]

#criando os vetores
vetor_1 = [] #Diferença de dias da data de imunização para data base de imunização
vetor_2 = [] #Quantidade imunizada (people_fully_vaccinated)

for i in df['date']:
    quantidade_dias = abs((i-x).days)
    vetor_1.append(quantidade_dias)

for i in df['people_fully_vaccinated']:
    vetor_2.append(i)

#print(vetor_1)
#print(vetor_2)


#regressão linear

X = vetor_1
X = np.reshape(X, (-1,1))
y = vetor_2
y = np.reshape(y, (-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,random_state = 1)

model = LinearRegression().fit(X_train, y_train)

print('R-squared score (training): {:.3f}'.format(model.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(model.score(X_test, y_test)))

#erro quadrado médio
print('Erro Quadrado Médio: {:.4f}'.format(np.mean((model.predict(X_test) - y_test) ** 2)))


pop = 213993441

def reg_d(dias):
    vacinados = model.intercept_ + (model.coef_*dias)
    print(vacinados)

def reg_v(vacinados):
    dias = (vacinados - model.intercept_) / model.coef_
    print(dias)

#1
vacinados = 0.43*pop
print('Dias a partir da data base para os efeitos da imunidade de rebanho com 43% da população imunizada:')
reg_v(vacinados)

#2
vacinados = 0.6*pop 
print('Dias a partir da data base para os efeitos da imunidade de rebanho com 60% da população imunizada:')
reg_v(vacinados)

#3
vacinados = pop 
print('Dias a partir da data base para atingirmos 100% da população imunizada:')
reg_v(vacinados)

#Gráfico

plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, model.predict(X_test), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.title('Imunização até atingir 100% da população')
plt.show()





