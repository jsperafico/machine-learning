#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'machine-learning/notebooks'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# <p style="font-family: Arial; font-size:16pt;color:Darkblue; font-style:bold"><br>
# Tarefa de Classificação (Frutas)
# <br>
# 
#%% [markdown]
# <p style="font-family: Arial; font-size:14pt;color:#2462C0; font-style:bold"><br>
# 
# Importação das bibliotecas
# </p>
# 
# Documentação das bibliotecas:
#%%
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
fruits = pd.read_table('../data/samples/fruit_data_with_colors.txt')
#%%
fruits.head()
#%%
fruits.shape

#%% [markdown]
# <p style="font-family: Arial; font-size:14pt;color:#2462C0; font-style:bold"><br>
# 
# Examinando os dados:
# </p>
#%%
from matplotlib import cm
x = fruits[['mass', 'width', 'height', 'color_score']]
y = fruits['fruit_label']
cmap = cm.get_cmap('gnuplot')
scatter = pd.plotting.scatter_matrix(x, c=y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

#%% [markdown]
# <p style="font-family: Arial; font-size:14pt;color:#2462C0; font-style:bold"><br>
# 
# Plotando atributos em 3 dimensões:
# </p>
#%%
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x['width'], x['height'], x['color_score'], c=y, marker='o', s=100)
ax.set_xlabel('width')
ax.set_ylabel('heigth')
ax.set_zlabel('color_score')
plt.show()

#%% [markdown]
# <p style="font-family: Arial; font-size:14pt;color:#2462C0; font-style:bold"><br>
# 
# Separando os dados em treino e teste:
# </p>
#%%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
#%%
# Criação de dicionário para facilitar a leitura dos dados de saída
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
lookup_fruit_name

#%% [markdown]
# <p style="font-family: Arial; font-size:14pt;color:#2462C0; font-style:bold"><br>
# 
# Criando um classificador com K-NN:
# </p>
# 
# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# 
# 
#%%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

#%% [markdown]
# <p style="font-family: Arial; font-size:14pt;color:#2462C0; font-style:bold"><br>
# 
# Treinar o classificador usando os dados de treino.
# </p>
#%%
knn.fit(X_train, y_train)

#%% [markdown]
# <p style="font-family: Arial; font-size:14pt;color:#2462C0; font-style:bold"><br>
# 
# Estimar a acurácia do classificador em dados futuros usando os dados de teste.
# </p>
# 
# Acurácia é a quantidade de objetos do conjunto de teste cujo label correto foi previsto pelo classificador.
#%%
knn.score(X_test, y_test)

#%% [markdown]
# Usando o classificador para classificar um objeto.
#%%
fruit_prediction = knn.predict([[20, 4.3, 5.5, 0.7]])
lookup_fruit_name[fruit_prediction[0]]


#%%
fruit_prediction = knn.predict([[100, 6.3, 8.5, 0.5]])
lookup_fruit_name[fruit_prediction[0]]

#%% [markdown]
# Quão a acurácia é impactada pelo valor de k?

#%%
k_range = range(1,20)
scores = []

for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
        
plt.figure()        
plt.xlabel('k')
plt.ylabel('Acuracia')
plt.plot(k_range, scores)
plt.xticks([0,5,10,15,20])


