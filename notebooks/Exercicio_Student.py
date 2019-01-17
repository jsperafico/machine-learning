#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'machine-learning/notebooks'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# ** Importar o pandas as pd.**
#%%
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# ** Salvar student-por.csv como um dataframe chamado df_student.**
#%%
df_student = pd.read_csv('../data/student/student-por.csv', sep=';')

#%% [markdown]
# **Visualizar o cabeçalho do DataFrame.**
#%%
df_student.head()

#%% [markdown]
# **Visualizar dados ('School', 'sex', 'traveltime', 'studytime', 'absences', 'GRegular', 'G3')**
#%%
df_student['GRegular'] = (df_student['G1'] + df_student['G2'])/2
df_student[['sex', 'traveltime', 'studytime', 'absences', 'GRegular', 'G3']].head()

#%%
df_student[['sex', 'traveltime', 'studytime', 'absences', 'GRegular', 'G3']].dtypes

#%%
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
df_student['tp_sex'] = labelEncoder.fit_transform(y = df_student['sex'].astype(str))
df_student['sex'] = df_student['sex'].astype('category')

#%%
df_student[['tp_sex', 'sex', 'traveltime', 'studytime', 'absences', 'GRegular', 'G3']].dtypes

#%% [markdown]
# **Categorizar entre M|F e abstenções maior ou igual à média**
#%%
df_absence = df_student[['tp_sex', 'sex', 'traveltime', 'studytime', 'absences']]
df_absence = df_absence[df_student['absences'] >= df_student['absences'].mean()]
df_absence.head()

#%% [markdown]
# **Plotando dados**
#%%
from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
x = df_absence[['traveltime', 'studytime', 'absences']]
y = df_absence['tp_sex']
cmap = cm.get_cmap('gnuplot')
scatter = pd.plotting.scatter_matrix(x, c=y, marker='o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

#%% [markdown]
# Separando os dados em treino e teste:
#%% 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

#%% [markdown]
# Treinar o classificador usando os dados de treino:
#%% 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(x_train, y_train)

#%% [markdown]
# Estimar a acurácia do classificador em dados futuros usando os dados de teste:
#%% 
knn.score(x_test, y_test)

#%% [markdown]
# Usando o classificador para classificar um objeto.
#%%
absences_prediction = knn.predict([[1.5, 2, 1]])
lookup_absence = dict(zip(df_absence['tp_sex'].unique(), df_absence['sex'].unique()))
lookup_absence[absences_prediction[0]]

#%% [markdown]
# Quão a acurácia é impactada pelo valor de k?
#%%
k_range = range(1,20)
scores = []

for k in k_range:
	knn = KNeighborsClassifier(n_neighbors = k)
	knn.fit(x_train, y_train)
	scores.append(knn.score(x_test, y_test))
        
plt.figure()        
plt.xlabel('k')
plt.ylabel('Acuracia')
plt.plot(k_range, scores)
plt.xticks([0,5,10,15,20])

#%% [markdown]
# Usando o classificador para classificar um objeto.
#%%
absences_prediction = knn.predict([[1.5, 2, 1]])
lookup_absence = dict(zip(df_absence['tp_sex'].unique(), df_absence['sex'].unique()))
lookup_absence[absences_prediction[0]]