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
# ** Salvar student-por.csv como um dataframe chamado student.**

#%%
df_student = pd.read_csv('../data/student/student-por.csv', sep=';')
#%% [markdown]
# **Visualizar o cabeçalho do DataFrame.**

#%%
df_student.head()