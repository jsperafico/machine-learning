#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'machine-learning/notebooks'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Exercícios Pré-Processamento
# 
# Iremos realizar um rápido exercício utilizando a biblioteca pandas, usando o dataset de salários da cidade de São Francisco do Kaggle [Salaries Dataset](https://www.kaggle.com/kaggle/sf-salaries).
#%% [markdown]
# ** Importar o pandas as pd.**

#%%
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# ** Salvar Salaries.csv como um dataframe chamado sal.**

#%%
df_sal = pd.read_csv('../data/samples/Salaries.csv')

#%% [markdown]
# ** Visualizar o cabeçalho do DataFrame. **

#%%
df_sal.head()

#%% [markdown]
# ** Usar o método .info() para visualizar a quantidade de registros.**

#%%
df_sal.info() # 148654 linhas

#%% [markdown]
# **Qual a média do salário base (BasePay) ?**

#%%
df_sal['BasePay'].mean()

#%% [markdown]
# **Distribuição dos salários (BasePay) ?**

#%%
df_sal.hist(column='BasePay')

#%% [markdown]
# ** Qual o maior valor pago de hora extra no dataset ? **

#%%
df_sal['OvertimePay'].max()

#%% [markdown]
# ** Qual o cargo de JOSEPH DRISCOLL ? Observação: Use todas as letras maiúsculas ou você pode receber uma resposta que não fecha (Existe um Joseph Driscoll com letras minúsculas). **

#%%
df_sal[df_sal['EmployeeName']=='JOSEPH DRISCOLL']['JobTitle']

#%% [markdown]
# ** Quanto JOSEPH DRISCOLL ganha (incluindo benefícios)? **

#%%
df_sal[df_sal['EmployeeName']=='JOSEPH DRISCOLL']['TotalPayBenefits']

#%% [markdown]
# ** Qual o nome da pessoa com maior salário (incluindo benefícios)?**

#%%
df_sal[df_sal['TotalPayBenefits']== df_sal['TotalPayBenefits'].max()] 

#%% [markdown]
# ** Qual a média de salário base dos empregados por ano (2011-2014) ? **

#%%
df_sal.groupby('Year').mean()['BasePay']

#%% [markdown]
# ** Quantos cargos distintos há?**

#%%
df_sal['JobTitle'].nunique()

#%% [markdown]
# ** Quais os 5 cargos mais comuns? **

#%%
df_sal['JobTitle'].value_counts().head(5)

#%% [markdown]
# ** Há correlação entre o tamanho do nome do cargo e o salário? **

#%%
df_sal['title_len'] = df_sal['JobTitle'].apply(len)


#%%
df_sal[['title_len','TotalPayBenefits']].corr() # Não há correlação

#%% [markdown]
# # Bom Trabalho!

