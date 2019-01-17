#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'machine-learning/notebooks'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# 
# # Exercícios compras no E-commerce
# 
# Neste dataset há alguns fake sobre algumas compras feitas na Amazon.
# (Este dataset retirado do curso Python for Data Science and Machine Learning Bootcamp de autoria de José Portila. Curso disponível na [Udemy](https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/).)
# 

#%%
import pandas as pd


#%%
df_ecom = pd.read_csv('../data/samples/Ecommerce Purchases')

#%% [markdown]
# **Visualizar o cabeçalho do DataFrame.**

#%%
df_ecom.head()

#%% [markdown]
# ** Quantas linhas e colunas há no dataset?**

#%%
df_ecom.info()

#%% [markdown]
# ** Qual a média do valor de compra? **

#%%
df_ecom['Purchase Price'].mean()

#%% [markdown]
# ** Quais o maior e o menor preço? **

#%%
df_ecom['Purchase Price'].max()


#%%
df_ecom['Purchase Price'].min()

#%% [markdown]
# ** Quantas pessoas possuem inglês 'en' como o idioma de escolha no website? **

#%%
df_ecom['Language'].value_counts()['en']


#%% [markdown]
# ** Quantas pessoas possuem o cargo "Lawyer" ? **
# 

#%%
df_ecom['Job'].value_counts()['Lawyer']

#%% [markdown]
# ** Quais os 5 cargos mais comuns? **

#%%
df_ecom['Job'].value_counts().head(5)

#%% [markdown]
# ** Bônus: Quantas pessoas possuem o cartão de crédito American Express (CC Provider) *e* fizeram compras acima de $95 ?**

#%%
df_ecom[(df_ecom['CC Provider']=='American Express') & (df_ecom['Purchase Price']>95)].count()['CC Provider']

#%% [markdown]
# # Bom trabalho!

