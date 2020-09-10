import streamlit as st
import pandas as pd

# st.write("""
# # Fazedor de Luas
# """)

# st.markdown(" $$\displaystyle\int_a^bf(x)dx = F(b)-F(a)$$ yeah ")

# n = st.slider('Tamanho amostral', 0, 1000, 250)

# bagunca = st.slider('Bagunça nos dados', 0, 1000, 200) /1000

# from sklearn import datasets

# X, Y = datasets.make_moons(n_samples = n, noise = bagunca)

# import matplotlib.pyplot as plt

# plt.scatter(X[:, 0], X[:, 1], c = Y, marker = 's', alpha = 0.5, cmap = 'Spectral')
# plt.axis('off')
# plt.show()
# st.pyplot()

import pandas as pd 
link = 'seguros.csv' 
dados = pd.read_csv(link)  

st.write("""
# Previsão dos Custos com um Plano de Saúde
""")
   
st.write(""" ## Amostra do conjunto de dados utilizado na modelagem:""") 
st.write(dados.sample(7))

st.sidebar.title('Entre com as informações do indivíduo para alimentar o modelo de machine learning:')
 

idade = st.sidebar.slider('Idade', 18, 65, 30)
sexo = st.sidebar.selectbox("Sexo", ['Masculino', 'Feminino'])
imc = st.sidebar.slider('Índice de Massa Corporal', 15, 54, 24)
criancas = st.sidebar.selectbox("Quantidade de filhos", [0, 1, 2, 3, 4, 5])
fumante = st.sidebar.selectbox("É fumante?", ['Sim', 'Nao'])
regiao = st.sidebar.selectbox("Região em que mora", ['Sudeste', 'Noroeste', 'Sudoeste', 'Nordeste'])


st.write(""" ## Dados de entrada para inferência do modelo:""")
indiv = pd.DataFrame({'idade': [idade], 'sexo': [sexo.lower()], 'imc': [imc],
					  'criancas': [criancas], 'fumante': [fumante.lower()], 'regiao': [regiao.lower()] }) 


st.write(indiv)


aux = pd.concat([indiv, dados.iloc[[12, 1, 3, 8], :]])
 



def Preprocessamento(dados, novos = False):
	dados['sexo'] = dados['sexo'].map({'masculino': 1, 'feminino': 0})
	dados['fumante'] = dados['fumante'].map({'sim': 1, 'nao': 0})
	dados = pd.get_dummies(dados, columns = ['regiao'], drop_first = True)
	if novos:
		dados = dados.loc[0,:].to_frame().T.drop(['custos'], axis = 1)
	return dados

ind = Preprocessamento(aux, novos = True) 
dados = Preprocessamento(dados, novos = False) 

from sklearn.ensemble import RandomForestRegressor


X = dados.drop('custos', axis = 1)
Y = dados['custos']

parametros = {'max_depth': 4, 'max_features': 'auto', 'n_estimators': 500}
modelo = RandomForestRegressor(**parametros, random_state = 42)
modelo.fit(X,Y)
p = modelo.predict(ind)
s = '$' + str(p.round(2)[0])


st.write(""" ## **Previsão de Custos pelo modelo Random Forest:**""")

if st.button('Calcular a previsão de custos'):
	st.write(s)
