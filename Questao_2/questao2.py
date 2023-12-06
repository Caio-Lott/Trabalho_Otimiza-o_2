"""
Questão 2 - Trabalho Prático de Otimização 2

Descrição

Este script realiza a simulação de uma cadeia de Markov baseada em uma matriz de transição pré-definida.
O processo começa no estado 1 e simula a cadeia até que um estado absorvente (0 ou 4) seja atingido ou até
um limite máximo de 100 transições. A simulação é repetida 1000 vezes para estimar as probabilidades
de atingir cada um dos estados absorventes.

Integrantes do grupo

Caio Fernandes Lott Primola - 20193001742
Henrique Rodrigues Lima - 20193009473
Tarcisio Batista Prates - 20193008761

"""

import numpy as np

# Definindo a matriz de transição da cadeia de Markov
matriz_transicao = np.array([
    [1,   0,   0,   0,   0],
    [2/3, 0, 1/3,   0,   0],
    [0, 2/3,   0, 1/3,   0],
    [0,   0, 2/3,   0, 1/3],
    [0,   0,   0,   0,   1]
])

def cadeia_markov(matriz_transicao, estado_inicial, transicoes):
    """
    Simula uma única cadeia de Markov até atingir um estado absorvente ou o limite de transições.
    
    matriz_transicao: Matriz de transição da cadeia de Markov.
    estado_inicial: Estado inicial da cadeia.
    transicoes: Número máximo de transições.
    :return: Estado final da cadeia de Markov.
    """
    estado_atual = estado_inicial
    for _ in range(transicoes):
        if estado_atual in [0, 4]:  # Verifica se o estado atual é absorvente
            break
        estado_atual = np.random.choice(range(5), p=matriz_transicao[estado_atual])
    return estado_atual

def calcula_probabilidade_absorventes(matriz_transicao, estado_inicial, transicoes, simulaçoes):
    """
    Calcula as probabilidades de atingir cada estado absorvente em várias simulações.
    
    matriz_transicao: Matriz de transição da cadeia de Markov.
    estado_inicial: Estado inicial da cadeia.
    transicoes: Número máximo de transições permitidas por simulação.
    num_simulations: Número de simulações a serem executadas.
    :return: Probabilidades de atingir cada estado absorvente.
    """
    resultado = np.array([cadeia_markov(matriz_transicao, estado_inicial, transicoes) for _ in range(simulaçoes)])
    prob_to_0 = np.mean(resultado == 0)
    prob_to_4 = np.mean(resultado == 4)
    return prob_to_0, prob_to_4

# Parâmetros da simulação
estado_inicial = 1
transicoes = 100
simulaçoes = 1000

# Executando as simulações e calculando as probabilidades
probabilidade_estado_0, probabilidade_estado_4 = calcula_probabilidade_absorventes(matriz_transicao, estado_inicial, transicoes, simulaçoes)

# Imprimindo os resultados no terminal
print(f"Probabilidade estimada de atingir o estado absorvente 0: {probabilidade_estado_0:.2%}")
print(f"Probabilidade estimada de atingir o estado absorvente 4: {probabilidade_estado_4:.2%}")
