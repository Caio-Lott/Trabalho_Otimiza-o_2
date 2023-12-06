"""
Questão 1 - Trabalho Prático de Otimização 2

Descrição

O código fornecido resolve um jogo representado por uma matriz de prêmios 
e o simula usando tanto uma solução ótima quanto uma solução ligeiramente 
perturbada. Ele calcula e compara os prêmios totais obtidos em múltiplas 
simulações para cada solução. Os resultados são visualizados com um gráfico 
de dispersão, mostrando a solução ótima como uma linha contínua em azul e a 
solução distinta como pontos vermelhos dispersos. 

Integrantes do grupo

Caio Fernandes Lott Primola - 20193001742
Henrique Rodrigues Lima - 20193009473
Tarcisio Batista Prates - 20193008761


"""

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import pandas as pd

def solve_game(prize_matrix):
    # Número de estratégias para cada jogador
    num_strategies_player1 = len(prize_matrix)
    num_strategies_player2 = len(prize_matrix[0])

    # Vetor de coeficientes da função objetivo para minimização
    c = np.zeros(num_strategies_player1 * num_strategies_player2)
    c[:num_strategies_player1] = 1

    # Restrições de probabilidade para as estratégias de cada jogador
    A_eq = np.zeros((num_strategies_player1 + num_strategies_player2, num_strategies_player1 * num_strategies_player2))
    for i in range(num_strategies_player1):
        A_eq[i, i*num_strategies_player2:(i+1)*num_strategies_player2] = 1

    for j in range(num_strategies_player2):
        A_eq[num_strategies_player1+j, j::num_strategies_player2] = 1

    b_eq = np.ones(num_strategies_player1 + num_strategies_player2)

    # Resolvendo o problema de programação linear
    result = linprog(c, method='highs', bounds=(0, None), A_eq=A_eq, b_eq=b_eq)

    # Extraindo as probabilidades e o valor do jogo para cada jogador
    probabilities_player1 = result.x[:num_strategies_player1]
    probabilities_player2 = result.x[num_strategies_player1: num_strategies_player1 + num_strategies_player2]
    game_value = result.fun

    return probabilities_player1, probabilities_player2, -game_value

def simulate_game(probabilities_player1, probabilities_player2, prize_matrix, num_simulations):
    total_prize = 0
    for _ in range(num_simulations):
        strategy_player1 = np.random.choice(len(probabilities_player1), p=probabilities_player1)
        strategy_player2 = np.random.choice(len(probabilities_player2), p=probabilities_player2)
        total_prize += prize_matrix[strategy_player1][strategy_player2]

    return total_prize

# Tabela de prêmios para a questão c(i)
prize_matrix_c = np.array([[3, -1, -3],
                           [-2, 4, -1],
                           [-5, -6, -2]])

# Define the number of simulations
num_simulations = 100

# Lists to store results
optimal_results = []
distinct_results = []

for _ in range(num_simulations):
    # Resolving the game for the table of prizes c(i)
    probabilities_player1_c, probabilities_player2_c, _ = solve_game(prize_matrix_c)

    # Simulating the game 100 times with the optimal solution
    total_prize_optimal = simulate_game(probabilities_player1_c, probabilities_player2_c, prize_matrix_c, 100)

    # Generating a solution slightly different
    probabilities_player1_distinct = probabilities_player1_c + np.random.normal(0, 0.01, len(probabilities_player1_c))
    probabilities_player2_distinct = probabilities_player2_c + np.random.normal(0, 0.01, len(probabilities_player2_c))

    # Ensuring probabilities are non-negative
    probabilities_player1_distinct = np.maximum(0, probabilities_player1_distinct)
    probabilities_player2_distinct = np.maximum(0, probabilities_player2_distinct)

    # Normalizing the probabilities
    probabilities_player1_distinct /= sum(probabilities_player1_distinct)
    probabilities_player2_distinct /= sum(probabilities_player2_distinct)

    # Simulating the game 100 times with the distinct solution
    total_prize_distinct = simulate_game(probabilities_player1_distinct, probabilities_player2_distinct, prize_matrix_c, 100)

    optimal_results.append(total_prize_optimal)
    distinct_results.append(total_prize_distinct)

# Create a single scatter plot for both solutions
plt.figure(figsize=(10, 6))

# Scatter plot for optimal solution (in blue)
plt.plot(range(num_simulations), optimal_results, c='blue', label='Optimal', alpha=1)

# Scatter plot for distinct solution (in red)
plt.scatter(range(num_simulations), distinct_results, c='red', label='Distinct', alpha=0.5)

plt.xlabel('Simulation')
plt.ylabel('Sum of Prizes')
plt.title('Sum of Prizes per Execution (Optimal vs. Distinct)')
plt.legend()

# Save the scatter plot as a PNG file in the root directory
plt.savefig('sum_of_prizes_scatter_plot.png')

plt.show()

results_df = pd.DataFrame({'Simulation': range(1, num_simulations + 1),
                           'Optimal Results': optimal_results,
                           'Distinct Results': distinct_results})

# Save the table as a CSV file in the root directory
results_df.to_csv('results_table.csv', index=False)

print("Tabela com o somatório dos prêmios de ambas soluções:", results_df)
