# import math
# import random

# # --- Parâmetros do Algoritmo Genético ---
# MIN_X = -1.0  # Limite inferior do domínio de x, conforme o tutorial
# MAX_X = 2.0   # Limite superior do domínio de x, conforme o tutorial
# CHROMOSOME_LENGTH = 22 # Tamanho da cadeia de bits para o cromossomo, como no tutorial
# POPULATION_SIZE = 20 # Tamanho da população, como na Tabela 1 do tutorial
# NUM_GENERATIONS = 25 # Número de gerações para a evolução
# CROSSOVER_RATE = 0.8  # Taxa de crossover (80% é um valor comum, pode ajustar)
# MUTATION_RATE = 0.01  # Taxa de mutação (1% é um valor comum, pode ajustar)
# USE_ELITISM = True    # Ativar ou desativar o Elitismo (conforme Seção 1.5)

# # Aptidão para ranqueamento (Seção 1.3, Tabela 1): o 1º recebe 2.0, o último 0.0
# RANK_MAX_FITNESS = 2.0 
# RANK_MIN_FITNESS = 0.0 

# # --- 1. Funções de Codificação e Decodificação do Cromossomo ---

# def encode_x(value):
#     """
#     Codifica um valor real 'x' no intervalo [MIN_X, MAX_X] em uma cadeia de bits.
#     Utiliza a lógica inversa da Equação (1.2) do tutorial para obter o valor inteiro b10.
#     """
#     # Garante que o valor esteja dentro dos limites para evitar erros de mapeamento
#     value = max(MIN_X, min(MAX_X, value))
    
#     # Calcula o valor inteiro b10 correspondente ao valor real x
#     # b10 = (x - min_x) * (2^L - 1) / (max_x - min_x)
#     range_span = MAX_X - MIN_X
#     max_b10_value = (2**CHROMOSOME_LENGTH) - 1
    
#     # Arredonda para o inteiro mais próximo para a codificação binária
#     b10 = round((value - MIN_X) * max_b10_value / range_span)
    
#     # Converte o inteiro para uma string binária de comprimento fixo (preenche com zeros à esquerda)
#     binary_str = bin(int(b10))[2:].zfill(CHROMOSOME_LENGTH)
#     return binary_str

# def decode_chromosome(chromosome_str):
#     """
#     Decodifica uma cadeia de bits (cromossomo) em um valor real 'x'.
#     Baseado na Equação (1.2) do tutorial: x = min + (max - min) * b10 / (2^l - 1).
#     """
#     b10 = int(chromosome_str, 2) # Converte a string binária para um inteiro
#     max_b10_value = (2**CHROMOSOME_LENGTH) - 1 # Valor máximo que b10 pode assumir
    
#     # Aplica a fórmula de mapeamento para obter o valor real x
#     x = MIN_X + (MAX_X - MIN_X) * b10 / max_b10_value
#     return x

# # --- 2. Função Objetivo (Problema 1.1 do Tutorial) ---

# def objective_function(x):
#     """
#     Implementa a função objetivo f(x) = x * sin(10*pi*x) + 1.
#     O objetivo do Algoritmo Genético é maximizar esta função.
#     """
#     return x * math.sin(10 * math.pi * x) + 1

# # --- 3. Geração da População Inicial ---

# def generate_initial_population(size, length, distribution_type="random"):
#     """
#     Gera a população inicial de cromossomos.
    
#     Args:
#         size (int): O número de cromossomos na população.
#         length (int): O comprimento de cada cromossomo (número de bits).
#         distribution_type (str): "random" para cromossomos aleatórios,
#                                  "equidistant" para cromossomos que representam 
#                                  valores de x igualmente espaçados.
    
#     Returns:
#         list: Uma lista de strings binárias, representando a população.
#     """
#     population = []
#     if distribution_type == "random":
#         print("Gerando população inicial aleatória...")
#         for _ in range(size):
#             chromosome = ''.join(random.choice('01') for _ in range(length))
#             population.append(chromosome)
#     elif distribution_type == "equidistant":
#         print("Gerando população inicial equidistante...")
#         if size == 1:
#             x_value = (MIN_X + MAX_X) / 2 # Apenas o ponto central para uma população de 1
#             population.append(encode_x(x_value))
#         else:
#             # Gerar valores de x igualmente espaçados no domínio [MIN_X, MAX_X] e codificá-los
#             step_size = (MAX_X - MIN_X) / (size - 1)
#             for i in range(size):
#                 x_value = MIN_X + i * step_size
#                 population.append(encode_x(x_value))
#     else:
#         raise ValueError("Tipo de distribuição inválido. Use 'random' ou 'equidistant'.")
#     return population

# # --- 4. Avaliação da Aptidão (Fitness) com Ranqueamento ---

# def evaluate_population(population):
#     """
#     Avalia a aptidão de cada cromossomo na população.
#     A aptidão é ranqueada conforme Seção 1.3 e Tabela 1 do tutorial
#     (o melhor recebe RANK_MAX_FITNESS, o pior RANK_MIN_FITNESS, e os demais linearmente).
    
#     Returns:
#         list: Uma lista de dicionários, cada um contendo o cromossomo, 
#               o valor de x decodificado, o valor da função objetivo e a aptidão ranqueada.
#     """
#     evaluated_chromosomes = []
#     for chromosome in population:
#         x_value = decode_chromosome(chromosome)
#         objective_value = objective_function(x_value)
#         evaluated_chromosomes.append({
#             'chromosome': chromosome,
#             'x_value': x_value,
#             'objective_value': objective_value,
#             'fitness': 0 # A aptidão será preenchida após o ranqueamento
#         })
    
#     # Ordena a população pelo valor da função objetivo (descrescente para maximização)
#     # Isso coloca o "mais apto" (melhor valor de f(x)) no início da lista
#     evaluated_chromosomes.sort(key=lambda item: item['objective_value'], reverse=True)
    
#     # Atribui aptidão ranqueada baseada na posição ordenada
#     # fi = RANK_MIN_FITNESS + (RANK_MAX_FITNESS - RANK_MIN_FITNESS) * (N - 1 - i) / (N - 1)
#     N = len(evaluated_chromosomes)
#     for i, item in enumerate(evaluated_chromosomes):
#         if N > 1: # Evita divisão por zero se a população tiver apenas 1 elemento
#             item['fitness'] = RANK_MIN_FITNESS + (RANK_MAX_FITNESS - RANK_MIN_FITNESS) * (N - 1 - i) / (N - 1)
#         else: # Se houver apenas um elemento, ele recebe a aptidão máxima
#             item['fitness'] = RANK_MAX_FITNESS
            
#     return evaluated_chromosomes

# # --- 5. Seleção (Roleta) ---

# def select_parents(evaluated_population):
#     """
#     Seleciona os pais para a próxima geração usando o algoritmo da roleta.
#     Baseado na Figura 4 e Equação (1.3) do tutorial.
#     Os cromossomos com maior aptidão ranqueada têm maior probabilidade de serem selecionados.
#     """
#     total_fitness = sum(item['fitness'] for item in evaluated_population)
    
#     # Se a aptidão total for zero (ex: todos os cromossomos têm aptidão 0), 
#     # seleciona aleatoriamente para evitar divisão por zero ou loop infinito.
#     if total_fitness == 0:
#         return [random.choice([item['chromosome'] for item in evaluated_population]) for _ in range(POPULATION_SIZE)]

#     # Cria uma lista de aptidões acumuladas para a roleta
#     cumulative_fitness = []
#     current_cumulative = 0
#     for item in evaluated_population:
#         current_cumulative += item['fitness']
#         cumulative_fitness.append(current_cumulative)
        
#     selected_parents = []
#     for _ in range(POPULATION_SIZE):
#         # Gera um número aleatório entre 0 e o total_fitness
#         r = random.uniform(0, total_fitness)
        
#         # Encontra o cromossomo correspondente na roleta
#         # Percorre a aptidão acumulada para encontrar o "slot" onde 'r' cai
#         for i, cumulative_val in enumerate(cumulative_fitness):
#             if r <= cumulative_val:
#                 selected_parents.append(evaluated_population[i]['chromosome'])
#                 break
#     return selected_parents

# # --- 6. Crossover (Ponto Único) ---

# def crossover(parent1, parent2):
#     """
#     Realiza o crossover de ponto único entre dois pais.
#     Baseado na Seção 1.4 do tutorial.
#     Aplica-se com uma probabilidade CROSSOVER_RATE.
#     """
#     if random.random() < CROSSOVER_RATE:
#         # Ponto de corte aleatório. O tutorial sugere que não seja no início nem no fim
#         # para que haja troca de material genético.
#         point = random.randint(1, CHROMOSOME_LENGTH - 1) # Garante que o ponto esteja entre 1 e L-1
        
#         # Realiza a troca das "caudas" dos cromossomos
#         child1 = parent1[:point] + parent2[point:]
#         child2 = parent2[:point] + parent1[point:]
#         return child1, child2
#     else:
#         # Se não ocorrer crossover, os filhos são cópias exatas dos pais
#         return parent1, parent2

# # --- 7. Mutação (Inversão de Bit) ---

# def mutate(chromosome):
#     """
#     Realiza a mutação de inversão de bit em um cromossomo.
#     Baseado na Seção 1.4 do tutorial.
#     Cada bit tem uma probabilidade MUTATION_RATE de ser invertido (0 vira 1, 1 vira 0).
#     """
#     mutated_chromosome = list(chromosome) # Converte a string para uma lista de caracteres para poder modificar
#     for i in range(CHROMOSOME_LENGTH):
#         if random.random() < MUTATION_RATE:
#             mutated_chromosome[i] = '1' if mutated_chromosome[i] == '0' else '0' # Inverte o bit
#     return "".join(mutated_chromosome) # Retorna a lista de caracteres como uma string

# # --- Algoritmo Genético Principal ---

# def run_genetic_algorithm(distribution_type="random"):
#     """
#     Executa o algoritmo genético completo.
#     Monitora e imprime o melhor valor da função objetivo e a média da população ao longo das gerações.
#     """    
#     print(f"População: {POPULATION_SIZE}, Gerações: {NUM_GENERATIONS}")
#     print(f"Crossover rate: {CROSSOVER_RATE*100:.0f}%, Mutação rate: {MUTATION_RATE*100:.1f}%")
#     print(f"Elitismo: {'Ativado' if USE_ELITISM else 'Desativado'}, Distribuição inicial: '{distribution_type}'")
    
#     # Geração da população inicial
#     population = generate_initial_population(POPULATION_SIZE, CHROMOSOME_LENGTH, distribution_type)
    
#     # Variáveis para armazenar o melhor indivíduo global encontrado
#     best_overall_individual = None
#     best_overall_objective_value = -float('inf') # Inicializa com valor negativo infinito para maximização

#     # Listas para armazenar dados de desempenho por geração (útil para gráficos)
#     best_objective_per_generation = []
#     avg_objective_per_generation = []

#     print("\n--- Evolução das gerações ---")
#     for generation in range(NUM_GENERATIONS):
#         # 1. Avaliar a população atual
#         evaluated_population = evaluate_population(population)
        
#         # Encontrar o melhor indivíduo da GERAÇÃO ATUAL (baseado no valor da função objetivo)
#         current_best_individual_data = max(evaluated_population, key=lambda item: item['objective_value'])
#         current_best_chromosome = current_best_individual_data['chromosome']
#         current_best_x_value = current_best_individual_data['x_value']
#         current_best_objective = current_best_individual_data['objective_value']
        
#         # Atualizar o melhor indivíduo GLOBAL, se o atual for melhor
#         if current_best_objective > best_overall_objective_value:
#             best_overall_objective_value = current_best_objective
#             best_overall_individual = current_best_chromosome
        
#         # Calcular o valor médio da função objetivo para a geração atual
#         avg_objective = sum(item['objective_value'] for item in evaluated_population) / POPULATION_SIZE
        
#         # Registrar os dados para análise
#         best_objective_per_generation.append(current_best_objective)
#         avg_objective_per_generation.append(avg_objective)

#         # Imprimir o progresso a cada 10 gerações ou na primeira/última
#         if (generation + 1) % 10 == 0 or generation == 0 or generation == NUM_GENERATIONS - 1:
#             print(f"Geração {generation + 1:3d}: Melhor f(x) = {current_best_objective:.6f} (x={current_best_x_value:.6f}), Média f(x) = {avg_objective:.6f}")

#         # 2. Seleção: Escolher pais para a próxima geração
#         parents = select_parents(evaluated_population)
        
#         # Inicializar a próxima população
#         next_population = []
        
#         # 3. Elitismo (Seção 1.5 do tutorial): Transferir o melhor indivíduo para a próxima geração
#         if USE_ELITISM:
#             next_population.append(current_best_individual_data['chromosome']) # Adiciona o melhor da geração atual
        
#         # 4. Crossover e Mutação: Gerar o restante da próxima população
#         # Os pais são escolhidos aleatoriamente da lista 'parents' (que já foi selecionada pela roleta)
#         while len(next_population) < POPULATION_SIZE:
#             # Escolhe dois pais aleatoriamente para reprodução
#             p1 = random.choice(parents)
#             p2 = random.choice(parents)
            
#             # Realiza crossover para gerar filhos
#             child1, child2 = crossover(p1, p2)
            
#             # Aplica mutação em cada filho
#             child1 = mutate(child1)
#             child2 = mutate(child2)
            
#             # Adiciona os filhos à próxima população
#             next_population.append(child1)
#             if len(next_population) < POPULATION_SIZE: # Garante que não exceda o tamanho da população
#                 next_population.append(child2)
        
#         # A população atual se torna a próxima população para a próxima iteração
#         population = next_population

#     # --- Resultados Finais ---
#     print("\n--- Resultados finais da execução ---")
#     if best_overall_individual:
#         best_x_overall = decode_chromosome(best_overall_individual)
#         print(f"Melhor cromossomo global encontrado: {best_overall_individual}")
#         print(f"Valor decodificado de x: {best_x_overall:.6f}")
#         print(f"Valor máximo da função f(x): {best_overall_objective_value:.6f}")
#     else:
#         print("Nenhum indivíduo foi rastreado. Verifique a lógica do algoritmo.")
        
#     # O valor global de máximo para f(x)=x sin(10πx) + 1 no intervalo [-1, 2] é
#     # f(1.85055) = 2.85027, conforme mencionado na Seção 1.1 do tutorial.
#     print(f"\nValor ótimo global esperado (segundo o tutorial): f(1.85055) = 2.85027")

#     return best_objective_per_generation, avg_objective_per_generation

# # --- Execução Principal do Script ---
# if __name__ == "__main__":
#     # Execute o AG com distribuição aleatória inicial
#     best_random_run, avg_random_run = run_genetic_algorithm(distribution_type="random")
    
#     print("\n" + "="*80 + "\n") # Separador para facilitar a visualização
    
#     # Execute o AG com distribuição equidistante inicial
#     best_equidistant_run, avg_equidistant_run = run_genetic_algorithm(distribution_type="equidistant")

#     # --- Opcional: Visualização dos Resultados ---
#     # Para visualizar a evolução do fitness, você pode descomentar o código abaixo
#     # (requer a biblioteca matplotlib).
    
#     import matplotlib.pyplot as plt
    
#     plt.figure(figsize=(14, 7))
#     plt.plot(best_random_run, label='Melhor f(x) (Pop. Aleatória)', color='blue')
#     plt.plot(avg_random_run, label='Média f(x) (Pop. Aleatória)', linestyle='--', color='lightblue')
#     plt.plot(best_equidistant_run, label='Melhor f(x) (Pop. Equidistante)', color='red')
#     plt.plot(avg_equidistant_run, label='Média f(x) (Pop. Equidistante)', linestyle='--', color='salmon')
    
#     # Linha de referência para o máximo global teórico
#     theoretical_max = 2.85027
#     plt.axhline(y=theoretical_max, color='green', linestyle=':', label=f'Máximo global teórico ({theoretical_max:.5f})')
    
#     plt.title('Evolução do valor da função objetivo por geração (AG)')
#     plt.xlabel('Geração')
#     plt.ylabel('Valor de f(x)')
#     plt.grid(True)
#     plt.legend()
#     plt.show()










# import math
# import random
# import matplotlib.pyplot as plt
# import numpy as np

# # --- Parâmetros do Algoritmo Genético ---
# MIN_X = -1.0  # Limite inferior do domínio de x, conforme o tutorial
# MAX_X = 2.0   # Limite superior do domínio de x, conforme o tutorial
# CHROMOSOME_LENGTH = 22 # Tamanho da cadeia de bits para o cromossomo, como no tutorial
# POPULATION_SIZE = 30 # Tamanho da população, como na Tabela 1 do tutorial
# NUM_GENERATIONS = 25 # Número de gerações para a evolução
# CROSSOVER_RATE = 0.8  # Taxa de crossover (80% é um valor comum, pode ajustar)
# MUTATION_RATE = 0.01  # Taxa de mutação (1% é um valor comum, pode ajustar)
# USE_ELITISM = True    # Ativar ou desativar o Elitismo (conforme Seção 1.5)

# # NOVO: Parâmetro para escolher o tipo de crossover
# CROSSOVER_POINTS = 1  # 1 para ponto único, 2 para dois pontos

# # Aptidão para ranqueamento (Seção 1.3, Tabela 1): o 1º recebe 2.0, o último 0.0
# RANK_MAX_FITNESS = 2.0 
# RANK_MIN_FITNESS = 0.0 

# # --- 1. Funções de Codificação e Decodificação do Cromossomo ---

# def encode_x(value):
#     """
#     Codifica um valor real 'x' no intervalo [MIN_X, MAX_X] em uma cadeia de bits.
#     Utiliza a lógica inversa da Equação (1.2) do tutorial para obter o valor inteiro b10.
#     """
#     # Garante que o valor esteja dentro dos limites para evitar erros de mapeamento
#     value = max(MIN_X, min(MAX_X, value))
    
#     # Calcula o valor inteiro b10 correspondente ao valor real x
#     # b10 = (x - min_x) * (2^L - 1) / (max_x - min_x)
#     range_span = MAX_X - MIN_X
#     max_b10_value = (2**CHROMOSOME_LENGTH) - 1
    
#     # Arredonda para o inteiro mais próximo para a codificação binária
#     b10 = round((value - MIN_X) * max_b10_value / range_span)
    
#     # Converte o inteiro para uma string binária de comprimento fixo (preenche com zeros à esquerda)
#     binary_str = bin(int(b10))[2:].zfill(CHROMOSOME_LENGTH)
#     return binary_str

# def decode_chromosome(chromosome_str):
#     """
#     Decodifica uma cadeia de bits (cromossomo) em um valor real 'x'.
#     Baseado na Equação (1.2) do tutorial: x = min + (max - min) * b10 / (2^l - 1).
#     """
#     b10 = int(chromosome_str, 2) # Converte a string binária para um inteiro
#     max_b10_value = (2**CHROMOSOME_LENGTH) - 1 # Valor máximo que b10 pode assumir
    
#     # Aplica a fórmula de mapeamento para obter o valor real x
#     x = MIN_X + (MAX_X - MIN_X) * b10 / max_b10_value
#     return x

# # --- 2. Função Objetivo (Problema 1.1 do Tutorial) ---

# def objective_function(x):
#     """
#     Implementa a função objetivo f(x) = x * sin(10*pi*x) + 1.
#     O objetivo do Algoritmo Genético é maximizar esta função.
#     """
#     return x * math.sin(10 * math.pi * x) + 1

# # --- 3. Geração da População Inicial ---

# def generate_initial_population(size, length, distribution_type="random"):
#     """
#     Gera a população inicial de cromossomos.
    
#     Args:
#         size (int): O número de cromossomos na população.
#         length (int): O comprimento de cada cromossomo (número de bits).
#         distribution_type (str): "random" para cromossomos aleatórios,
#                                  "equidistant" para cromossomos que representam 
#                                  valores de x igualmente espaçados.
    
#     Returns:
#         list: Uma lista de strings binárias, representando a população.
#     """
#     population = []
#     if distribution_type == "random":
#         print("Gerando população inicial aleatória...")
#         for _ in range(size):
#             chromosome = ''.join(random.choice('01') for _ in range(length))
#             population.append(chromosome)
#     elif distribution_type == "equidistant":
#         print("Gerando população inicial equidistante...")
#         if size == 1:
#             x_value = (MIN_X + MAX_X) / 2 # Apenas o ponto central para uma população de 1
#             population.append(encode_x(x_value))
#         else:
#             # Gerar valores de x igualmente espaçados no domínio [MIN_X, MAX_X] e codificá-los
#             step_size = (MAX_X - MIN_X) / (size - 1)
#             for i in range(size):
#                 x_value = MIN_X + i * step_size
#                 population.append(encode_x(x_value))
#     else:
#         raise ValueError("Tipo de distribuição inválido. Use 'random' ou 'equidistant'.")
#     return population

# # --- 4. Avaliação da Aptidão (Fitness) com Ranqueamento ---

# def evaluate_population(population):
#     """
#     Avalia a aptidão de cada cromossomo na população.
#     A aptidão é ranqueada conforme Seção 1.3 e Tabela 1 do tutorial
#     (o melhor recebe RANK_MAX_FITNESS, o pior RANK_MIN_FITNESS, e os demais linearmente).
    
#     Returns:
#         list: Uma lista de dicionários, cada um contendo o cromossomo, 
#               o valor de x decodificado, o valor da função objetivo e a aptidão ranqueada.
#     """
#     evaluated_chromosomes = []
#     for chromosome in population:
#         x_value = decode_chromosome(chromosome)
#         objective_value = objective_function(x_value)
#         evaluated_chromosomes.append({
#             'chromosome': chromosome,
#             'x_value': x_value,
#             'objective_value': objective_value,
#             'fitness': 0 # A aptidão será preenchida após o ranqueamento
#         })
    
#     # Ordena a população pelo valor da função objetivo (descrescente para maximização)
#     # Isso coloca o "mais apto" (melhor valor de f(x)) no início da lista
#     evaluated_chromosomes.sort(key=lambda item: item['objective_value'], reverse=True)
    
#     # Atribui aptidão ranqueada baseada na posição ordenada
#     # fi = RANK_MIN_FITNESS + (RANK_MAX_FITNESS - RANK_MIN_FITNESS) * (N - 1 - i) / (N - 1)
#     N = len(evaluated_chromosomes)
#     for i, item in enumerate(evaluated_chromosomes):
#         if N > 1: # Evita divisão por zero se a população tiver apenas 1 elemento
#             item['fitness'] = RANK_MIN_FITNESS + (RANK_MAX_FITNESS - RANK_MIN_FITNESS) * (N - 1 - i) / (N - 1)
#         else: # Se houver apenas um elemento, ele recebe a aptidão máxima
#             item['fitness'] = RANK_MAX_FITNESS
            
#     return evaluated_chromosomes

# # --- 5. Seleção (Roleta) ---

# def select_parents(evaluated_population):
#     """
#     Seleciona os pais para a próxima geração usando o algoritmo da roleta.
#     Baseado na Figura 4 e Equação (1.3) do tutorial.
#     Os cromossomos com maior aptidão ranqueada têm maior probabilidade de serem selecionados.
#     """
#     total_fitness = sum(item['fitness'] for item in evaluated_population)
    
#     # Se a aptidão total for zero (ex: todos os cromossomos têm aptidão 0), 
#     # seleciona aleatoriamente para evitar divisão por zero ou loop infinito.
#     if total_fitness == 0:
#         return [random.choice([item['chromosome'] for item in evaluated_population]) for _ in range(POPULATION_SIZE)]

#     # Cria uma lista de aptidões acumuladas para a roleta
#     cumulative_fitness = []
#     current_cumulative = 0
#     for item in evaluated_population:
#         current_cumulative += item['fitness']
#         cumulative_fitness.append(current_cumulative)
        
#     selected_parents = []
#     for _ in range(POPULATION_SIZE):
#         # Gera um número aleatório entre 0 e o total_fitness
#         r = random.uniform(0, total_fitness)
        
#         # Encontra o cromossomo correspondente na roleta
#         # Percorre a aptidão acumulada para encontrar o "slot" onde 'r' cai
#         for i, cumulative_val in enumerate(cumulative_fitness):
#             if r <= cumulative_val:
#                 selected_parents.append(evaluated_population[i]['chromosome'])
#                 break
#     return selected_parents

# # --- 6. NOVO: Crossover Flexível (1 ou 2 Pontos) ---

# def crossover(parent1, parent2, num_points=1):
#     """
#     Realiza crossover entre dois pais com número variável de pontos.
    
#     Args:
#         parent1, parent2 (str): Cromossomos pais
#         num_points (int): Número de pontos de crossover (1 ou 2)
    
#     Returns:
#         tuple: Dois cromossomos filhos
#     """
#     if random.random() < CROSSOVER_RATE:
#         if num_points == 1:
#             # Crossover de ponto único (como no código original)
#             point = random.randint(1, CHROMOSOME_LENGTH - 1)
#             child1 = parent1[:point] + parent2[point:]
#             child2 = parent2[:point] + parent1[point:]
#             return child1, child2
            
#         elif num_points == 2:
#             # Crossover de dois pontos
#             # Garante que os pontos sejam diferentes e ordenados
#             point1 = random.randint(1, CHROMOSOME_LENGTH - 2)
#             point2 = random.randint(point1 + 1, CHROMOSOME_LENGTH - 1)
            
#             # Troca o segmento central entre os dois pontos
#             child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
#             child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
#             return child1, child2
#         else:
#             raise ValueError("Número de pontos deve ser 1 ou 2")
#     else:
#         # Se não ocorrer crossover, os filhos são cópias exatas dos pais
#         return parent1, parent2

# # --- 7. Mutação (Inversão de Bit) ---

# def mutate(chromosome):
#     """
#     Realiza a mutação de inversão de bit em um cromossomo.
#     Baseado na Seção 1.4 do tutorial.
#     Cada bit tem uma probabilidade MUTATION_RATE de ser invertido (0 vira 1, 1 vira 0).
#     """
#     mutated_chromosome = list(chromosome) # Converte a string para uma lista de caracteres para poder modificar
#     for i in range(CHROMOSOME_LENGTH):
#         if random.random() < MUTATION_RATE:
#             mutated_chromosome[i] = '1' if mutated_chromosome[i] == '0' else '0' # Inverte o bit
#     return "".join(mutated_chromosome) # Retorna a lista de caracteres como uma string

# # --- 8. NOVA: Função para Plotar Evolução das Soluções ---

# def plot_solution_evolution(evolution_data, distribution_type):
#     """
#     Plota a evolução das soluções no espaço de busca, similar ao tutorial.
    
#     Args:
#         evolution_data (list): Lista com dados de cada geração
#         distribution_type (str): Tipo de distribuição inicial
#     """
#     # Criar figura com subplots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
#     # Gráfico 1: Função objetivo e evolução das soluções
#     x_func = np.linspace(MIN_X, MAX_X, 1000)
#     y_func = x_func * np.sin(10 * np.pi * x_func) + 1
    
#     ax1.plot(x_func, y_func, 'k-', linewidth=1, alpha=0.7, label='f(x) = x sin(10πx) + 1')
    
#     # Plotar população inicial
#     initial_generation = evolution_data[0]
#     initial_x = [item['x_value'] for item in initial_generation['population']]
#     initial_y = [item['objective_value'] for item in initial_generation['population']]
#     ax1.scatter(initial_x, initial_y, c='lightblue', marker='o', s=50, alpha=0.7, label='População Inicial')
    
#     # Plotar melhores soluções de cada geração
#     best_x_evolution = [gen['best_individual']['x_value'] for gen in evolution_data]
#     best_y_evolution = [gen['best_individual']['objective_value'] for gen in evolution_data]
    
#     # Usar cores diferentes para cada geração
#     colors = plt.cm.viridis(np.linspace(0, 1, len(evolution_data)))
    
#     for i, (x, y) in enumerate(zip(best_x_evolution, best_y_evolution)):
#         ax1.scatter(x, y, c=[colors[i]], marker='*', s=100, 
#                    label=f'Melhor Gen {i+1}' if i < 5 or i % 5 == 0 or i == len(evolution_data)-1 else "")
    
#     # Destacar a melhor solução final
#     final_best_x = best_x_evolution[-1]
#     final_best_y = best_y_evolution[-1]
#     ax1.scatter(final_best_x, final_best_y, c='red', marker='*', s=200, 
#                edgecolors='black', linewidth=2, label='Melhor Final')
    
#     # Marcar máximo teórico
#     ax1.scatter(1.85055, 2.85027, c='green', marker='D', s=100, 
#                edgecolors='black', linewidth=1, label='Máximo Teórico')
    
#     ax1.set_xlabel('x')
#     ax1.set_ylabel('f(x)')
#     ax1.set_title(f'Evolução das Soluções - {distribution_type.title()}')
#     ax1.grid(True, alpha=0.3)
#     ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     ax1.set_xlim(MIN_X, MAX_X)
    
#     # Gráfico 2: Convergência do fitness
#     generations = list(range(1, len(evolution_data) + 1))
#     best_fitness = [gen['best_individual']['objective_value'] for gen in evolution_data]
#     avg_fitness = [gen['avg_objective'] for gen in evolution_data]
    
#     ax2.plot(generations, best_fitness, 'b-', marker='o', markersize=4, label='Melhor f(x)')
#     ax2.plot(generations, avg_fitness, 'r--', marker='s', markersize=3, label='Média f(x)')
#     ax2.axhline(y=2.85027, color='green', linestyle=':', label='Máximo Teórico')
    
#     ax2.set_xlabel('Geração')
#     ax2.set_ylabel('Valor de f(x)')
#     ax2.set_title('Convergência do Fitness')
#     ax2.grid(True, alpha=0.3)
#     ax2.legend()
    
#     plt.tight_layout()
#     plt.show()

# # --- Algoritmo Genético Principal ---

# def run_genetic_algorithm(distribution_type="random"):
#     """
#     Executa o algoritmo genético completo.
#     Monitora e imprime o melhor valor da função objetivo e a média da população ao longo das gerações.
#     """    
#     crossover_type = "ponto único" if CROSSOVER_POINTS == 1 else "dois pontos"
#     print(f"População: {POPULATION_SIZE}, Gerações: {NUM_GENERATIONS}")
#     print(f"Crossover: {crossover_type} ({CROSSOVER_RATE*100:.0f}%), Mutação: {MUTATION_RATE*100:.1f}%")
#     print(f"Elitismo: {'Ativado' if USE_ELITISM else 'Desativado'}, Distribuição inicial: '{distribution_type}'")
    
#     # Geração da população inicial
#     population = generate_initial_population(POPULATION_SIZE, CHROMOSOME_LENGTH, distribution_type)
    
#     # Variáveis para armazenar o melhor indivíduo global encontrado
#     best_overall_individual = None
#     best_overall_objective_value = -float('inf') # Inicializa com valor negativo infinito para maximização

#     # Listas para armazenar dados de desempenho por geração (útil para gráficos)
#     best_objective_per_generation = []
#     avg_objective_per_generation = []
    
#     # NOVA: Lista para armazenar dados detalhados de evolução
#     evolution_data = []

#     print("\n--- Evolução das gerações ---")
#     for generation in range(NUM_GENERATIONS):
#         # 1. Avaliar a população atual
#         evaluated_population = evaluate_population(population)
        
#         # Encontrar o melhor indivíduo da GERAÇÃO ATUAL (baseado no valor da função objetivo)
#         current_best_individual_data = max(evaluated_population, key=lambda item: item['objective_value'])
#         current_best_chromosome = current_best_individual_data['chromosome']
#         current_best_x_value = current_best_individual_data['x_value']
#         current_best_objective = current_best_individual_data['objective_value']
        
#         # Atualizar o melhor indivíduo GLOBAL, se o atual for melhor
#         if current_best_objective > best_overall_objective_value:
#             best_overall_objective_value = current_best_objective
#             best_overall_individual = current_best_chromosome
        
#         # Calcular o valor médio da função objetivo para a geração atual
#         avg_objective = sum(item['objective_value'] for item in evaluated_population) / POPULATION_SIZE
        
#         # Registrar os dados para análise
#         best_objective_per_generation.append(current_best_objective)
#         avg_objective_per_generation.append(avg_objective)
        
#         # NOVO: Armazenar dados detalhados para visualização
#         evolution_data.append({
#             'generation': generation + 1,
#             'population': evaluated_population.copy(),
#             'best_individual': current_best_individual_data.copy(),
#             'avg_objective': avg_objective
#         })

#         # Imprimir o progresso a cada 10 gerações ou na primeira/última
#         if (generation + 1) % 10 == 0 or generation == 0 or generation == NUM_GENERATIONS - 1:
#             print(f"Geração {generation + 1:3d}: Melhor f(x) = {current_best_objective:.6f} (x={current_best_x_value:.6f}), Média f(x) = {avg_objective:.6f}")

#         # 2. Seleção: Escolher pais para a próxima geração
#         parents = select_parents(evaluated_population)
        
#         # Inicializar a próxima população
#         next_population = []
        
#         # 3. Elitismo (Seção 1.5 do tutorial): Transferir o melhor indivíduo para a próxima geração
#         if USE_ELITISM:
#             next_population.append(current_best_individual_data['chromosome']) # Adiciona o melhor da geração atual
        
#         # 4. Crossover e Mutação: Gerar o restante da próxima população
#         # Os pais são escolhidos aleatoriamente da lista 'parents' (que já foi selecionada pela roleta)
#         while len(next_population) < POPULATION_SIZE:
#             # Escolhe dois pais aleatoriamente para reprodução
#             p1 = random.choice(parents)
#             p2 = random.choice(parents)
            
#             # Realiza crossover para gerar filhos (NOVO: usando parâmetro de pontos)
#             child1, child2 = crossover(p1, p2, CROSSOVER_POINTS)
            
#             # Aplica mutação em cada filho
#             child1 = mutate(child1)
#             child2 = mutate(child2)
            
#             # Adiciona os filhos à próxima população
#             next_population.append(child1)
#             if len(next_population) < POPULATION_SIZE: # Garante que não exceda o tamanho da população
#                 next_population.append(child2)
        
#         # A população atual se torna a próxima população para a próxima iteração
#         population = next_population

#     # --- Resultados Finais ---
#     print("\n--- Resultados finais da execução ---")
#     if best_overall_individual:
#         best_x_overall = decode_chromosome(best_overall_individual)
#         print(f"Melhor cromossomo global encontrado: {best_overall_individual}")
#         print(f"Valor decodificado de x: {best_x_overall:.6f}")
#         print(f"Valor máximo da função f(x): {best_overall_objective_value:.6f}")
#     else:
#         print("Nenhum indivíduo foi rastreado. Verifique a lógica do algoritmo.")
        
#     # O valor global de máximo para f(x)=x sin(10πx) + 1 no intervalo [-1, 2] é
#     # f(1.85055) = 2.85027, conforme mencionado na Seção 1.1 do tutorial.
#     print(f"\nValor ótimo global esperado (segundo o tutorial): f(1.85055) = 2.85027")

#     # NOVO: Plotar evolução das soluções
#     plot_solution_evolution(evolution_data, distribution_type)

#     return best_objective_per_generation, avg_objective_per_generation

# # --- Execução Principal do Script ---
# if __name__ == "__main__":
#     print(f"\n=== TESTE COM CROSSOVER DE {CROSSOVER_POINTS} PONTO(S) ===")
    
#     # Execute o AG com distribuição aleatória inicial
#     best_random_run, avg_random_run = run_genetic_algorithm(distribution_type="random")
    
#     print("\n" + "="*80 + "\n") # Separador para facilitar a visualização
    
#     # Execute o AG com distribuição equidistante inicial
#     best_equidistant_run, avg_equidistant_run = run_genetic_algorithm(distribution_type="equidistant")

#     # --- Comparação Final dos Resultados ---
#     plt.figure(figsize=(14, 7))
#     plt.plot(best_random_run, label='Melhor f(x) (Pop. Aleatória)', color='blue', marker='o', markersize=3)
#     plt.plot(avg_random_run, label='Média f(x) (Pop. Aleatória)', linestyle='--', color='lightblue')
#     plt.plot(best_equidistant_run, label='Melhor f(x) (Pop. Equidistante)', color='red', marker='s', markersize=3)
#     plt.plot(avg_equidistant_run, label='Média f(x) (Pop. Equidistante)', linestyle='--', color='salmon')
    
#     # Linha de referência para o máximo global teórico
#     theoretical_max = 2.85027
#     plt.axhline(y=theoretical_max, color='green', linestyle=':', label=f'Máximo global teórico ({theoretical_max:.5f})')
    
#     crossover_type = "Ponto Único" if CROSSOVER_POINTS == 1 else "Dois Pontos"
#     plt.title(f'Comparação: Evolução do Fitness por Geração (Crossover {crossover_type})')
#     plt.xlabel('Geração')
#     plt.ylabel('Valor de f(x)')
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.show()







import math
import random
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# --- Parâmetros do Algoritmo Genético ---
MIN_X = -1.0  # Limite inferior do domínio de x, conforme o tutorial
MAX_X = 2.0   # Limite superior do domínio de x, conforme o tutorial
CHROMOSOME_LENGTH = 22 # Tamanho da cadeia de bits para o cromossomo, como no tutorial
POPULATION_SIZE = 30 # Tamanho da população, como na Tabela 1 do tutorial
NUM_GENERATIONS = 25 # Número de gerações para a evolução
CROSSOVER_RATE = 0.8  # Taxa de crossover (80% é um valor comum, pode ajustar)
MUTATION_RATE = 0.01  # Taxa de mutação (1% é um valor comum, pode ajustar)
USE_ELITISM = False    # Ativar ou desativar o Elitismo (conforme Seção 1.5)

# NOVO: Parâmetro para escolher o tipo de crossover
CROSSOVER_POINTS = 1  # 1 para ponto único, 2 para dois pontos

# Aptidão para ranqueamento (Seção 1.3, Tabela 1): o 1º recebe 2.0, o último 0.0
RANK_MAX_FITNESS = 2.0 
RANK_MIN_FITNESS = 0.0 

# --- 1. Funções de Codificação e Decodificação do Cromossomo ---

def encode_x(value):
    """
    Codifica um valor real 'x' no intervalo [MIN_X, MAX_X] em uma cadeia de bits.
    Utiliza a lógica inversa da Equação (1.2) do tutorial para obter o valor inteiro b10.
    """
    # Garante que o valor esteja dentro dos limites para evitar erros de mapeamento
    value = max(MIN_X, min(MAX_X, value))
    
    # Calcula o valor inteiro b10 correspondente ao valor real x
    # b10 = (x - min_x) * (2^L - 1) / (max_x - min_x)
    range_span = MAX_X - MIN_X
    max_b10_value = (2**CHROMOSOME_LENGTH) - 1
    
    # Arredonda para o inteiro mais próximo para a codificação binária
    b10 = round((value - MIN_X) * max_b10_value / range_span)
    
    # Converte o inteiro para uma string binária de comprimento fixo (preenche com zeros à esquerda)
    binary_str = bin(int(b10))[2:].zfill(CHROMOSOME_LENGTH)
    return binary_str

def decode_chromosome(chromosome_str):
    """
    Decodifica uma cadeia de bits (cromossomo) em um valor real 'x'.
    Baseado na Equação (1.2) do tutorial: x = min + (max - min) * b10 / (2^l - 1).
    """
    b10 = int(chromosome_str, 2) # Converte a string binária para um inteiro
    max_b10_value = (2**CHROMOSOME_LENGTH) - 1 # Valor máximo que b10 pode assumir
    
    # Aplica a fórmula de mapeamento para obter o valor real x
    x = MIN_X + (MAX_X - MIN_X) * b10 / max_b10_value
    return x

# --- 2. Função Objetivo (Problema 1.1 do Tutorial) ---

def objective_function(x):
    """
    Implementa a função objetivo f(x) = x * sin(10*pi*x) + 1.
    O objetivo do Algoritmo Genético é maximizar esta função.
    """
    return x * math.sin(10 * math.pi * x) + 1

# --- 3. Geração da População Inicial ---

def generate_initial_population(size, length, distribution_type="random"):
    """
    Gera a população inicial de cromossomos.
    
    Args:
        size (int): O número de cromossomos na população.
        length (int): O comprimento de cada cromossomo (número de bits).
        distribution_type (str): "random" para cromossomos aleatórios,
                                 "equidistant" para cromossomos que representam 
                                 valores de x igualmente espaçados.
    
    Returns:
        list: Uma lista de strings binárias, representando a população.
    """
    population = []
    if distribution_type == "random":
        print("Gerando população inicial aleatória...")
        for _ in range(size):
            chromosome = ''.join(random.choice('01') for _ in range(length))
            population.append(chromosome)
    elif distribution_type == "equidistant":
        print("Gerando população inicial equidistante...")
        if size == 1:
            x_value = (MIN_X + MAX_X) / 2 # Apenas o ponto central para uma população de 1
            population.append(encode_x(x_value))
        else:
            # Gerar valores de x igualmente espaçados no domínio [MIN_X, MAX_X] e codificá-los
            step_size = (MAX_X - MIN_X) / (size - 1)
            for i in range(size):
                x_value = MIN_X + i * step_size
                population.append(encode_x(x_value))
    else:
        raise ValueError("Tipo de distribuição inválido. Use 'random' ou 'equidistant'.")
    return population

# --- 4. Avaliação da Aptidão (Fitness) com Ranqueamento ---

def evaluate_population(population):
    """
    Avalia a aptidão de cada cromossomo na população.
    A aptidão é ranqueada conforme Seção 1.3 e Tabela 1 do tutorial
    (o melhor recebe RANK_MAX_FITNESS, o pior RANK_MIN_FITNESS, e os demais linearmente).
    
    Returns:
        list: Uma lista de dicionários, cada um contendo o cromossomo, 
              o valor de x decodificado, o valor da função objetivo e a aptidão ranqueada.
    """
    evaluated_chromosomes = []
    for chromosome in population:
        x_value = decode_chromosome(chromosome)
        objective_value = objective_function(x_value)
        evaluated_chromosomes.append({
            'chromosome': chromosome,
            'x_value': x_value,
            'objective_value': objective_value,
            'fitness': 0 # A aptidão será preenchida após o ranqueamento
        })
    
    # Ordena a população pelo valor da função objetivo (descrescente para maximização)
    # Isso coloca o "mais apto" (melhor valor de f(x)) no início da lista
    evaluated_chromosomes.sort(key=lambda item: item['objective_value'], reverse=True)
    
    # Atribui aptidão ranqueada baseada na posição ordenada
    # fi = RANK_MIN_FITNESS + (RANK_MAX_FITNESS - RANK_MIN_FITNESS) * (N - 1 - i) / (N - 1)
    N = len(evaluated_chromosomes)
    for i, item in enumerate(evaluated_chromosomes):
        if N > 1: # Evita divisão por zero se a população tiver apenas 1 elemento
            item['fitness'] = RANK_MIN_FITNESS + (RANK_MAX_FITNESS - RANK_MIN_FITNESS) * (N - 1 - i) / (N - 1)
        else: # Se houver apenas um elemento, ele recebe a aptidão máxima
            item['fitness'] = RANK_MAX_FITNESS
            
    return evaluated_chromosomes

# --- 5. Seleção (Roleta) ---

def select_parents(evaluated_population):
    """
    Seleciona os pais para a próxima geração usando o algoritmo da roleta.
    Baseado na Figura 4 e Equação (1.3) do tutorial.
    Os cromossomos com maior aptidão ranqueada têm maior probabilidade de serem selecionados.
    """
    total_fitness = sum(item['fitness'] for item in evaluated_population)
    
    # Se a aptidão total for zero (ex: todos os cromossomos têm aptidão 0), 
    # seleciona aleatoriamente para evitar divisão por zero ou loop infinito.
    if total_fitness == 0:
        return [random.choice([item['chromosome'] for item in evaluated_population]) for _ in range(POPULATION_SIZE)]

    # Cria uma lista de aptidões acumuladas para a roleta
    cumulative_fitness = []
    current_cumulative = 0
    for item in evaluated_population:
        current_cumulative += item['fitness']
        cumulative_fitness.append(current_cumulative)
        
    selected_parents = []
    for _ in range(POPULATION_SIZE):
        # Gera um número aleatório entre 0 e o total_fitness
        r = random.uniform(0, total_fitness)
        
        # Encontra o cromossomo correspondente na roleta
        # Percorre a aptidão acumulada para encontrar o "slot" onde 'r' cai
        for i, cumulative_val in enumerate(cumulative_fitness):
            if r <= cumulative_val:
                selected_parents.append(evaluated_population[i]['chromosome'])
                break
    return selected_parents

# --- 6. NOVO: Crossover Flexível (1 ou 2 Pontos) ---

def crossover(parent1, parent2, num_points=1):
    """
    Realiza crossover entre dois pais com número variável de pontos.
    
    Args:
        parent1, parent2 (str): Cromossomos pais
        num_points (int): Número de pontos de crossover (1 ou 2)
    
    Returns:
        tuple: Dois cromossomos filhos
    """
    if random.random() < CROSSOVER_RATE:
        if num_points == 1:
            # Crossover de ponto único (como no código original)
            point = random.randint(1, CHROMOSOME_LENGTH - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
            
        elif num_points == 2:
            # Crossover de dois pontos
            # Garante que os pontos sejam diferentes e ordenados
            point1 = random.randint(1, CHROMOSOME_LENGTH - 2)
            point2 = random.randint(point1 + 1, CHROMOSOME_LENGTH - 1)
            
            # Troca o segmento central entre os dois pontos
            child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
            child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
            return child1, child2
        else:
            raise ValueError("Número de pontos deve ser 1 ou 2")
    else:
        # Se não ocorrer crossover, os filhos são cópias exatas dos pais
        return parent1, parent2

# --- 7. Mutação (Inversão de Bit) ---

def mutate(chromosome):
    """
    Realiza a mutação de inversão de bit em um cromossomo.
    Baseado na Seção 1.4 do tutorial.
    Cada bit tem uma probabilidade MUTATION_RATE de ser invertido (0 vira 1, 1 vira 0).
    """
    mutated_chromosome = list(chromosome) # Converte a string para uma lista de caracteres para poder modificar
    for i in range(CHROMOSOME_LENGTH):
        if random.random() < MUTATION_RATE:
            mutated_chromosome[i] = '1' if mutated_chromosome[i] == '0' else '0' # Inverte o bit
    return "".join(mutated_chromosome) # Retorna a lista de caracteres como uma string

# --- 8. NOVA: Função para Plotar Evolução das Soluções (Interativo) ---

def plot_solution_evolution_interactive(evolution_data, distribution_type):
    """
    Plota a evolução das soluções no espaço de busca usando Plotly (interativo).
    
    Args:
        evolution_data (list): Lista com dados de cada geração
        distribution_type (str): Tipo de distribuição inicial
    """
    # Criar subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Evolução das Soluções - {distribution_type.title()}', 'Convergência do Fitness'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Gráfico 1: Função objetivo e evolução das soluções
    x_func = np.linspace(MIN_X, MAX_X, 1000)
    y_func = x_func * np.sin(10 * np.pi * x_func) + 1
    
    # Plotar função objetivo
    fig.add_trace(
        go.Scatter(x=x_func, y=y_func, mode='lines', name='f(x) = x sin(10πx) + 1',
                  line=dict(color='black', width=1), opacity=0.7),
        row=1, col=1
    )
    
    # Plotar população inicial (pontos pequenos e pretos)
    initial_generation = evolution_data[0]
    initial_x = [item['x_value'] for item in initial_generation['population']]
    initial_y = [item['objective_value'] for item in initial_generation['population']]
    
    fig.add_trace(
        go.Scatter(x=initial_x, y=initial_y, mode='markers', name='População Inicial',
                  marker=dict(color='black', size=4, opacity=0.7)),
        row=1, col=1
    )
    
    # Plotar melhores soluções de cada geração com números
    best_x_evolution = [gen['best_individual']['x_value'] for gen in evolution_data]
    best_y_evolution = [gen['best_individual']['objective_value'] for gen in evolution_data]
    
    # Criar escala de cores (mais escuro = geração mais recente)
    num_generations = len(evolution_data)
    colors = px.colors.sequential.Viridis
    
    # Determinar quais gerações mostrar na legenda (para evitar poluição visual)
    legend_generations = []
    if num_generations <= 10:
        legend_generations = list(range(num_generations))
    else:
        # Mostrar primeira, última e algumas intermediárias
        step = max(1, num_generations // 5)
        legend_generations = [0] + list(range(step, num_generations-1, step)) + [num_generations-1]
    
    for i, (x, y) in enumerate(zip(best_x_evolution, best_y_evolution)):
        # Calcular intensidade da cor (0 = mais claro, 1 = mais escuro)
        color_intensity = i / (num_generations - 1) if num_generations > 1 else 0
        color_idx = int(color_intensity * (len(colors) - 1))
        
        # Determinar se deve aparecer na legenda
        show_legend = i in legend_generations
        legend_name = f'Geração {i+1}' if show_legend else None
        
        fig.add_trace(
            go.Scatter(
                x=[x], y=[y], 
                mode='markers+text',
                text=[str(i+1)],
                textposition="middle center",
                textfont=dict(color="white", size=8),
                name=legend_name,
                showlegend=show_legend,
                marker=dict(
                    color=colors[color_idx],
                    size=12,
                    line=dict(color='black', width=1)
                ),
                hovertemplate=f"Geração {i+1}<br>x: %{{x:.4f}}<br>f(x): %{{y:.4f}}<extra></extra>"
            ),
            row=1, col=1
        )
    
    # # Marcar máximo teórico
    # fig.add_trace(
    #     go.Scatter(x=[1.85055], y=[2.85027], mode='markers', name='Máximo Teórico',
    #               marker=dict(color='green', size=10, symbol='diamond',
    #                          line=dict(color='black', width=1))),
    #     row=1, col=1
    # )
    
    # Gráfico 2: Convergência do fitness
    generations = list(range(1, len(evolution_data) + 1))
    best_fitness = [gen['best_individual']['objective_value'] for gen in evolution_data]
    avg_fitness = [gen['avg_objective'] for gen in evolution_data]
    
    fig.add_trace(
        go.Scatter(x=generations, y=best_fitness, mode='lines+markers', name='Melhor f(x)',
                  line=dict(color='blue'), marker=dict(size=4)),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=generations, y=avg_fitness, mode='lines+markers', name='Média f(x)',
                  line=dict(color='red', dash='dash'), marker=dict(size=3, symbol='square')),
        row=1, col=2
    )
    
    # Linha de referência para o máximo teórico
    fig.add_hline(y=2.85027, line_dash="dot", line_color="green", 
                  annotation_text="Máximo teórico", row=1, col=2)
    
    # Configurar layout
    fig.update_xaxes(title_text="x", row=1, col=1, range=[MIN_X, MAX_X])
    fig.update_yaxes(title_text="f(x)", row=1, col=1)
    fig.update_xaxes(title_text="Geração", row=1, col=2)
    fig.update_yaxes(title_text="Valor de f(x)", row=1, col=2)
    
    crossover_type = "Ponto único" if CROSSOVER_POINTS == 1 else "Dois pontos"
    fig.update_layout(
        title=f"Algoritmo genético - crossover {crossover_type}",
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    fig.show()

# --- Algoritmo Genético Principal ---

def run_genetic_algorithm(distribution_type="random"):
    """
    Executa o algoritmo genético completo.
    Monitora e imprime o melhor valor da função objetivo e a média da população ao longo das gerações.
    """    
    crossover_type = "ponto único" if CROSSOVER_POINTS == 1 else "dois pontos"
    print(f"População: {POPULATION_SIZE}, Gerações: {NUM_GENERATIONS}")
    print(f"Crossover: {crossover_type} ({CROSSOVER_RATE*100:.0f}%), Mutação: {MUTATION_RATE*100:.1f}%")
    print(f"Elitismo: {'Ativado' if USE_ELITISM else 'Desativado'}, Distribuição inicial: '{distribution_type}'")
    
    # Geração da população inicial
    population = generate_initial_population(POPULATION_SIZE, CHROMOSOME_LENGTH, distribution_type)
    
    # Variáveis para armazenar o melhor indivíduo global encontrado
    best_overall_individual = None
    best_overall_objective_value = -float('inf') # Inicializa com valor negativo infinito para maximização

    # Listas para armazenar dados de desempenho por geração (útil para gráficos)
    best_objective_per_generation = []
    avg_objective_per_generation = []
    
    # NOVA: Lista para armazenar dados detalhados de evolução
    evolution_data = []

    print("\n--- Evolução das gerações ---")
    for generation in range(NUM_GENERATIONS):
        # 1. Avaliar a população atual
        evaluated_population = evaluate_population(population)
        
        # Encontrar o melhor indivíduo da GERAÇÃO ATUAL (baseado no valor da função objetivo)
        current_best_individual_data = max(evaluated_population, key=lambda item: item['objective_value'])
        current_best_chromosome = current_best_individual_data['chromosome']
        current_best_x_value = current_best_individual_data['x_value']
        current_best_objective = current_best_individual_data['objective_value']
        
        # Atualizar o melhor indivíduo GLOBAL, se o atual for melhor
        if current_best_objective > best_overall_objective_value:
            best_overall_objective_value = current_best_objective
            best_overall_individual = current_best_chromosome
        
        # Calcular o valor médio da função objetivo para a geração atual
        avg_objective = sum(item['objective_value'] for item in evaluated_population) / POPULATION_SIZE
        
        # Registrar os dados para análise
        best_objective_per_generation.append(current_best_objective)
        avg_objective_per_generation.append(avg_objective)
        
        # NOVO: Armazenar dados detalhados para visualização
        evolution_data.append({
            'generation': generation + 1,
            'population': evaluated_population.copy(),
            'best_individual': current_best_individual_data.copy(),
            'avg_objective': avg_objective
        })

        # Imprimir o progresso a cada 10 gerações ou na primeira/última
        if (generation + 1) % 10 == 0 or generation == 0 or generation == NUM_GENERATIONS - 1:
            print(f"Geração {generation + 1:3d}: Melhor f(x) = {current_best_objective:.6f} (x={current_best_x_value:.6f}), Média f(x) = {avg_objective:.6f}")

        # 2. Seleção: Escolher pais para a próxima geração
        parents = select_parents(evaluated_population)
        
        # Inicializar a próxima população
        next_population = []
        
        # 3. Elitismo (Seção 1.5 do tutorial): Transferir o melhor indivíduo para a próxima geração
        if USE_ELITISM:
            next_population.append(current_best_individual_data['chromosome']) # Adiciona o melhor da geração atual
        
        # 4. Crossover e Mutação: Gerar o restante da próxima população
        # Os pais são escolhidos aleatoriamente da lista 'parents' (que já foi selecionada pela roleta)
        while len(next_population) < POPULATION_SIZE:
            # Escolhe dois pais aleatoriamente para reprodução
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            
            # Realiza crossover para gerar filhos (NOVO: usando parâmetro de pontos)
            child1, child2 = crossover(p1, p2, CROSSOVER_POINTS)
            
            # Aplica mutação em cada filho
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            # Adiciona os filhos à próxima população
            next_population.append(child1)
            if len(next_population) < POPULATION_SIZE: # Garante que não exceda o tamanho da população
                next_population.append(child2)
        
        # A população atual se torna a próxima população para a próxima iteração
        population = next_population

    # --- Resultados Finais ---
    print("\n--- Resultados finais da execução ---")
    if best_overall_individual:
        best_x_overall = decode_chromosome(best_overall_individual)
        print(f"Melhor cromossomo global encontrado: {best_overall_individual}")
        print(f"Valor decodificado de x: {best_x_overall:.6f}")
        print(f"Valor máximo da função f(x): {best_overall_objective_value:.6f}")
    else:
        print("Nenhum indivíduo foi rastreado. Verifique a lógica do algoritmo.")
        
    # O valor global de máximo para f(x)=x sin(10πx) + 1 no intervalo [-1, 2] é
    # f(1.85055) = 2.85027, conforme mencionado na Seção 1.1 do tutorial.
    print(f"\nValor ótimo global esperado (segundo o tutorial): f(1.85055) = 2.85027")

    # NOVO: Plotar evolução das soluções (interativo)
    plot_solution_evolution_interactive(evolution_data, distribution_type)

    return best_objective_per_generation, avg_objective_per_generation

# --- Execução Principal do Script ---
if __name__ == "__main__":
    print(f"\n=== TESTE COM CROSSOVER DE {CROSSOVER_POINTS} PONTO(S) ===")
    
    # Execute o AG com distribuição aleatória inicial
    best_random_run, avg_random_run = run_genetic_algorithm(distribution_type="random")
    
    print("\n" + "="*80 + "\n") # Separador para facilitar a visualização
    
    # Execute o AG com distribuição equidistante inicial
    best_equidistant_run, avg_equidistant_run = run_genetic_algorithm(distribution_type="equidistant")

    # --- Comparação Final dos Resultados (Matplotlib) ---
    plt.figure(figsize=(14, 7))
    plt.plot(best_random_run, label='Melhor f(x) (Pop. Aleatória)', color='blue', marker='o', markersize=3)
    plt.plot(avg_random_run, label='Média f(x) (Pop. Aleatória)', linestyle='--', color='lightblue')
    plt.plot(best_equidistant_run, label='Melhor f(x) (Pop. Equidistante)', color='red', marker='s', markersize=3)
    plt.plot(avg_equidistant_run, label='Média f(x) (Pop. Equidistante)', linestyle='--', color='salmon')
    
    # Linha de referência para o máximo global teórico
    theoretical_max = 2.85027
    plt.axhline(y=theoretical_max, color='green', linestyle=':', label=f'Máximo global teórico ({theoretical_max:.5f})')
    
    crossover_type = "Ponto Único" if CROSSOVER_POINTS == 1 else "Dois Pontos"
    plt.title(f'Comparação: Evolução do Fitness por Geração (Crossover {crossover_type})')
    plt.xlabel('Geração')
    plt.ylabel('Valor de f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()