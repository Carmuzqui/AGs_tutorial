import math
import random

# --- Parâmetros do Algoritmo Genético ---
MIN_X = -1.0  # Limite inferior do domínio de x, conforme o tutorial
MAX_X = 2.0   # Limite superior do domínio de x, conforme o tutorial
CHROMOSOME_LENGTH = 22 # Tamanho da cadeia de bits para o cromossomo, como no tutorial
POPULATION_SIZE = 30 # Tamanho da população, como na Tabela 1 do tutorial
NUM_GENERATIONS = 25 # Número de gerações para a evolução
CROSSOVER_RATE = 0.8  # Taxa de crossover (80% é um valor comum, pode ajustar)
MUTATION_RATE = 0.01  # Taxa de mutação (1% é um valor comum, pode ajustar)
USE_ELITISM = True    # Ativar ou desativar o Elitismo (conforme Seção 1.5)

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
    Seleciona os pais para a próxima geração usando o Algoritmo da Roleta.
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

# --- 6. Crossover (Ponto Único) ---

def crossover(parent1, parent2):
    """
    Realiza o crossover de ponto único entre dois pais.
    Baseado na Seção 1.4 do tutorial.
    Aplica-se com uma probabilidade CROSSOVER_RATE.
    """
    if random.random() < CROSSOVER_RATE:
        # Ponto de corte aleatório. O tutorial sugere que não seja no início nem no fim
        # para que haja troca de material genético.
        point = random.randint(1, CHROMOSOME_LENGTH - 1) # Garante que o ponto esteja entre 1 e L-1
        
        # Realiza a troca das "caudas" dos cromossomos
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
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

# --- Algoritmo Genético Principal ---

def run_genetic_algorithm(distribution_type="random"):
    """
    Executa o algoritmo genético completo.
    Monitora e imprime o melhor valor da função objetivo e a média da população ao longo das gerações.
    """
    print(f"\n--- Iniciando Algoritmo Genético com Parâmetros ---")
    print(f"População: {POPULATION_SIZE}, Gerações: {NUM_GENERATIONS}")
    print(f"Crossover Rate: {CROSSOVER_RATE*100:.0f}%, Mutação Rate: {MUTATION_RATE*100:.1f}%")
    print(f"Elitismo: {'Ativado' if USE_ELITISM else 'Desativado'}, Distribuição Inicial: '{distribution_type}'")
    
    # Geração da população inicial
    population = generate_initial_population(POPULATION_SIZE, CHROMOSOME_LENGTH, distribution_type)
    
    # Variáveis para armazenar o melhor indivíduo global encontrado
    best_overall_individual = None
    best_overall_objective_value = -float('inf') # Inicializa com valor negativo infinito para maximização

    # Listas para armazenar dados de desempenho por geração (útil para gráficos)
    best_objective_per_generation = []
    avg_objective_per_generation = []

    print("\n--- Evolução das Gerações ---")
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
            
            # Realiza crossover para gerar filhos
            child1, child2 = crossover(p1, p2)
            
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
    print("\n--- Resultados Finais da Execução ---")
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

    return best_objective_per_generation, avg_objective_per_generation

# --- Execução Principal do Script ---
if __name__ == "__main__":
    # Execute o AG com distribuição aleatória inicial
    best_random_run, avg_random_run = run_genetic_algorithm(distribution_type="random")
    
    print("\n" + "="*80 + "\n") # Separador para facilitar a visualização
    
    # Execute o AG com distribuição equidistante inicial
    best_equidistant_run, avg_equidistant_run = run_genetic_algorithm(distribution_type="equidistant")

    # --- Opcional: Visualização dos Resultados ---
    # Para visualizar a evolução do fitness, você pode descomentar o código abaixo
    # (requer a biblioteca matplotlib).
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(14, 7))
    plt.plot(best_random_run, label='Melhor f(x) (Pop. Aleatória)', color='blue')
    plt.plot(avg_random_run, label='Média f(x) (Pop. Aleatória)', linestyle='--', color='lightblue')
    plt.plot(best_equidistant_run, label='Melhor f(x) (Pop. Equidistante)', color='red')
    plt.plot(avg_equidistant_run, label='Média f(x) (Pop. Equidistante)', linestyle='--', color='salmon')
    
    # Linha de referência para o máximo global teórico
    theoretical_max = 2.85027
    plt.axhline(y=theoretical_max, color='green', linestyle=':', label=f'Máximo Global Teórico ({theoretical_max:.5f})')
    
    plt.title('Evolução do Valor da Função Objetivo por Geração (AG)')
    plt.xlabel('Geração')
    plt.ylabel('Valor de f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()