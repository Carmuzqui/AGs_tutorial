import streamlit as st
import math
import random
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import time
from scipy.optimize import differential_evolution

# Configuración de la página
st.set_page_config(
    page_title="Algoritmos genéticos",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Funciones del Algoritmo Genético ---

def encode_x(value, min_x, max_x, chromosome_length):
    """Codifica um valor real 'x' em uma cadeia de bits."""
    value = max(min_x, min(max_x, value))
    range_span = max_x - min_x
    max_b10_value = (2**chromosome_length) - 1
    b10 = round((value - min_x) * max_b10_value / range_span)
    binary_str = bin(int(b10))[2:].zfill(chromosome_length)
    return binary_str

def decode_chromosome(chromosome_str, min_x, max_x, chromosome_length):
    """Decodifica uma cadeia de bits em um valor real 'x'."""
    b10 = int(chromosome_str, 2)
    max_b10_value = (2**chromosome_length) - 1
    x = min_x + (max_x - min_x) * b10 / max_b10_value
    return x

def objective_function(x):
    """Implementa a função objetivo f(x) = x * sin(10*pi*x) + 1."""
    return x * math.sin(10 * math.pi * x) + 1

def generate_initial_population(size, length, distribution_type, min_x, max_x):
    """Gera a população inicial de cromossomos."""
    population = []
    if distribution_type == "random":
        for _ in range(size):
            chromosome = ''.join(random.choice('01') for _ in range(length))
            population.append(chromosome)
    elif distribution_type == "equidistant":
        if size == 1:
            x_value = (min_x + max_x) / 2
            population.append(encode_x(x_value, min_x, max_x, length))
        else:
            step_size = (max_x - min_x) / (size - 1)
            for i in range(size):
                x_value = min_x + i * step_size
                population.append(encode_x(x_value, min_x, max_x, length))
    return population

def evaluate_population(population, min_x, max_x, chromosome_length, rank_max_fitness, rank_min_fitness):
    """Avalia a aptidão de cada cromossomo na população."""
    evaluated_chromosomes = []
    for chromosome in population:
        x_value = decode_chromosome(chromosome, min_x, max_x, chromosome_length)
        objective_value = objective_function(x_value)
        evaluated_chromosomes.append({
            'chromosome': chromosome,
            'x_value': x_value,
            'objective_value': objective_value,
            'fitness': 0
        })
    
    # Ordenar e atribuir aptidão ranqueada
    evaluated_chromosomes.sort(key=lambda item: item['objective_value'], reverse=True)
    N = len(evaluated_chromosomes)
    for i, item in enumerate(evaluated_chromosomes):
        if N > 1:
            item['fitness'] = rank_min_fitness + (rank_max_fitness - rank_min_fitness) * (N - 1 - i) / (N - 1)
        else:
            item['fitness'] = rank_max_fitness
    
    return evaluated_chromosomes

def select_parents(evaluated_population, population_size):
    """Seleciona os pais usando o algoritmo da roleta."""
    total_fitness = sum(item['fitness'] for item in evaluated_population)
    
    if total_fitness == 0:
        return [random.choice([item['chromosome'] for item in evaluated_population]) for _ in range(population_size)]

    cumulative_fitness = []
    current_cumulative = 0
    for item in evaluated_population:
        current_cumulative += item['fitness']
        cumulative_fitness.append(current_cumulative)
        
    selected_parents = []
    for _ in range(population_size):
        r = random.uniform(0, total_fitness)
        for i, cumulative_val in enumerate(cumulative_fitness):
            if r <= cumulative_val:
                selected_parents.append(evaluated_population[i]['chromosome'])
                break
    return selected_parents

def crossover(parent1, parent2, num_points, crossover_rate, chromosome_length):
    """Realiza crossover entre dois pais."""
    if random.random() < crossover_rate:
        if num_points == 1:
            point = random.randint(1, chromosome_length - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        elif num_points == 2:
            point1 = random.randint(1, chromosome_length - 2)
            point2 = random.randint(point1 + 1, chromosome_length - 1)
            child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
            child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
            return child1, child2
    return parent1, parent2

def mutate(chromosome, mutation_rate, chromosome_length):
    """Realiza a mutação de inversão de bit."""
    mutated_chromosome = list(chromosome)
    for i in range(chromosome_length):
        if random.random() < mutation_rate:
            mutated_chromosome[i] = '1' if mutated_chromosome[i] == '0' else '0'
    return "".join(mutated_chromosome)


def create_evolution_plot(evolution_data, distribution_type, crossover_points, min_x, max_x):
    """Cria o gráfico interativo da evolução das soluções."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Evolução das soluções - {distribution_type.title()}', 'Convergência do fitness'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Função objetivo
    x_func = np.linspace(min_x, max_x, 1000)
    y_func = x_func * np.sin(10 * np.pi * x_func) + 1
    
    # # NOVO: Calcular o máximo teórico automaticamente
    # theoretical_max_value = np.max(y_func)
    # theoretical_max_x = x_func[np.argmax(y_func)]

    # Función objetivo
    def objective_function(x):
        return x * np.sin(10 * np.pi * x) + 1
    
    # MÁXIMO GLOBAL usando differential_evolution
    result = differential_evolution(lambda x: -objective_function(x[0]), 
                                  [(min_x, max_x)], 
                                  seed=42,
                                  maxiter=1000)
    
    theoretical_max_x = result.x[0]
    theoretical_max_value = objective_function(theoretical_max_x)
    

    
    fig.add_trace(
        go.Scatter(x=x_func, y=y_func, mode='lines', name='f(x) = x sin(10πx) + 1',
                  line=dict(color='black', width=1), opacity=0.7),
        row=1, col=1
    )
    
    # População inicial (pontos pequenos e pretos)
    initial_generation = evolution_data[0]
    initial_x = [item['x_value'] for item in initial_generation['population']]
    initial_y = [item['objective_value'] for item in initial_generation['population']]
    
    fig.add_trace(
        go.Scatter(x=initial_x, y=initial_y, mode='markers', name='População Inicial',
                  marker=dict(color='black', size=4, opacity=0.7)),
        row=1, col=1
    )
    
    # Melhores soluções de cada geração (todas com cor celeste e texto preto)
    best_x_evolution = [gen['best_individual']['x_value'] for gen in evolution_data]
    best_y_evolution = [gen['best_individual']['objective_value'] for gen in evolution_data]
    
    # Adicionar todas as gerações com uma única entrada na legenda
    for i, (x, y) in enumerate(zip(best_x_evolution, best_y_evolution)):
        show_legend = (i == 0)  # Só mostrar na legenda para a primeira iteração
        legend_name = 'Iterações' if show_legend else None
        
        fig.add_trace(
            go.Scatter(
                x=[x], y=[y], 
                mode='markers+text',
                text=[str(i+1)],
                textposition="middle center",
                textfont=dict(color="black", size=8),
                name=legend_name,
                showlegend=show_legend,
                marker=dict(
                    color='lightblue',
                    size=12,
                    line=dict(color='black', width=1)
                ),
                hovertemplate=f"Geração {i+1}<br>x: %{{x:.4f}}<br>f(x): %{{y:.4f}}<extra></extra>"
            ),
            row=1, col=1
        )
    
    # Gráfico de convergência
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

    # MODIFICADO: Línea de referencia usando el máximo calculado automáticamente
    fig.add_trace(
        go.Scatter(x=generations, y=[theoretical_max_value]*len(generations), 
                  mode='lines', name=f'Máximo teórico (x={theoretical_max_x:.3f})',
                  line=dict(color='green', dash='dot')),
        row=1, col=2
    )
    
    # MODIFICADO: Configurar layout con rangos dinámicos
    # Calcular margen para visualización completa de la función
    y_margin = (np.max(y_func) - np.min(y_func)) * 0.05  # 5% de margen
    
    fig.update_xaxes(title_text="x", row=1, col=1, range=[min_x, max_x])
    fig.update_yaxes(title_text="f(x)", row=1, col=1, 
                     range=[np.min(y_func) - y_margin, np.max(y_func) + y_margin])
    fig.update_xaxes(title_text="Geração", row=1, col=2)
    fig.update_yaxes(title_text="Valor de f(x)", row=1, col=2, 
                     range=[None, max(theoretical_max_value, max(best_fitness)) * 1.05])
    
    crossover_type = "ponto único" if crossover_points == 1 else "dois pontos"
    fig.update_layout(
        title=f"Algoritmo genético - Crossover {crossover_type}",
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
    
    return fig, theoretical_max_x, theoretical_max_value







def run_genetic_algorithm_streamlit(params):
    """Executa o algoritmo genético com os parâmetros fornecidos."""
    # Extrair parâmetros
    min_x = params['min_x']
    max_x = params['max_x']
    chromosome_length = params['chromosome_length']
    population_size = params['population_size']
    num_generations = params['num_generations']
    crossover_rate = params['crossover_rate']
    mutation_rate = params['mutation_rate']
    use_elitism = params['use_elitism']
    crossover_points = params['crossover_points']
    distribution_type = params['distribution_type']
    rank_max_fitness = params['rank_max_fitness']
    rank_min_fitness = params['rank_min_fitness']
    
    # Gerar população inicial
    population = generate_initial_population(population_size, chromosome_length, distribution_type, min_x, max_x)
    
    # Variáveis de controle
    best_overall_individual = None
    best_overall_objective_value = -float('inf')
    best_objective_per_generation = []
    avg_objective_per_generation = []
    evolution_data = []
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for generation in range(num_generations):
        # Atualizar progress bar
        progress = (generation + 1) / num_generations
        progress_bar.progress(progress)
        status_text.text(f'Executando geração {generation + 1}/{num_generations}...')
        
        # Avaliar população
        evaluated_population = evaluate_population(population, min_x, max_x, chromosome_length, rank_max_fitness, rank_min_fitness)
        
        # Encontrar melhor indivíduo
        current_best_individual_data = max(evaluated_population, key=lambda item: item['objective_value'])
        current_best_objective = current_best_individual_data['objective_value']
        
        # Atualizar melhor global
        if current_best_objective > best_overall_objective_value:
            best_overall_objective_value = current_best_objective
            best_overall_individual = current_best_individual_data['chromosome']
        
        # Calcular média
        avg_objective = sum(item['objective_value'] for item in evaluated_population) / population_size
        
        # Armazenar dados
        best_objective_per_generation.append(current_best_objective)
        avg_objective_per_generation.append(avg_objective)
        evolution_data.append({
            'generation': generation + 1,
            'population': evaluated_population.copy(),
            'best_individual': current_best_individual_data.copy(),
            'avg_objective': avg_objective
        })
        
        # Seleção
        parents = select_parents(evaluated_population, population_size)
        
        # Nova população
        next_population = []
        
        # Elitismo
        if use_elitism:
            next_population.append(current_best_individual_data['chromosome'])
        
        # Crossover e mutação
        while len(next_population) < population_size:
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            child1, child2 = crossover(p1, p2, crossover_points, crossover_rate, chromosome_length)
            child1 = mutate(child1, mutation_rate, chromosome_length)
            child2 = mutate(child2, mutation_rate, chromosome_length)
            next_population.append(child1)
            if len(next_population) < population_size:
                next_population.append(child2)
        
        population = next_population
        
        # Pequeno delay para visualizar o progresso
        time.sleep(0.01)
    
    # Limpar progress bar
    progress_bar.empty()
    status_text.empty()
    
    return {
        'evolution_data': evolution_data,
        'best_individual': best_overall_individual,
        'best_objective': best_overall_objective_value,
        'best_x': decode_chromosome(best_overall_individual, min_x, max_x, chromosome_length),
        'best_per_generation': best_objective_per_generation,
        'avg_per_generation': avg_objective_per_generation
    }

# --- Interface Streamlit ---


def main():
    st.title("Dashboard dos algoritmos genéticos")
    st.markdown("**Implementação interativa do tutorial de algoritmos genéticos**")

    # random.seed(42)

    # Sidebar com parâmetros
    st.sidebar.header("⚙️ Parâmetros do algoritmo")
    
    # Parâmetros do domínio
    st.sidebar.subheader("Domínio da função")
    min_x = st.sidebar.number_input("Limite inferior (min_x)", value=-1.0, step=0.1)
    max_x = st.sidebar.number_input("Limite superior (max_x)", value=2.0, step=0.1)
    chromosome_length = st.sidebar.slider("Comprimento do cromossomo", 10, 30, 22)
    
    # Parâmetros da população
    st.sidebar.subheader("População")
    population_size = st.sidebar.slider("Tamanho da população", 10, 100, 30)
    distribution_type = st.sidebar.selectbox("Distribuição inicial", ["random", "equidistant"])
    
    # Parâmetros evolutivos
    st.sidebar.subheader("Operadores genéticos")
    num_generations = st.sidebar.slider("Número de gerações", 10, 200, 25)
    crossover_points = st.sidebar.selectbox("Pontos de crossover", [1, 2])
    crossover_rate = st.sidebar.slider("Taxa de crossover", 0.0, 1.0, 0.8, 0.05)
    mutation_rate = st.sidebar.slider("Taxa de mutação", 0.001, 0.1, 0.01, 0.001)
    use_elitism = st.sidebar.checkbox("Usar elitismo", value=True)
    
    # Parâmetros de aptidão
    st.sidebar.subheader("Aptidão")
    rank_max_fitness = st.sidebar.number_input("Aptidão máxima", value=2.0, step=0.1)
    rank_min_fitness = st.sidebar.number_input("Aptidão mínima", value=0.0, step=0.1)
    
    # Botão para executar
    if st.sidebar.button("Executar algoritmo", type="primary"):
        # Preparar parâmetros
        params = {
            'min_x': min_x,
            'max_x': max_x,
            'chromosome_length': chromosome_length,
            'population_size': population_size,
            'num_generations': num_generations,
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate,
            'use_elitism': use_elitism,
            'crossover_points': crossover_points,
            'distribution_type': distribution_type,
            'rank_max_fitness': rank_max_fitness,
            'rank_min_fitness': rank_min_fitness
        }
        
        # Executar algoritmo
        with st.spinner("Executando algoritmo genético..."):
            results = run_genetic_algorithm_streamlit(params)
        
        # Armazenar resultados no session state
        st.session_state.results = results
        st.session_state.params = params
        st.success("✅ Algoritmo executado com sucesso!")
    
    # Mostrar resultados se existirem
    if 'results' in st.session_state:
        results = st.session_state.results
        params = st.session_state.params
        
        # Gráfico principal CON CAPTURA de valores teóricos
        st.subheader("📊 Visualização da evolução")
        fig, theoretical_max_x, theoretical_max_value = create_evolution_plot(
            results['evolution_data'], 
            params['distribution_type'], 
            params['crossover_points'], 
            params['min_x'], 
            params['max_x']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Métricas principais
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Melhor f(x)", f"{results['best_objective']:.6f}")
        
        with col2:
            st.metric("Melhor x", f"{results['best_x']:.6f}")
        
        with col3:
            st.metric("Máximo teórico", f"{theoretical_max_value:.6f}")
        
        with col4:
            crossover_type = "Ponto único" if params['crossover_points'] == 1 else "Dois pontos"
            st.metric("Crossover", crossover_type)
        
        with col5:
            elitism_status = "Ativado" if params['use_elitism'] else "Desativado"
            st.metric("Elitismo", elitism_status)
        
        # Informações detalhadas
        with st.expander("📋 Informações detalhadas"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Melhor cromossomo")
                st.code(results['best_individual'])
                
                st.subheader("Parâmetros utilizados")
                param_df = pd.DataFrame([
                    {"Parâmetro": "População", "Valor": params['population_size']},
                    {"Parâmetro": "Gerações", "Valor": params['num_generations']},
                    {"Parâmetro": "Taxa crossover", "Valor": f"{params['crossover_rate']:.1%}"},
                    {"Parâmetro": "Taxa mutação", "Valor": f"{params['mutation_rate']:.1%}"},
                    {"Parâmetro": "Distribuição", "Valor": params['distribution_type'].title()},
                ])
                
                # Función helper para limpiar DataFrame
                def clean_dataframe_for_streamlit(df):
                    """Limpia el DataFrame para evitar errores de PyArrow en Streamlit"""
                    df_clean = df.copy()
                    
                    for col in df_clean.columns:
                        if df_clean[col].dtype == 'object':
                            # Intentar convertir porcentajes a números
                            if df_clean[col].astype(str).str.contains('%').any():
                                try:
                                    df_clean[col] = pd.to_numeric(
                                        df_clean[col].astype(str).str.replace('%', ''), 
                                        errors='coerce'
                                    )
                                except:
                                    pass
                    
                    return df_clean
                
                st.dataframe(clean_dataframe_for_streamlit(param_df), hide_index=True)
            
            with col2:
                st.subheader("Estatísticas por geração")
                stats_df = pd.DataFrame({
                    'Geração': range(1, len(results['best_per_generation']) + 1),
                    'Melhor f(x)': results['best_per_generation'],
                    'Média f(x)': results['avg_per_generation']
                })
                st.dataframe(stats_df, height=300)
        
        # Comparação com máximo teórico - MENSAJE SIMPLE
        gap = theoretical_max_value - results['best_objective']
        gap_percent = (gap / theoretical_max_value) * 100 if theoretical_max_value != 0 else 0
        
        st.info(f"""
        **📈 Comparação com máximo teórico:**
        - Máximo teórico: f({theoretical_max_x:.5f}) = {theoretical_max_value:.5f}
        - Melhor encontrado: f({results['best_x']:.5f}) = {results['best_objective']:.5f}
        - Gap: {gap:.5f} ({gap_percent:.2f}%)
        """)

if __name__ == "__main__":
    main()