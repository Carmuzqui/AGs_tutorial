# import streamlit as st
# import math
# import random
# import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import pandas as pd
# from datetime import datetime
# import time
# from scipy.optimize import differential_evolution

# # Configuração da página
# st.set_page_config(
#     page_title="Algoritmos Genéticos",
#     page_icon="🧬",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- Funções do Algoritmo Genético ---

# # Função para converter um valor real em representação binária
# # Transforma um número decimal em uma cadeia de bits de comprimento fixo
# def codificar_x(valor, min_x, max_x, comprimento_cromossomo):
#     """Codifica um valor real 'x' em uma cadeia de bits."""
#     valor = max(min_x, min(max_x, valor))
#     amplitude_intervalo = max_x - min_x
#     valor_maximo_b10 = (2**comprimento_cromossomo) - 1
#     b10 = round((valor - min_x) * valor_maximo_b10 / amplitude_intervalo)
#     cadeia_binaria = bin(int(b10))[2:].zfill(comprimento_cromossomo)
#     return cadeia_binaria

# # Função para converter uma representação binária de volta para valor real
# # Transforma uma cadeia de bits em um número decimal dentro do intervalo especificado
# def decodificar_cromossomo(cadeia_cromossomo, min_x, max_x, comprimento_cromossomo):
#     """Decodifica uma cadeia de bits em um valor real 'x'."""
#     b10 = int(cadeia_cromossomo, 2)
#     valor_maximo_b10 = (2**comprimento_cromossomo) - 1
#     x = min_x + (max_x - min_x) * b10 / valor_maximo_b10
#     return x

# # Função objetivo que queremos otimizar
# # Define o problema matemático: f(x) = x * sin(10*pi*x) + 1
# def funcao_objetivo(x):
#     """Implementa a função objetivo f(x) = x * sin(10*pi*x) + 1."""
#     return x * math.sin(10 * math.pi * x) + 1

# # Função para criar a população inicial de cromossomos
# # Gera indivíduos iniciais de forma aleatória ou equidistante
# def gerar_populacao_inicial(tamanho, comprimento, tipo_distribuicao, min_x, max_x):
#     """Gera a população inicial de cromossomos."""
#     populacao = []
#     if tipo_distribuicao == "aleatoria":
#         for _ in range(tamanho):
#             cromossomo = ''.join(random.choice('01') for _ in range(comprimento))
#             populacao.append(cromossomo)
#     elif tipo_distribuicao == "equidistante":
#         if tamanho == 1:
#             valor_x = (min_x + max_x) / 2
#             populacao.append(codificar_x(valor_x, min_x, max_x, comprimento))
#         else:
#             tamanho_passo = (max_x - min_x) / (tamanho - 1)
#             for i in range(tamanho):
#                 valor_x = min_x + i * tamanho_passo
#                 populacao.append(codificar_x(valor_x, min_x, max_x, comprimento))
#     return populacao

# # Função para avaliar a qualidade (fitness) de cada cromossomo
# # Calcula o valor da função objetivo e atribui aptidão baseada no ranking
# def avaliar_populacao(populacao, min_x, max_x, comprimento_cromossomo, aptidao_maxima_rank, aptidao_minima_rank):
#     """Avalia a aptidão de cada cromossomo na população."""
#     cromossomos_avaliados = []
#     for cromossomo in populacao:
#         valor_x = decodificar_cromossomo(cromossomo, min_x, max_x, comprimento_cromossomo)
#         valor_objetivo = funcao_objetivo(valor_x)
#         cromossomos_avaliados.append({
#             'cromossomo': cromossomo,
#             'valor_x': valor_x,
#             'valor_objetivo': valor_objetivo,
#             'aptidao': 0
#         })
    
#     # Ordenar e atribuir aptidão ranqueada
#     cromossomos_avaliados.sort(key=lambda item: item['valor_objetivo'], reverse=True)
#     N = len(cromossomos_avaliados)
#     for i, item in enumerate(cromossomos_avaliados):
#         if N > 1:
#             item['aptidao'] = aptidao_minima_rank + (aptidao_maxima_rank - aptidao_minima_rank) * (N - 1 - i) / (N - 1)
#         else:
#             item['aptidao'] = aptidao_maxima_rank
    
#     return cromossomos_avaliados

# # Função para selecionar pais para reprodução
# # Implementa o método da roleta viciada baseado na aptidão
# def selecionar_pais(populacao_avaliada, tamanho_populacao):
#     """Seleciona os pais usando o algoritmo da roleta."""
#     aptidao_total = sum(item['aptidao'] for item in populacao_avaliada)
    
#     if aptidao_total == 0:
#         return [random.choice([item['cromossomo'] for item in populacao_avaliada]) for _ in range(tamanho_populacao)]

#     aptidao_cumulativa = []
#     cumulativo_atual = 0
#     for item in populacao_avaliada:
#         cumulativo_atual += item['aptidao']
#         aptidao_cumulativa.append(cumulativo_atual)
        
#     pais_selecionados = []
#     for _ in range(tamanho_populacao):
#         r = random.uniform(0, aptidao_total)
#         for i, valor_cumulativo in enumerate(aptidao_cumulativa):
#             if r <= valor_cumulativo:
#                 pais_selecionados.append(populacao_avaliada[i]['cromossomo'])
#                 break
#     return pais_selecionados

# # Função para realizar cruzamento entre dois pais
# # Combina material genético de dois cromossomos para gerar descendência
# def cruzamento(pai1, pai2, num_pontos, taxa_cruzamento, comprimento_cromossomo):
#     """Realiza cruzamento entre dois pais."""
#     if random.random() < taxa_cruzamento:
#         if num_pontos == 1:
#             ponto = random.randint(1, comprimento_cromossomo - 1)
#             filho1 = pai1[:ponto] + pai2[ponto:]
#             filho2 = pai2[:ponto] + pai1[ponto:]
#             return filho1, filho2
#         elif num_pontos == 2:
#             ponto1 = random.randint(1, comprimento_cromossomo - 2)
#             ponto2 = random.randint(ponto1 + 1, comprimento_cromossomo - 1)
#             filho1 = pai1[:ponto1] + pai2[ponto1:ponto2] + pai1[ponto2:]
#             filho2 = pai2[:ponto1] + pai1[ponto1:ponto2] + pai2[ponto2:]
#             return filho1, filho2
#     return pai1, pai2

# # Função para introduzir mutações nos cromossomos
# # Inverte bits aleatoriamente para manter diversidade genética
# def mutar(cromossomo, taxa_mutacao, comprimento_cromossomo):
#     """Realiza a mutação de inversão de bit."""
#     cromossomo_mutado = list(cromossomo)
#     for i in range(comprimento_cromossomo):
#         if random.random() < taxa_mutacao:
#             cromossomo_mutado[i] = '1' if cromossomo_mutado[i] == '0' else '0'
#     return "".join(cromossomo_mutado)

# # Função para criar visualização gráfica da evolução
# # Gera gráficos interativos mostrando a convergência do algoritmo
# def criar_grafico_evolucao(dados_evolucao, tipo_distribuicao, pontos_cruzamento, min_x, max_x):
#     """Cria o gráfico interativo da evolução das soluções."""
#     fig = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=(f'Evolução das soluções - {tipo_distribuicao.title()}', 'Convergência da aptidão'),
#         specs=[[{"secondary_y": False}, {"secondary_y": False}]]
#     )
    
#     # Função objetivo
#     x_func = np.linspace(min_x, max_x, 1000)
#     y_func = x_func * np.sin(10 * np.pi * x_func) + 1
    
#     # Função objetivo para otimização
#     def funcao_objetivo_otimizacao(x):
#         return x * np.sin(10 * np.pi * x) + 1
    
#     # MÁXIMO GLOBAL usando differential_evolution
#     resultado = differential_evolution(lambda x: -funcao_objetivo_otimizacao(x[0]), 
#                                      [(min_x, max_x)], 
#                                      seed=42,
#                                      maxiter=1000)
    
#     x_maximo_teorico = resultado.x[0]
#     valor_maximo_teorico = funcao_objetivo_otimizacao(x_maximo_teorico)
    
#     fig.add_trace(
#         go.Scatter(x=x_func, y=y_func, mode='lines', name='f(x) = x sin(10πx) + 1',
#                   line=dict(color='black', width=1), opacity=0.7),
#         row=1, col=1
#     )
    
#     # População inicial (pontos pequenos e pretos)
#     geracao_inicial = dados_evolucao[0]
#     x_inicial = [item['valor_x'] for item in geracao_inicial['populacao']]
#     y_inicial = [item['valor_objetivo'] for item in geracao_inicial['populacao']]
    
#     fig.add_trace(
#         go.Scatter(x=x_inicial, y=y_inicial, mode='markers', name='População Inicial',
#                   marker=dict(color='black', size=4, opacity=0.7)),
#         row=1, col=1
#     )
    
#     # Melhores soluções de cada geração (todas com cor celeste e texto preto)
#     evolucao_melhor_x = [gen['melhor_individuo']['valor_x'] for gen in dados_evolucao]
#     evolucao_melhor_y = [gen['melhor_individuo']['valor_objetivo'] for gen in dados_evolucao]
    
#     # Adicionar todas as gerações com uma única entrada na legenda
#     for i, (x, y) in enumerate(zip(evolucao_melhor_x, evolucao_melhor_y)):
#         mostrar_legenda = (i == 0)  # Só mostrar na legenda para a primeira iteração
#         nome_legenda = 'Iterações' if mostrar_legenda else None
        
#         fig.add_trace(
#             go.Scatter(
#                 x=[x], y=[y], 
#                 mode='markers+text',
#                 text=[str(i+1)],
#                 textposition="middle center",
#                 textfont=dict(color="black", size=8),
#                 name=nome_legenda,
#                 showlegend=mostrar_legenda,
#                 marker=dict(
#                     color='lightblue',
#                     size=12,
#                     line=dict(color='black', width=1)
#                 ),
#                 hovertemplate=f"Geração {i+1}<br>x: %{{x:.4f}}<br>f(x): %{{y:.4f}}<extra></extra>"
#             ),
#             row=1, col=1
#         )
    
#     # Gráfico de convergência
#     geracoes = list(range(1, len(dados_evolucao) + 1))
#     melhor_aptidao = [gen['melhor_individuo']['valor_objetivo'] for gen in dados_evolucao]
#     aptidao_media = [gen['objetivo_medio'] for gen in dados_evolucao]
    
#     fig.add_trace(
#         go.Scatter(x=geracoes, y=melhor_aptidao, mode='lines+markers', name='Melhor f(x)',
#                   line=dict(color='blue'), marker=dict(size=4)),
#         row=1, col=2
#     )
    
#     fig.add_trace(
#         go.Scatter(x=geracoes, y=aptidao_media, mode='lines+markers', name='Média f(x)',
#                   line=dict(color='red', dash='dash'), marker=dict(size=3, symbol='square')),
#         row=1, col=2
#     )

#     # Linha de referência usando o máximo calculado automaticamente
#     fig.add_trace(
#         go.Scatter(x=geracoes, y=[valor_maximo_teorico]*len(geracoes), 
#                   mode='lines', name=f'Máximo teórico (x={x_maximo_teorico:.3f})',
#                   line=dict(color='green', dash='dot')),
#         row=1, col=2
#     )
    
#     # Configurar layout com rangos dinâmicos
#     # Calcular margem para visualização completa da função
#     margem_y = (np.max(y_func) - np.min(y_func)) * 0.05  # 5% de margem
    
#     fig.update_xaxes(title_text="x", row=1, col=1, range=[min_x, max_x])
#     fig.update_yaxes(title_text="f(x)", row=1, col=1, 
#                      range=[np.min(y_func) - margem_y, np.max(y_func) + margem_y])
#     fig.update_xaxes(title_text="Geração", row=1, col=2)
#     fig.update_yaxes(title_text="Valor de f(x)", row=1, col=2, 
#                      range=[None, max(valor_maximo_teorico, max(melhor_aptidao)) * 1.05])
    
#     tipo_cruzamento = "ponto único" if pontos_cruzamento == 1 else "dois pontos"
#     fig.update_layout(
#         title=f"Algoritmo Genético - Cruzamento {tipo_cruzamento}",
#         height=600,
#         showlegend=True,
#         legend=dict(
#             orientation="v",
#             yanchor="top",
#             y=1,
#             xanchor="left",
#             x=1.02
#         )
#     )
    
#     return fig, x_maximo_teorico, valor_maximo_teorico

# # Função principal que coordena todo o processo evolutivo
# # Executa o algoritmo genético completo com todas as etapas
# def executar_algoritmo_genetico_streamlit(parametros):
#     """Executa o algoritmo genético com os parâmetros fornecidos."""
#     # Extrair parâmetros
#     min_x = parametros['min_x']
#     max_x = parametros['max_x']
#     comprimento_cromossomo = parametros['comprimento_cromossomo']
#     tamanho_populacao = parametros['tamanho_populacao']
#     num_geracoes = parametros['num_geracoes']
#     taxa_cruzamento = parametros['taxa_cruzamento']
#     taxa_mutacao = parametros['taxa_mutacao']
#     usar_elitismo = parametros['usar_elitismo']
#     pontos_cruzamento = parametros['pontos_cruzamento']
#     tipo_distribuicao = parametros['tipo_distribuicao']
#     aptidao_maxima_rank = parametros['aptidao_maxima_rank']
#     aptidao_minima_rank = parametros['aptidao_minima_rank']
    
#     # Gerar população inicial
#     populacao = gerar_populacao_inicial(tamanho_populacao, comprimento_cromossomo, tipo_distribuicao, min_x, max_x)
    
#     # Variáveis de controle
#     melhor_individuo_geral = None
#     melhor_valor_objetivo_geral = -float('inf')
#     melhor_objetivo_por_geracao = []
#     objetivo_medio_por_geracao = []
#     dados_evolucao = []
    
#     # Barra de progresso
#     barra_progresso = st.progress(0)
#     texto_status = st.empty()
    
#     for geracao in range(num_geracoes):
#         # Atualizar barra de progresso
#         progresso = (geracao + 1) / num_geracoes
#         barra_progresso.progress(progresso)
#         texto_status.text(f'Executando geração {geracao + 1}/{num_geracoes}...')
        
#         # Avaliar população
#         populacao_avaliada = avaliar_populacao(populacao, min_x, max_x, comprimento_cromossomo, aptidao_maxima_rank, aptidao_minima_rank)
        
#         # Encontrar melhor indivíduo
#         dados_melhor_individuo_atual = max(populacao_avaliada, key=lambda item: item['valor_objetivo'])
#         melhor_objetivo_atual = dados_melhor_individuo_atual['valor_objetivo']
        
#         # Atualizar melhor global
#         if melhor_objetivo_atual > melhor_valor_objetivo_geral:
#             melhor_valor_objetivo_geral = melhor_objetivo_atual
#             melhor_individuo_geral = dados_melhor_individuo_atual['cromossomo']
        
#         # Calcular média
#         objetivo_medio = sum(item['valor_objetivo'] for item in populacao_avaliada) / tamanho_populacao
        
#         # Armazenar dados
#         melhor_objetivo_por_geracao.append(melhor_objetivo_atual)
#         objetivo_medio_por_geracao.append(objetivo_medio)
#         dados_evolucao.append({
#             'geracao': geracao + 1,
#             'populacao': populacao_avaliada.copy(),
#             'melhor_individuo': dados_melhor_individuo_atual.copy(),
#             'objetivo_medio': objetivo_medio
#         })
        
#         # Seleção
#         pais = selecionar_pais(populacao_avaliada, tamanho_populacao)
        
#         # Nova população
#         proxima_populacao = []
        
#         # Elitismo
#         if usar_elitismo:
#             proxima_populacao.append(dados_melhor_individuo_atual['cromossomo'])
        
#         # Cruzamento e mutação
#         while len(proxima_populacao) < tamanho_populacao:
#             p1 = random.choice(pais)
#             p2 = random.choice(pais)
#             filho1, filho2 = cruzamento(p1, p2, pontos_cruzamento, taxa_cruzamento, comprimento_cromossomo)
#             filho1 = mutar(filho1, taxa_mutacao, comprimento_cromossomo)
#             filho2 = mutar(filho2, taxa_mutacao, comprimento_cromossomo)
#             proxima_populacao.append(filho1)
#             if len(proxima_populacao) < tamanho_populacao:
#                 proxima_populacao.append(filho2)
        
#         populacao = proxima_populacao
        
#         # Pequeno delay para visualizar o progresso
#         time.sleep(0.01)
    
#     # Limpar barra de progresso
#     barra_progresso.empty()
#     texto_status.empty()
    
#     return {
#         'dados_evolucao': dados_evolucao,
#         'melhor_individuo': melhor_individuo_geral,
#         'melhor_objetivo': melhor_valor_objetivo_geral,
#         'melhor_x': decodificar_cromossomo(melhor_individuo_geral, min_x, max_x, comprimento_cromossomo),
#         'melhor_por_geracao': melhor_objetivo_por_geracao,
#         'media_por_geracao': objetivo_medio_por_geracao
#     }

# # --- Interface Streamlit ---

# # Función principal que maneja toda la interfaz de usuario
# # Crea el dashboard interactivo y coordina la ejecución del algoritmo
# def principal():
#     st.title("Dashboard dos Algoritmos Genéticos")
#     st.markdown("**Implementação interativa do tutorial de algoritmos genéticos**")

#     # Sidebar com parâmetros
#     st.sidebar.header("⚙️ Parâmetros do Algoritmo")
    
#     # Parâmetros do domínio
#     st.sidebar.subheader("Domínio da Função")
#     min_x = st.sidebar.number_input("Limite inferior (min_x)", value=-1.0, step=0.1)
#     max_x = st.sidebar.number_input("Limite superior (max_x)", value=2.0, step=0.1)
#     comprimento_cromossomo = st.sidebar.slider("Comprimento do cromossomo", 10, 30, 22)
    
#     # Parâmetros da população
#     st.sidebar.subheader("População")
#     tamanho_populacao = st.sidebar.slider("Tamanho da população", 10, 100, 30)
#     tipo_distribuicao = st.sidebar.selectbox("Distribuição inicial", ["aleatoria", "equidistante"])
    
#     # Parâmetros evolutivos
#     st.sidebar.subheader("Operadores Genéticos")
#     num_geracoes = st.sidebar.slider("Número de gerações", 10, 200, 25)
#     pontos_cruzamento = st.sidebar.selectbox("Pontos de cruzamento", [1, 2])
#     taxa_cruzamento = st.sidebar.slider("Taxa de cruzamento", 0.0, 1.0, 0.8, 0.05)
#     taxa_mutacao = st.sidebar.slider("Taxa de mutação", 0.001, 0.1, 0.01, 0.001)
#     usar_elitismo = st.sidebar.checkbox("Usar elitismo", value=True)
    
#     # Parâmetros de aptidão
#     st.sidebar.subheader("Aptidão")
#     aptidao_maxima_rank = st.sidebar.number_input("Aptidão máxima", value=2.0, step=0.1)
#     aptidao_minima_rank = st.sidebar.number_input("Aptidão mínima", value=0.0, step=0.1)
    
#     # Botão para executar
#     if st.sidebar.button("Executar Algoritmo", type="primary"):
#         # Preparar parâmetros
#         parametros = {
#             'min_x': min_x,
#             'max_x': max_x,
#             'comprimento_cromossomo': comprimento_cromossomo,
#             'tamanho_populacao': tamanho_populacao,
#             'num_geracoes': num_geracoes,
#             'taxa_cruzamento': taxa_cruzamento,
#             'taxa_mutacao': taxa_mutacao,
#             'usar_elitismo': usar_elitismo,
#             'pontos_cruzamento': pontos_cruzamento,
#             'tipo_distribuicao': tipo_distribuicao,
#             'aptidao_maxima_rank': aptidao_maxima_rank,
#             'aptidao_minima_rank': aptidao_minima_rank
#         }
        
#         # Executar algoritmo
#         with st.spinner("Executando algoritmo genético..."):
#             resultados = executar_algoritmo_genetico_streamlit(parametros)
        
#         # Armazenar resultados no session state
#         st.session_state.resultados = resultados
#         st.session_state.parametros = parametros
#         st.success("✅ Algoritmo executado com sucesso!")
    
#     # Mostrar resultados se existirem
#     if 'resultados' in st.session_state:
#         resultados = st.session_state.resultados
#         parametros = st.session_state.parametros
        
#         # Gráfico principal COM CAPTURA de valores teóricos
#         st.subheader("📊 Visualização da Evolução")
#         fig, x_maximo_teorico, valor_maximo_teorico = criar_grafico_evolucao(
#             resultados['dados_evolucao'], 
#             parametros['tipo_distribuicao'], 
#             parametros['pontos_cruzamento'], 
#             parametros['min_x'], 
#             parametros['max_x']
#         )
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Métricas principais
#         col1, col2, col3, col4, col5 = st.columns(5)
        
#         with col1:
#             st.metric("Melhor f(x)", f"{resultados['melhor_objetivo']:.6f}")
        
#         with col2:
#             st.metric("Melhor x", f"{resultados['melhor_x']:.6f}")
        
#         with col3:
#             st.metric("Máximo Teórico", f"{valor_maximo_teorico:.6f}")
        
#         with col4:
#             tipo_cruzamento = "Ponto único" if parametros['pontos_cruzamento'] == 1 else "Dois pontos"
#             st.metric("Cruzamento", tipo_cruzamento)
        
#         with col5:
#             status_elitismo = "Ativado" if parametros['usar_elitismo'] else "Desativado"
#             st.metric("Elitismo", status_elitismo)
        
#         # Informações detalhadas
#         with st.expander("📋 Informações Detalhadas"):
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.subheader("Melhor Cromossomo")
#                 st.code(resultados['melhor_individuo'])
                
#                 st.subheader("Parâmetros Utilizados")
#                 df_parametros = pd.DataFrame([
#                     {"Parâmetro": "População", "Valor": parametros['tamanho_populacao']},
#                     {"Parâmetro": "Gerações", "Valor": parametros['num_geracoes']},
#                     {"Parâmetro": "Taxa cruzamento", "Valor": f"{parametros['taxa_cruzamento']:.1%}"},
#                     {"Parâmetro": "Taxa mutação", "Valor": f"{parametros['taxa_mutacao']:.1%}"},
#                     {"Parâmetro": "Distribuição", "Valor": parametros['tipo_distribuicao'].title()},
#                 ])
                
#                 # Função auxiliar para limpar DataFrame
#                 def limpar_dataframe_para_streamlit(df):
#                     """Limpa o DataFrame para evitar erros do PyArrow no Streamlit"""
#                     df_limpo = df.copy()
                    
#                     for col in df_limpo.columns:
#                         if df_limpo[col].dtype == 'object':
#                             # Tentar converter porcentagens para números
#                             if df_limpo[col].astype(str).str.contains('%').any():
#                                 try:
#                                     df_limpo[col] = pd.to_numeric(
#                                         df_limpo[col].astype(str).str.replace('%', ''), 
#                                         errors='coerce'
#                                     )
#                                 except:
#                                     pass
                    
#                     return df_limpo
                
#                 st.dataframe(limpar_dataframe_para_streamlit(df_parametros), hide_index=True)
            
#             with col2:
#                 st.subheader("Estatísticas por Geração")
#                 df_estatisticas = pd.DataFrame({
#                     'Geração': range(1, len(resultados['melhor_por_geracao']) + 1),
#                     'Melhor f(x)': resultados['melhor_por_geracao'],
#                     'Média f(x)': resultados['media_por_geracao']
#                 })
#                 st.dataframe(df_estatisticas, height=300)
        
#         # Comparação com máximo teórico - MENSAGEM SIMPLES
#         gap = valor_maximo_teorico - resultados['melhor_objetivo']
#         gap_percentual = (gap / valor_maximo_teorico) * 100 if valor_maximo_teorico != 0 else 0
        
#         st.info(f"""
#         **📈 Comparação com Máximo Teórico:**
#         - Máximo teórico: f({x_maximo_teorico:.5f}) = {valor_maximo_teorico:.5f}
#         - Melhor encontrado: f({resultados['melhor_x']:.5f}) = {resultados['melhor_objetivo']:.5f}
#         - Gap: {gap:.5f} ({gap_percentual:.2f}%)
#         """)

# if __name__ == "__main__":
#     principal()








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

# Configuração da página
st.set_page_config(
    page_title="Algoritmos Genéticos",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Funções do Algoritmo Genético ---

# Função para converter um valor real em representação binária
# Transforma um número decimal em uma cadeia de bits de comprimento fixo
def codificar_x(valor, min_x, max_x, comprimento_cromossomo):
    """Codifica um valor real 'x' em uma cadeia de bits."""
    valor = max(min_x, min(max_x, valor))
    amplitude_intervalo = max_x - min_x
    valor_maximo_b10 = (2**comprimento_cromossomo) - 1
    b10 = round((valor - min_x) * valor_maximo_b10 / amplitude_intervalo)
    cadeia_binaria = bin(int(b10))[2:].zfill(comprimento_cromossomo)
    return cadeia_binaria

# Função para converter uma representação binária de volta para valor real
# Transforma uma cadeia de bits em um número decimal dentro do intervalo especificado
def decodificar_cromossomo(cadeia_cromossomo, min_x, max_x, comprimento_cromossomo):
    """Decodifica uma cadeia de bits em um valor real 'x'."""
    b10 = int(cadeia_cromossomo, 2)
    valor_maximo_b10 = (2**comprimento_cromossomo) - 1
    x = min_x + (max_x - min_x) * b10 / valor_maximo_b10
    return x

# Função objetivo que queremos otimizar
# Define o problema matemático: f(x) = x * sin(10*pi*x) + 1
def funcao_objetivo(x):
    """Implementa a função objetivo f(x) = x * sin(10*pi*x) + 1."""
    return x * math.sin(10 * math.pi * x) + 1

# Função para criar a população inicial de cromossomos
# Gera indivíduos iniciais de forma aleatória ou equidistante
def gerar_populacao_inicial(tamanho, comprimento, tipo_distribuicao, min_x, max_x):
    """Gera a população inicial de cromossomos."""
    populacao = []
    if tipo_distribuicao == "aleatoria":
        for _ in range(tamanho):
            cromossomo = ''.join(random.choice('01') for _ in range(comprimento))
            populacao.append(cromossomo)
    elif tipo_distribuicao == "equidistante":
        if tamanho == 1:
            valor_x = (min_x + max_x) / 2
            populacao.append(codificar_x(valor_x, min_x, max_x, comprimento))
        else:
            tamanho_passo = (max_x - min_x) / (tamanho - 1)
            for i in range(tamanho):
                valor_x = min_x + i * tamanho_passo
                populacao.append(codificar_x(valor_x, min_x, max_x, comprimento))
    return populacao

# Função para avaliar a qualidade (fitness) de cada cromossomo
# Calcula o valor da função objetivo e atribui aptidão baseada no ranking
def avaliar_populacao(populacao, min_x, max_x, comprimento_cromossomo, aptidao_maxima_rank, aptidao_minima_rank):
    """Avalia a aptidão de cada cromossomo na população."""
    cromossomos_avaliados = []
    for cromossomo in populacao:
        valor_x = decodificar_cromossomo(cromossomo, min_x, max_x, comprimento_cromossomo)
        valor_objetivo = funcao_objetivo(valor_x)
        cromossomos_avaliados.append({
            'cromossomo': cromossomo,
            'valor_x': valor_x,
            'valor_objetivo': valor_objetivo,
            'aptidao': 0
        })
    
    # Ordenar e atribuir aptidão ranqueada
    cromossomos_avaliados.sort(key=lambda item: item['valor_objetivo'], reverse=True)
    N = len(cromossomos_avaliados)
    for i, item in enumerate(cromossomos_avaliados):
        if N > 1:
            item['aptidao'] = aptidao_minima_rank + (aptidao_maxima_rank - aptidao_minima_rank) * (N - 1 - i) / (N - 1)
        else:
            item['aptidao'] = aptidao_maxima_rank
    
    return cromossomos_avaliados

# Função para selecionar pais para reprodução
# Implementa o método da roleta viciada baseado na aptidão
def selecionar_pais(populacao_avaliada, tamanho_populacao):
    """Seleciona os pais usando o algoritmo da roleta."""
    aptidao_total = sum(item['aptidao'] for item in populacao_avaliada)
    
    if aptidao_total == 0:
        return [random.choice([item['cromossomo'] for item in populacao_avaliada]) for _ in range(tamanho_populacao)]

    aptidao_cumulativa = []
    cumulativo_atual = 0
    for item in populacao_avaliada:
        cumulativo_atual += item['aptidao']
        aptidao_cumulativa.append(cumulativo_atual)
        
    pais_selecionados = []
    for _ in range(tamanho_populacao):
        r = random.uniform(0, aptidao_total)
        for i, valor_cumulativo in enumerate(aptidao_cumulativa):
            if r <= valor_cumulativo:
                pais_selecionados.append(populacao_avaliada[i]['cromossomo'])
                break
    return pais_selecionados

# Função para realizar cruzamento entre dois pais
# Combina material genético de dois cromossomos para gerar descendência
def cruzamento(pai1, pai2, tipo_cruzamento, taxa_cruzamento, comprimento_cromossomo):
    """Realiza cruzamento entre dois pais."""
    if random.random() < taxa_cruzamento:
        if tipo_cruzamento == "Um ponto":
            ponto = random.randint(1, comprimento_cromossomo - 1)
            filho1 = pai1[:ponto] + pai2[ponto:]
            filho2 = pai2[:ponto] + pai1[ponto:]
            return filho1, filho2
        elif tipo_cruzamento == "Dois pontos":
            ponto1 = random.randint(1, comprimento_cromossomo - 2)
            ponto2 = random.randint(ponto1 + 1, comprimento_cromossomo - 1)
            filho1 = pai1[:ponto1] + pai2[ponto1:ponto2] + pai1[ponto2:]
            filho2 = pai2[:ponto1] + pai1[ponto1:ponto2] + pai2[ponto2:]
            return filho1, filho2
        elif tipo_cruzamento == "Uniforme":
            # Implementação do crossover uniforme (máscara de bits)
            mascara = [random.choice([0, 1]) for _ in range(comprimento_cromossomo)]
            filho1 = ""
            filho2 = ""
            
            for i in range(comprimento_cromossomo):
                if mascara[i] == 1:
                    # Se máscara é 1: filho1 herda de pai1, filho2 herda de pai2
                    filho1 += pai1[i]
                    filho2 += pai2[i]
                else:
                    # Se máscara é 0: filho1 herda de pai2, filho2 herda de pai1
                    filho1 += pai2[i]
                    filho2 += pai1[i]
            
            return filho1, filho2
    return pai1, pai2

# Função para introduzir mutações nos cromossomos
# Inverte bits aleatoriamente para manter diversidade genética
def mutar(cromossomo, taxa_mutacao, comprimento_cromossomo):
    """Realiza a mutação de inversão de bit."""
    cromossomo_mutado = list(cromossomo)
    for i in range(comprimento_cromossomo):
        if random.random() < taxa_mutacao:
            cromossomo_mutado[i] = '1' if cromossomo_mutado[i] == '0' else '0'
    return "".join(cromossomo_mutado)

# Função para criar visualização gráfica da evolução
# Gera gráficos interativos mostrando a convergência do algoritmo
def criar_grafico_evolucao(dados_evolucao, tipo_distribuicao, tipo_cruzamento, min_x, max_x):
    """Cria o gráfico interativo da evolução das soluções."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Evolução das soluções - {tipo_distribuicao.title()}', 'Convergência da aptidão'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Função objetivo
    x_func = np.linspace(min_x, max_x, 1000)
    y_func = x_func * np.sin(10 * np.pi * x_func) + 1
    
    # Função objetivo para otimização
    def funcao_objetivo_otimizacao(x):
        return x * np.sin(10 * np.pi * x) + 1
    
    # MÁXIMO GLOBAL usando differential_evolution
    resultado = differential_evolution(lambda x: -funcao_objetivo_otimizacao(x[0]), 
                                     [(min_x, max_x)], 
                                     seed=42,
                                     maxiter=1000)
    
    x_maximo_teorico = resultado.x[0]
    valor_maximo_teorico = funcao_objetivo_otimizacao(x_maximo_teorico)
    
    fig.add_trace(
        go.Scatter(x=x_func, y=y_func, mode='lines', name='f(x) = x sin(10πx) + 1',
                  line=dict(color='black', width=1), opacity=0.7),
        row=1, col=1
    )
    
    # População inicial (pontos pequenos e pretos)
    geracao_inicial = dados_evolucao[0]
    x_inicial = [item['valor_x'] for item in geracao_inicial['populacao']]
    y_inicial = [item['valor_objetivo'] for item in geracao_inicial['populacao']]
    
    fig.add_trace(
        go.Scatter(x=x_inicial, y=y_inicial, mode='markers', name='População Inicial',
                  marker=dict(color='black', size=4, opacity=0.7)),
        row=1, col=1
    )
    
    # Melhores soluções de cada geração (todas com cor celeste e texto preto)
    evolucao_melhor_x = [gen['melhor_individuo']['valor_x'] for gen in dados_evolucao]
    evolucao_melhor_y = [gen['melhor_individuo']['valor_objetivo'] for gen in dados_evolucao]
    
    # Adicionar todas as gerações com uma única entrada na legenda
    for i, (x, y) in enumerate(zip(evolucao_melhor_x, evolucao_melhor_y)):
        mostrar_legenda = (i == 0)  # Só mostrar na legenda para a primeira iteração
        nome_legenda = 'Iterações' if mostrar_legenda else None
        
        fig.add_trace(
            go.Scatter(
                x=[x], y=[y], 
                mode='markers+text',
                text=[str(i+1)],
                textposition="middle center",
                textfont=dict(color="black", size=8),
                name=nome_legenda,
                showlegend=mostrar_legenda,
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
    geracoes = list(range(1, len(dados_evolucao) + 1))
    melhor_aptidao = [gen['melhor_individuo']['valor_objetivo'] for gen in dados_evolucao]
    aptidao_media = [gen['objetivo_medio'] for gen in dados_evolucao]
    
    fig.add_trace(
        go.Scatter(x=geracoes, y=melhor_aptidao, mode='lines+markers', name='Melhor f(x)',
                  line=dict(color='blue'), marker=dict(size=4)),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=geracoes, y=aptidao_media, mode='lines+markers', name='Média f(x)',
                  line=dict(color='red', dash='dash'), marker=dict(size=3, symbol='square')),
        row=1, col=2
    )

    # Linha de referência usando o máximo calculado automaticamente
    fig.add_trace(
        go.Scatter(x=geracoes, y=[valor_maximo_teorico]*len(geracoes), 
                  mode='lines', name=f'Máximo teórico (x={x_maximo_teorico:.3f})',
                  line=dict(color='green', dash='dot')),
        row=1, col=2
    )
    
    # Configurar layout com rangos dinâmicos
    # Calcular margem para visualização completa da função
    margem_y = (np.max(y_func) - np.min(y_func)) * 0.05  # 5% de margem
    
    fig.update_xaxes(title_text="x", row=1, col=1, range=[min_x, max_x])
    fig.update_yaxes(title_text="f(x)", row=1, col=1, 
                     range=[np.min(y_func) - margem_y, np.max(y_func) + margem_y])
    fig.update_xaxes(title_text="Geração", row=1, col=2)
    fig.update_yaxes(title_text="Valor de f(x)", row=1, col=2, 
                     range=[None, max(valor_maximo_teorico, max(melhor_aptidao)) * 1.05])
    
    fig.update_layout(
        title=f"Algoritmo Genético - Cruzamento {tipo_cruzamento}",
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
    
    return fig, x_maximo_teorico, valor_maximo_teorico

# Função principal que coordena todo o processo evolutivo
# Executa o algoritmo genético completo com todas as etapas
def executar_algoritmo_genetico_streamlit(parametros):
    """Executa o algoritmo genético com os parâmetros fornecidos."""
    # Extrair parâmetros
    min_x = parametros['min_x']
    max_x = parametros['max_x']
    comprimento_cromossomo = parametros['comprimento_cromossomo']
    tamanho_populacao = parametros['tamanho_populacao']
    num_geracoes = parametros['num_geracoes']
    taxa_cruzamento = parametros['taxa_cruzamento']
    taxa_mutacao = parametros['taxa_mutacao']
    usar_elitismo = parametros['usar_elitismo']
    tipo_cruzamento = parametros['tipo_cruzamento']
    tipo_distribuicao = parametros['tipo_distribuicao']
    aptidao_maxima_rank = parametros['aptidao_maxima_rank']
    aptidao_minima_rank = parametros['aptidao_minima_rank']
    
    # Gerar população inicial
    populacao = gerar_populacao_inicial(tamanho_populacao, comprimento_cromossomo, tipo_distribuicao, min_x, max_x)
    
    # Variáveis de controle
    melhor_individuo_geral = None
    melhor_valor_objetivo_geral = -float('inf')
    melhor_objetivo_por_geracao = []
    objetivo_medio_por_geracao = []
    dados_evolucao = []
    
    # Barra de progresso
    barra_progresso = st.progress(0)
    texto_status = st.empty()
    
    for geracao in range(num_geracoes):
        # Atualizar barra de progresso
        progresso = (geracao + 1) / num_geracoes
        barra_progresso.progress(progresso)
        texto_status.text(f'Executando geração {geracao + 1}/{num_geracoes}...')
        
        # Avaliar população
        populacao_avaliada = avaliar_populacao(populacao, min_x, max_x, comprimento_cromossomo, aptidao_maxima_rank, aptidao_minima_rank)
        
        # Encontrar melhor indivíduo
        dados_melhor_individuo_atual = max(populacao_avaliada, key=lambda item: item['valor_objetivo'])
        melhor_objetivo_atual = dados_melhor_individuo_atual['valor_objetivo']
        
        # Atualizar melhor global
        if melhor_objetivo_atual > melhor_valor_objetivo_geral:
            melhor_valor_objetivo_geral = melhor_objetivo_atual
            melhor_individuo_geral = dados_melhor_individuo_atual['cromossomo']
        
        # Calcular média
        objetivo_medio = sum(item['valor_objetivo'] for item in populacao_avaliada) / tamanho_populacao
        
        # Armazenar dados
        melhor_objetivo_por_geracao.append(melhor_objetivo_atual)
        objetivo_medio_por_geracao.append(objetivo_medio)
        dados_evolucao.append({
            'geracao': geracao + 1,
            'populacao': populacao_avaliada.copy(),
            'melhor_individuo': dados_melhor_individuo_atual.copy(),
            'objetivo_medio': objetivo_medio
        })
        
        # Seleção
        pais = selecionar_pais(populacao_avaliada, tamanho_populacao)
        
        # Nova população
        proxima_populacao = []
        
        # Elitismo
        if usar_elitismo:
            proxima_populacao.append(dados_melhor_individuo_atual['cromossomo'])
        
        # Cruzamento e mutação
        while len(proxima_populacao) < tamanho_populacao:
            p1 = random.choice(pais)
            p2 = random.choice(pais)
            filho1, filho2 = cruzamento(p1, p2, tipo_cruzamento, taxa_cruzamento, comprimento_cromossomo)
            filho1 = mutar(filho1, taxa_mutacao, comprimento_cromossomo)
            filho2 = mutar(filho2, taxa_mutacao, comprimento_cromossomo)
            proxima_populacao.append(filho1)
            if len(proxima_populacao) < tamanho_populacao:
                proxima_populacao.append(filho2)
        
        populacao = proxima_populacao
        
        # Pequeno delay para visualizar o progresso
        time.sleep(0.01)
    
    # Limpar barra de progresso
    barra_progresso.empty()
    texto_status.empty()
    
    return {
        'dados_evolucao': dados_evolucao,
        'melhor_individuo': melhor_individuo_geral,
        'melhor_objetivo': melhor_valor_objetivo_geral,
        'melhor_x': decodificar_cromossomo(melhor_individuo_geral, min_x, max_x, comprimento_cromossomo),
        'melhor_por_geracao': melhor_objetivo_por_geracao,
        'media_por_geracao': objetivo_medio_por_geracao
    }

# --- Interface Streamlit ---

# Función principal que maneja toda la interfaz de usuario
# Crea el dashboard interactivo e coordina a execução do algoritmo
def principal():
    st.title("Dashboard dos Algoritmos Genéticos")
    st.markdown("**Implementação interativa do tutorial de algoritmos genéticos**")

    # Sidebar com parâmetros
    st.sidebar.header("⚙️ Parâmetros do Algoritmo")
    
    # Parâmetros do domínio
    st.sidebar.subheader("Domínio da Função")
    min_x = st.sidebar.number_input("Limite inferior (min_x)", value=-1.0, step=0.1)
    max_x = st.sidebar.number_input("Limite superior (max_x)", value=2.0, step=0.1)
    comprimento_cromossomo = st.sidebar.slider("Comprimento do cromossomo", 10, 30, 22)
    
    # Parâmetros da população
    st.sidebar.subheader("População")
    tamanho_populacao = st.sidebar.slider("Tamanho da população", 10, 100, 30)
    tipo_distribuicao = st.sidebar.selectbox("Distribuição inicial", ["aleatoria", "equidistante"])
    
    # Parâmetros evolutivos
    st.sidebar.subheader("Operadores Genéticos")
    num_geracoes = st.sidebar.slider("Número de gerações", 10, 200, 25)
    tipo_cruzamento = st.sidebar.selectbox("Tipo de cruzamento", ["Um ponto", "Dois pontos", "Uniforme"])
    taxa_cruzamento = st.sidebar.slider("Taxa de cruzamento", 0.0, 1.0, 0.8, 0.05)
    taxa_mutacao = st.sidebar.slider("Taxa de mutação", 0.001, 0.1, 0.01, 0.001)
    usar_elitismo = st.sidebar.checkbox("Usar elitismo", value=True)
    
    # Parâmetros de aptidão
    st.sidebar.subheader("Aptidão")
    aptidao_maxima_rank = st.sidebar.number_input("Aptidão máxima", value=2.0, step=0.1)
    aptidao_minima_rank = st.sidebar.number_input("Aptidão mínima", value=0.0, step=0.1)
    
    # Botão para executar
    if st.sidebar.button("Executar Algoritmo", type="primary"):
        # Preparar parâmetros
        parametros = {
            'min_x': min_x,
            'max_x': max_x,
            'comprimento_cromossomo': comprimento_cromossomo,
            'tamanho_populacao': tamanho_populacao,
            'num_geracoes': num_geracoes,
            'taxa_cruzamento': taxa_cruzamento,
            'taxa_mutacao': taxa_mutacao,
            'usar_elitismo': usar_elitismo,
            'tipo_cruzamento': tipo_cruzamento,
            'tipo_distribuicao': tipo_distribuicao,
            'aptidao_maxima_rank': aptidao_maxima_rank,
            'aptidao_minima_rank': aptidao_minima_rank
        }
        
        # Executar algoritmo
        with st.spinner("Executando algoritmo genético..."):
            resultados = executar_algoritmo_genetico_streamlit(parametros)
        
        # Armazenar resultados no session state
        st.session_state.resultados = resultados
        st.session_state.parametros = parametros
        st.success("✅ Algoritmo executado com sucesso!")
    
    # Mostrar resultados se existirem
    if 'resultados' in st.session_state:
        resultados = st.session_state.resultados
        parametros = st.session_state.parametros
        
        # Gráfico principal COM CAPTURA de valores teóricos
        st.subheader("📊 Visualização da Evolução")
        fig, x_maximo_teorico, valor_maximo_teorico = criar_grafico_evolucao(
            resultados['dados_evolucao'], 
            parametros['tipo_distribuicao'], 
            parametros['tipo_cruzamento'], 
            parametros['min_x'], 
            parametros['max_x']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Métricas principais
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Melhor f(x)", f"{resultados['melhor_objetivo']:.6f}")
        
        with col2:
            st.metric("Melhor x", f"{resultados['melhor_x']:.6f}")
        
        with col3:
            st.metric("Máximo Teórico", f"{valor_maximo_teorico:.6f}")
        
        with col4:
            st.metric("Cruzamento", parametros['tipo_cruzamento'])
        
        with col5:
            status_elitismo = "Ativado" if parametros['usar_elitismo'] else "Desativado"
            st.metric("Elitismo", status_elitismo)
        
        # Informações detalhadas
        with st.expander("📋 Informações Detalhadas"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Melhor Cromossomo")
                st.code(resultados['melhor_individuo'])
                
                st.subheader("Parâmetros Utilizados")
                df_parametros = pd.DataFrame([
                    {"Parâmetro": "População", "Valor": parametros['tamanho_populacao']},
                    {"Parâmetro": "Gerações", "Valor": parametros['num_geracoes']},
                    {"Parâmetro": "Taxa cruzamento", "Valor": f"{parametros['taxa_cruzamento']:.1%}"},
                    {"Parâmetro": "Taxa mutação", "Valor": f"{parametros['taxa_mutacao']:.1%}"},
                    {"Parâmetro": "Distribuição", "Valor": parametros['tipo_distribuicao'].title()},
                    {"Parâmetro": "Tipo cruzamento", "Valor": parametros['tipo_cruzamento']},
                ])
                
                # Função auxiliar para limpar DataFrame
                def limpar_dataframe_para_streamlit(df):
                    """Limpa o DataFrame para evitar erros do PyArrow no Streamlit"""
                    df_limpo = df.copy()
                    
                    for col in df_limpo.columns:
                        if df_limpo[col].dtype == 'object':
                            # Tentar converter porcentagens para números
                            if df_limpo[col].astype(str).str.contains('%').any():
                                try:
                                    df_limpo[col] = pd.to_numeric(
                                        df_limpo[col].astype(str).str.replace('%', ''), 
                                        errors='coerce'
                                    )
                                except:
                                    pass
                    
                    return df_limpo
                
                st.dataframe(limpar_dataframe_para_streamlit(df_parametros), hide_index=True)
            
            with col2:
                st.subheader("Estatísticas por Geração")
                df_estatisticas = pd.DataFrame({
                    'Geração': range(1, len(resultados['melhor_por_geracao']) + 1),
                    'Melhor f(x)': resultados['melhor_por_geracao'],
                    'Média f(x)': resultados['media_por_geracao']
                })
                st.dataframe(df_estatisticas, height=300)
        
        # Comparação com máximo teórico - MENSAGEM SIMPLES
        gap = valor_maximo_teorico - resultados['melhor_objetivo']
        gap_percentual = (gap / valor_maximo_teorico) * 100 if valor_maximo_teorico != 0 else 0
        
        st.info(f"""
        **📈 Comparação com Máximo Teórico:**
        - Máximo teórico: f({x_maximo_teorico:.5f}) = {valor_maximo_teorico:.5f}
        - Melhor encontrado: f({resultados['melhor_x']:.5f}) = {resultados['melhor_objetivo']:.5f}
        - Gap: {gap:.5f} ({gap_percentual:.2f}%)
        """)

if __name__ == "__main__":
    principal()