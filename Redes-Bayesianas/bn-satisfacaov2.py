# --- PROJETO 1: REDES BAYESIANAS PARA DECISÃO DE ROBÔ ASPIRADOR (v6 - Interativo) ---
# Disciplina: FGA0221 - Inteligência Artificial
# Tema: Tratando Incerteza
#
# Objetivo: Modelar a decisão de compra de um robô aspirador, permitindo
# que o usuário insira as características de um novo modelo para
# calcular sua probabilidade de satisfação.
# -----------------------------------------------------------------

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
except ImportError:
    print("Erro: Biblioteca 'pgmpy' não encontrada.")
    print("Por favor, instale usando: pip install pgmpy")
    exit()

def criar_modelo_decisao_refatorado():
    """
    Cria e retorna o modelo da Rede Bayesiana (refatorado) para a
    decisão do robô, incluindo a avaliação da loja.
    """
    print("1. Criando a estrutura da Rede Bayesiana (DAG) refatorada...")

    # 1.1 Definição da Estrutura (DAG)
    modelo = BayesianNetwork([
        # Nível 0 (Entradas/Causas Raiz)
        ('Marca', 'Disponibilidade_Pecas'),
        ('Navegacao', 'Qualidade_Limpeza'),
        ('Tipo_Mop', 'Qualidade_Limpeza'),
        ('Base_Limpeza', 'Satisfacao_Uso'),
        
        # Nova Lógica de Risco
        ('Mercado_Cinza', 'Risco_Problema'),
        ('Avaliacao_Loja', 'Risco_Problema'),
        
        # Nível 1 (Riscos e Qualidade)
        ('Qualidade_Limpeza', 'Satisfacao_Uso'),
        
        # Nível 2 (Agregação de Utilidade)
        ('Risco_Problema', 'Satisfacao_PosVenda'),
        ('Disponibilidade_Pecas', 'Satisfacao_PosVenda'),
        
        # Nível 3 (Decisão Final)
        ('Satisfacao_Uso', 'Satisfacao_Final'),
        ('Satisfacao_PosVenda', 'Satisfacao_Final')
    ])

    print("2. Definindo as Tabelas de Probabilidade Condicional (CPTs)...")

    # 2.1 CPTs dos Nós-Raiz
    
    # Adicionamos 'Kabum'
    cpd_marca = TabularCPD(
        variable='Marca', variable_card=4,
        values=[[0.25], [0.25], [0.25], [0.25]],
        state_names={'Marca': ['Xiaomi', 'Roborock', 'Dreame', 'Kabum']}
    )
    
    # 'Navegacao' agora inclui 'Giro/Outro'
    cpd_nav = TabularCPD(
        variable='Navegacao', variable_card=2,
        values=[[0.8], [0.2]], # Assumindo que a maioria é 'LDS/LiDAR'
        state_names={'Navegacao': ['LDS/LiDAR', 'Giro/Outro']}
    )
    
    cpd_mop = TabularCPD(
        variable='Tipo_Mop', variable_card=2,
        values=[[0.5], [0.5]],
        state_names={'Tipo_Mop': ['Normal', 'Giratorio']}
    )
    
    cpd_base = TabularCPD(
        variable='Base_Limpeza', variable_card=2,
        values=[[0.5], [0.5]],
        state_names={'Base_Limpeza': ['S', 'N']}
    )
    
    # NOVO NÓ-RAIZ: Mercado_Cinza (agora é uma escolha/fato de entrada)
    cpd_mc = TabularCPD(
        variable='Mercado_Cinza', variable_card=2,
        values=[[0.5], [0.5]], # Assumindo 50% de chance de ser uma compra MC
        state_names={'Mercado_Cinza': ['S', 'N']}
    )

    # NOVO NÓ-RAIZ: Avaliacao_Loja (discretizado)
    # Alta (>90%), Media (80-90%), Baixa (<80%)
    cpd_loja = TabularCPD(
        variable='Avaliacao_Loja', variable_card=3,
        values=[[0.33], [0.34], [0.33]],
        state_names={'Avaliacao_Loja': ['Alta', 'Media', 'Baixa']}
    )

    # 2.2 CPTs dos Nós Intermediários (Refatorados)
    
    # P(Disponibilidade_Pecas | Marca)
    # Kabum (Liectroux) -> Media
    cpd_dp = TabularCPD(
        variable='Disponibilidade_Pecas', variable_card=2,
        values=[[0.8, 0.5, 0.5, 0.5],  # P(DP=Alta)
                [0.2, 0.5, 0.5, 0.5]], # P(DP=Media)
        evidence=['Marca'], evidence_card=[4],
        state_names={'Disponibilidade_Pecas': ['Alta', 'Media'],
                     'Marca': ['Xiaomi', 'Roborock', 'Dreame', 'Kabum']}
    )

    # P(Qualidade_Limpeza | Navegacao, Tipo_Mop)
    # (Giro/Outro, Normal) -> Raz (Kabum)
    cpd_qualidade = TabularCPD(
        variable='Qualidade_Limpeza', variable_card=3,
        # Re-mapeando para a nova 'Navegacao'
        # (LDS/LiDAR, Normal) -> Col 1 (Raz/Boa)
        # (LDS/LiDAR, Giratorio) -> Col 2 (Exc)
        # (Giro/Outro, Normal) -> Col 3 (Raz)
        # (Giro/Outro, Giratorio) -> Col 4 (Raz/Boa - Suposição)
        values=[[0.20, 0.95, 0.05, 0.30], # P(Qualidade=Exc)
                [0.60, 0.04, 0.25, 0.50], # P(Qualidade=Boa)
                [0.20, 0.01, 0.70, 0.20]],# P(Qualidade=Raz)
        evidence=['Navegacao', 'Tipo_Mop'], evidence_card=[2, 2],
        state_names={'Qualidade_Limpeza': ['Exc', 'Boa', 'Raz'],
                     'Navegacao': ['LDS/LiDAR', 'Giro/Outro'],
                     'Tipo_Mop': ['Normal', 'Giratorio']}
    )
    
    # CPT REATORADA PRINCIPAL: P(Risco_Problema | Mercado_Cinza, Avaliacao_Loja)
    # Colunas: (MC, Loja)
    # (S, Alta), (S, Media), (S, Baixa), (N, Alta), (N, Media), (N, Baixa)
    cpd_risco = TabularCPD(
        variable='Risco_Problema', variable_card=2,
        values=[[0.40, 0.70, 0.85, 0.05, 0.15, 0.50],  # P(Risco=Alto)
                [0.60, 0.30, 0.15, 0.95, 0.85, 0.50]], # P(Risco=Baixo)
        evidence=['Mercado_Cinza', 'Avaliacao_Loja'], evidence_card=[2, 3],
        state_names={'Risco_Problema': ['Alto', 'Baixo'],
                     'Mercado_Cinza': ['S', 'N'],
                     'Avaliacao_Loja': ['Alta', 'Media', 'Baixa']}
    )

    # 2.3 CPTs dos Nós de Utilidade (Lógica inalterada)
    cpd_sat_uso = TabularCPD(
        variable='Satisfacao_Uso', variable_card=2,
        values=[[0.99, 0.80, 0.90, 0.60, 0.70, 0.30], # P(SatUso=Alta)
                [0.01, 0.20, 0.10, 0.40, 0.30, 0.70]],# P(SatUso=Baixa)
        evidence=['Qualidade_Limpeza', 'Base_Limpeza'], evidence_card=[3, 2],
        state_names={'Satisfacao_Uso': ['Alta', 'Baixa'],
                     'Qualidade_Limpeza': ['Exc', 'Boa', 'Raz'],
                     'Base_Limpeza': ['S', 'N']}
    )
    
    cpd_sat_posvenda = TabularCPD(
        variable='Satisfacao_PosVenda', variable_card=2,
        values=[[0.30, 0.10, 0.90, 0.70], # P(SatPosVenda=Alta)
                [0.70, 0.90, 0.10, 0.30]],# P(SatPosVenda=Baixa)
        evidence=['Risco_Problema', 'Disponibilidade_Pecas'], evidence_card=[2, 2],
        state_names={'Satisfacao_PosVenda': ['Alta', 'Baixa'],
                     'Risco_Problema': ['Alto', 'Baixo'],
                     'Disponibilidade_Pecas': ['Alta', 'Media']}
    )

    # 2.4 CPT do Nó Final (Lógica inalterada)
    cpd_sat_final = TabularCPD(
        variable='Satisfacao_Final', variable_card=2,
        values=[[0.98, 0.70, 0.60, 0.10], # P(SatFinal=Alta)
                [0.02, 0.30, 0.40, 0.90]],# P(SatFinal=Baixa)
        evidence=['Satisfacao_Uso', 'Satisfacao_PosVenda'], evidence_card=[2, 2],
        state_names={'Satisfacao_Final': ['Alta', 'Baixa'],
                     'Satisfacao_Uso': ['Alta', 'Baixa'],
                     'Satisfacao_PosVenda': ['Alta', 'Baixa']}
    )
    
    print("3. Adicionando CPTs ao modelo...")
    modelo.add_cpds(
        cpd_marca, cpd_nav, cpd_mop, cpd_base, cpd_mc, cpd_loja, # Nós raiz
        cpd_dp, cpd_qualidade, cpd_risco, # Nós intermediários
        cpd_sat_uso, cpd_sat_posvenda, # Nós de utilidade
        cpd_sat_final # Nó final
    )
    
    # 4. Verificando a estrutura e CPTs
    print("4. Verificando a consistência do modelo...")
    if modelo.check_model():
        print("Modelo refatorado e validado com sucesso!")
    else:
        print("Erro na criação do modelo refatorado.")
        return None
        
    return modelo

def realizar_inferencias_fixas(inferencia):
    """
    Executa as consultas pré-definidas (baseline) no modelo.
    """
    print("\n" + "="*50)
    print("5. Realizando Inferências Fixas (Baseline)")
    print("="*50)
    
    resultados = {}

    # 5.1 Consulta 1: Roborock Q8 Max
    print("\n--- Consulta 1: Roborock Q8 Max ---")
    evidencia = {
        'Marca': 'Roborock', 'Navegacao': 'LDS/LiDAR', 'Tipo_Mop': 'Normal',
        'Base_Limpeza': 'N', 'Mercado_Cinza': 'N', 'Avaliacao_Loja': 'Media'
    }
    print(f"Evidências: {evidencia}")
    resultado_q8 = inferencia.query(['Satisfacao_Final'], evidence=evidencia)
    print(resultado_q8)
    resultados["Roborock Q8 Max"] = resultado_q8.values[0]

    # 5.2 Consulta 2: XIAOMI S20+
    print("\n--- Consulta 2: XIAOMI S20+ ---")
    evidencia = {
        'Marca': 'Xiaomi', 'Navegacao': 'LDS/LiDAR', 'Tipo_Mop': 'Giratorio',
        'Base_Limpeza': 'N', 'Mercado_Cinza': 'S', 'Avaliacao_Loja': 'Media'
    }
    print(f"Evidências: {evidencia}")
    resultado_s20 = inferencia.query(['Satisfacao_Final'], evidence=evidencia)
    print(resultado_s20)
    resultados["XIAOMI S20+"] = resultado_s20.values[0]

    # 5.3 Consulta 3: DREAME D10 Plus
    print("\n--- Consulta 3: DREAME D10 Plus ---")
    evidencia = {
        'Marca': 'Dreame', 'Navegacao': 'LDS/LiDAR', 'Tipo_Mop': 'Normal',
        'Base_Limpeza': 'S', 'Mercado_Cinza': 'N', 'Avaliacao_Loja': 'Media'
    }
    print(f"Evidências: {evidencia}")
    resultado_d10 = inferencia.query(['Satisfacao_Final'], evidence=evidencia)
    print(resultado_d10)
    resultados["DREAME D10 Plus"] = resultado_d10.values[0]
    
    # 5.4 Consulta 4: KABUM SMART 700
    print("\n--- Consulta 4: KABUM SMART 700 ---")
    evidencia = {
        'Marca': 'Kabum', 'Navegacao': 'Giro/Outro', 'Tipo_Mop': 'Normal',
        'Base_Limpeza': 'N', 'Mercado_Cinza': 'N', 'Avaliacao_Loja': 'Media'
    }
    print(f"Evidências: {evidencia}")
    resultado_k700 = inferencia.query(['Satisfacao_Final'], evidence=evidencia)
    print(resultado_k700)
    resultados["KABUM SMART 700"] = resultado_k700.values[0]
    
    # 5.5 Consulta 5: XIAOMI S40c
    print("\n--- Consulta 5: XIAOMI S40c (Venda Oficial, Loja Alta) ---")
    evidencia = {
        'Marca': 'Xiaomi', 'Navegacao': 'LDS/LiDAR', 'Tipo_Mop': 'Normal',
        'Base_Limpeza': 'N', 'Mercado_Cinza': 'N', 'Avaliacao_Loja': 'Alta'
    }
    print(f"Evidências: {evidencia}")
    resultado_s40c = inferencia.query(['Satisfacao_Final'], evidence=evidencia)
    print(resultado_s40c)
    resultados["XIAOMI S40c"] = resultado_s40c.values[0]
    
    # 5.6 Consulta 6: ROBOROCK Q7 L5
    print("\n--- Consulta 6: ROBOROCK Q7 L5 ---")
    evidencia = {
        'Marca': 'Roborock', 'Navegacao': 'LDS/LiDAR', 'Tipo_Mop': 'Normal',
        'Base_Limpeza': 'N', 'Mercado_Cinza': 'N', 'Avaliacao_Loja': 'Media'
    }
    print(f"Evidências: {evidencia}")
    resultado_q7 = inferencia.query(['Satisfacao_Final'], evidence=evidencia)
    print(resultado_q7)
    resultados["ROBOROCK Q7 L5"] = resultado_q7.values[0]
    
    print("\n--- Análise das Consultas Fixas ---")
    
    # Ordena os resultados do melhor para o pior
    resultados_ordenados = sorted(resultados.items(), key=lambda item: item[1], reverse=True)
    
    for nome, prob in resultados_ordenados:
        print(f"Prob. Satisfação Alta ({nome}): {prob*100:.2f}%")
    
    print(f"\nDecisão (Baseline): O modelo {resultados_ordenados[0][0]} oferece a maior probabilidade de satisfação.")

def prompt_usuario(pergunta, opcoes):
    """
    Função auxiliar para solicitar uma entrada válida do usuário.
    """
    print(f"\n{pergunta}")
    for i, opcao in enumerate(opcoes):
        print(f"  {i+1}. {opcao}")
    
    while True:
        try:
            escolha = int(input(f"Digite o número da sua escolha (1-{len(opcoes)}): "))
            if 1 <= escolha <= len(opcoes):
                return opcoes[escolha - 1]
            else:
                print(f"Escolha inválida. Por favor, digite um número entre 1 e {len(opcoes)}.")
        except ValueError:
            print("Entrada inválida. Por favor, digite um número.")

def realizar_inferencia_interativa(inferencia, modelo):
    """
    Solicita ao usuário as características de um novo robô e
    calcula a probabilidade de satisfação.
    """
    print("\n" + "="*50)
    print("6. Consulta Interativa: Monte seu Próprio Robô")
    print("="*50)
    print("Por favor, insira as características do robô que você deseja avaliar.")

    # Coleta as evidências dos nós-raiz do modelo
    evidencia = {}
    evidencia['Marca'] = prompt_usuario("Marca do Robô?", modelo.get_cpds('Marca').state_names['Marca'])
    evidencia['Navegacao'] = prompt_usuario("Tipo de Navegação?", modelo.get_cpds('Navegacao').state_names['Navegacao'])
    evidencia['Tipo_Mop'] = prompt_usuario("Tipo de Mop?", modelo.get_cpds('Tipo_Mop').state_names['Tipo_Mop'])
    evidencia['Base_Limpeza'] = prompt_usuario("Tem Base Autolimpante?", modelo.get_cpds('Base_Limpeza').state_names['Base_Limpeza'])
    evidencia['Mercado_Cinza'] = prompt_usuario("É do Mercado Cinza?", modelo.get_cpds('Mercado_Cinza').state_names['Mercado_Cinza'])
    evidencia['Avaliacao_Loja'] = prompt_usuario("Avaliação da Loja?", modelo.get_cpds('Avaliacao_Loja').state_names['Avaliacao_Loja'])
    
    print("\n--- Calculando a probabilidade para o robô com as seguintes características: ---")
    print(evidencia)

    # Realiza a inferência
    resultado_custom = inferencia.query(
        variables=['Satisfacao_Final'],
        evidence=evidencia
    )
    
    print("\n--- Resultado da Inferência ---")
    print(resultado_custom)
    
    prob_alta = resultado_custom.values[0] # Índice 0 é 'Alta'
    print(f"\nEste robô customizado tem {prob_alta*100:.2f}% de chance de gerar 'Satisfação Alta'.")


# --- Função Principal ---
if __name__ == "__main__":
    modelo_robos = criar_modelo_decisao_refatorado()
    
    if modelo_robos:
        # Criar o objeto de inferência UMA VEZ
        print("\nCriando objeto de inferência (VariableElimination)...")
        inferencia_global = VariableElimination(modelo_robos)
        
        # 1. Executar as consultas de baseline
        realizar_inferencias_fixas(inferencia_global)
        
        # 2. Executar a consulta interativa
        realizar_inferencia_interativa(inferencia_global, modelo_robos)