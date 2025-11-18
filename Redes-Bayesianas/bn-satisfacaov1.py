# --- PROJETO 1: REDES BAYESIANAS PARA DECISÃO DE ROBÔ ASPIRADOR (v1 - ) ---
# Disciplina: FGA0221 - Inteligência Artificial
# Tema: Tratando Incerteza
#
# Objetivo: Modelar a decisão de compra de um robô aspirador,
# incluindo a avaliação da loja como um fator que mitiga o risco.
# -----------------------------------------------------------------

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
except ImportError:
    print("Erro: Biblioteca 'pgmpy' não encontrada.")
    print("Por favor, instale usando: pip install pgmpy")
    exit()

def criar_modelo_decisao():
    """
    Cria e retorna o modelo da Rede Bayesiana para a
    decisão do robô, incluindo a avaliação da loja.
    """
    print("1. Criando a estrutura da Rede Bayesiana (DAG) ...")

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
    
    cpd_marca = TabularCPD(
        variable='Marca', variable_card=4,
        values=[[0.25], [0.25], [0.25], [0.25]],
        state_names={'Marca': ['Xiaomi', 'Roborock', 'Dreame', 'Kabum']}
    )
    
    cpd_nav = TabularCPD(
        variable='Navegacao', variable_card=2,
        values=[[0.8], [0.2]], # Assumindo que a maioria é 'LDS/LiDAR'
        state_names={'Navegacao': ['LDS/LiDAR', 'Giro/Outro']}
    )
    
    cpd_mop = TabularCPD(
        variable='Tipo_Mop', variable_card=2,
        values=[[0.7], [0.3]],
        state_names={'Tipo_Mop': ['Normal', 'Giratorio']}
    )
    
    cpd_base = TabularCPD(
        variable='Base_Limpeza', variable_card=2,
        values=[[0.5], [0.5]],
        state_names={'Base_Limpeza': ['S', 'N']}
    )
    
    # NÓ-RAIZ: Mercado_Cinza (agora é uma escolha/fato de entrada)
    cpd_mc = TabularCPD(
        variable='Mercado_Cinza', variable_card=2,
        values=[[0.5], [0.5]], # Assumindo 50% de chance de ser uma compra MC
        state_names={'Mercado_Cinza': ['S', 'N']}
    )

    # NÓ-RAIZ: Avaliacao_Loja (discretizado)
    # Alta (>90%), Media (80-90%), Baixa (<80%)
    cpd_loja = TabularCPD(
        variable='Avaliacao_Loja', variable_card=3,
        values=[[0.33], [0.34], [0.33]],
        state_names={'Avaliacao_Loja': ['Alta', 'Media', 'Baixa']}
    )

    # 2.2 CPTs dos Nós Intermediários
    
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
    
    # CPT PRINCIPAL: P(Risco_Problema | Mercado_Cinza, Avaliacao_Loja)
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
    # Lógica da CPT de Risco:
    # (S, Baixa) -> Risco Alto (0.85) - Pior caso (S20+ em loja ruim)
    # (S, Media) -> Risco Alto (0.70) - (S20+ em loja 85%)
    # (S, Alta)  -> Risco Alto (0.40) - Risco mitigado pela loja boa
    # (N, Baixa) -> Risco Alto (0.50) - Venda oficial, mas loja ruim (Kabum/D9)
    # (N, Media) -> Risco Baixo (0.85) - Venda oficial, loja ok (Q7, Q8, D10)
    # (N, Alta)  -> Risco Baixo (0.95) - Melhor caso (S40c em loja 95%)

    # 2.3 CPTs dos Nós de Utilidade
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
        print("Modelo validado com sucesso!")
    else:
        print("Erro na criação do modelo.")
        return None
        
    return modelo

def realizar_inferencia(modelo):
    """
    Executa consultas (inferências) no modelo Bayesiano .
    """
    print("\n" + "="*50)
    print("5. Realizando Inferência (Consultas de Decisão)")
    print("="*50)

    inferencia = VariableElimination(modelo)

    # 5.1 Consulta 1: Roborock Q8 Max (Oficial, Loja Média)
    print("\n--- Consulta 1: Roborock Q8 Max ---")
    print("Evidências: Marca=Roborock, Nav=LDS/LiDAR, Mop=Normal, Base=N, MC=N, Loja=Media")
    resultado_q8 = inferencia.query(
        variables=['Satisfacao_Final'],
        evidence={
            'Marca': 'Roborock', 'Navegacao': 'LDS/LiDAR', 'Tipo_Mop': 'Normal',
            'Base_Limpeza': 'N', 'Mercado_Cinza': 'N', 'Avaliacao_Loja': 'Media'
        }
    )
    print(resultado_q8)

    # 5.2 Consulta 2: XIAOMI S20+ (Mercado Cinza, Loja Média)
    print("\n--- Consulta 2: XIAOMI S20+ ---")
    print("Evidências: Marca=Xiaomi, Nav=LDS/LiDAR, Mop=Giratorio, Base=N, MC=S, Loja=Media")
    resultado_s20 = inferencia.query(
        variables=['Satisfacao_Final'],
        evidence={
            'Marca': 'Xiaomi', 'Navegacao': 'LDS/LiDAR', 'Tipo_Mop': 'Giratorio',
            'Base_Limpeza': 'N', 'Mercado_Cinza': 'S', 'Avaliacao_Loja': 'Media'
        }
    )
    print(resultado_s20)

    # 5.3 Consulta 3: DREAME D10 Plus (Oficial, Loja Média)
    print("\n--- Consulta 3: DREAME D10 Plus ---")
    print("Evidências: Marca=Dreame, Nav=LDS/LiDAR, Mop=Normal, Base=S, MC=N, Loja=Media")
    resultado_d10 = inferencia.query(
        variables=['Satisfacao_Final'],
        evidence={
            'Marca': 'Dreame', 'Navegacao': 'LDS/LiDAR', 'Tipo_Mop': 'Normal',
            'Base_Limpeza': 'S', 'Mercado_Cinza': 'N', 'Avaliacao_Loja': 'Media'
        }
    )
    print(resultado_d10)
    
    # 5.4 Consulta 4: KABUM SMART 700 (Oficial, Loja Média/Baixa)
    print("\n--- Consulta 4: KABUM SMART 700 ---")
    print("Evidências: Marca=Kabum, Nav=Giro/Outro, Mop=Normal, Base=N, MC=N, Loja=Media")
    resultado_k700 = inferencia.query(
        variables=['Satisfacao_Final'],
        evidence={
            'Marca': 'Kabum', 'Navegacao': 'Giro/Outro', 'Tipo_Mop': 'Normal',
            'Base_Limpeza': 'N', 'Mercado_Cinza': 'N', 'Avaliacao_Loja': 'Media'
        }
    )
    print(resultado_k700)
    
    # 5.5 Consulta 5: XIAOMI S40c (Oficial, Loja Alta)
    print("\n--- Consulta 5: XIAOMI S40c (Venda Oficial, Loja Alta) ---")
    print("Evidências: Marca=Xiaomi, Nav=LDS/LiDAR, Mop=Normal, Base=N, MC=N, Loja=Alta")
    resultado_s40c = inferencia.query(
        variables=['Satisfacao_Final'],
        evidence={
            'Marca': 'Xiaomi', 'Navegacao': 'LDS/LiDAR', 'Tipo_Mop': 'Normal',
            'Base_Limpeza': 'N', 'Mercado_Cinza': 'N', 'Avaliacao_Loja': 'Alta'
        }
    )
    print(resultado_s40c)
    
    # 5.6 Consulta 6: ROBOROCK Q7 L5 (Oficial, Loja Média)
    print("\n--- Consulta 6: ROBOROCK Q7 L5 ---")
    print("Evidências: Marca=Roborock, Nav=LDS/LiDAR, Mop=Normal, Base=N, MC=N, Loja=Media")
    resultado_q7 = inferencia.query(
        variables=['Satisfacao_Final'],
        evidence={
            'Marca': 'Roborock', 'Navegacao': 'LDS/LiDAR', 'Tipo_Mop': 'Normal',
            'Base_Limpeza': 'N', 'Mercado_Cinza': 'N', 'Avaliacao_Loja': 'Media'
        }
    )
    print(resultado_q7)
    # *** FIM DA NOVA CONSULTA ***
    
    print("\n--- Análise das Consultas ---")
    prob_q8 = resultado_q8.values[0]
    prob_s20 = resultado_s20.values[0]
    prob_d10 = resultado_d10.values[0]
    prob_k700 = resultado_k700.values[0]
    prob_s40c = resultado_s40c.values[0]
    prob_q7 = resultado_q7.values[0]

    print(f"Prob. Satisfação Alta (XIAOMI S40c):     {prob_s40c*100:.2f}%")
    print(f"Prob. Satisfação Alta (DREAME D10 Plus):  {prob_d10*100:.2f}%")
    print(f"Prob. Satisfação Alta (Roborock Q8 Max): {prob_q8*100:.2f}%")
    print(f"Prob. Satisfação Alta (ROBOROCK Q7 L5):  {prob_q7*100:.2f}%")
    print(f"Prob. Satisfação Alta (XIAOMI S20+):     {prob_s20*100:.2f}%")
    print(f"Prob. Satisfação Alta (KABUM SMART 700): {prob_k700*100:.2f}%")
    
    
    melhor_modelo = max(
        ("Roborock Q8 Max", prob_q8), 
        ("XIAOMI S20+", prob_s20), 
        ("DREAME D10 Plus", prob_d10),
        ("KABUM SMART 700", prob_k700),
        ("XIAOMI S40c", prob_s40c),
        ("ROBOROCK Q7 L5", prob_q7),
        key=lambda item: item[1]
    )
    print(f"\nDecisão: O modelo {melhor_modelo[0]} oferece a maior probabilidade de satisfação.")


# --- Função Principal ---
if __name__ == "__main__":
    modelo_robos = criar_modelo_decisao()
    
    if modelo_robos:
        realizar_inferencia(modelo_robos)