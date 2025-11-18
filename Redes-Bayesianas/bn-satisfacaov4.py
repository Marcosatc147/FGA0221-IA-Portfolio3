# --- PROJETO 1: REDES BAYESIANAS PARA DECISÃO DE ROBÔ ASPIRADOR (v4 - Risco Financeiro) ---
# Disciplina: FGA0221 - Inteligência Artificial
# Tema: Tratando Incerteza
#
# Objetivo: Modelo calibrado que inclui o 'Valor_Investimento' como
# um fator que influencia o 'Risco_Problema'.
# -----------------------------------------------------------------

import os
import traceback

print("DEBUG: Iniciando importações...")
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    from pgmpy.readwrite import XMLBIFWriter
    print("DEBUG: Importações do 'pgmpy' (core e XMLBIFWriter) OK.")
except ImportError as e:
    print("="*50); print("!!! ERRO CRÍTICO: Falha ao importar o 'pgmpy'. !!!"); print(f"Detalhe do Erro: {e}"); print("="*50)
    traceback.print_exc()
    exit()
except Exception as e:
    print("="*50); print("!!! ERRO INESPERADO AO IMPORTAR MÓDULOS !!!"); print(f"Detalhe do Erro: {e}"); print("="*50)
    traceback.print_exc()
    exit()

def criar_modelo_decisao_refatorado():
    """
    Cria e retorna o modelo da Rede Bayesiana (refatorado) para a
    decisão do robô, incluindo a avaliação da loja e o valor do investimento.
    """
    print("\n1. Criando a estrutura da Rede Bayesiana (DAG) refatorada...")

    # 1.1 Definição da Estrutura (DAG)
    # Valor_Investimento é um novo pai de Risco_Problema
    modelo = BayesianNetwork([
        # Nível 0 (Entradas/Causas Raiz)
        ('Marca', 'Disponibilidade_Pecas'),
        ('Navegacao', 'Qualidade_Limpeza'),
        ('Tipo_Mop', 'Qualidade_Limpeza'),
        ('Base_Limpeza', 'Satisfacao_Uso'),
        
        # Nova Lógica de Risco (3 Pais)
        ('Mercado_Cinza', 'Risco_Problema'),
        ('Avaliacao_Loja', 'Risco_Problema'),
        ('Valor_Investimento', 'Risco_Problema'), # <- NOVA LIGAÇÃO
        
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
        values=[[0.8], [0.2]], 
        state_names={'Navegacao': ['LDS_LiDAR', 'Giro_Outro']}
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
    cpd_mc = TabularCPD(
        variable='Mercado_Cinza', variable_card=2,
        values=[[0.5], [0.5]], 
        state_names={'Mercado_Cinza': ['S', 'N']}
    )
    cpd_loja = TabularCPD(
        variable='Avaliacao_Loja', variable_card=3,
        values=[[0.33], [0.34], [0.33]],
        state_names={'Avaliacao_Loja': ['Alta', 'Media', 'Baixa']}
    )

    # NOVO NÓ-RAIZ: Valor_Investimento
    # Baixo (<1500), Medio (1500-2300), Alto (>2300)
    cpd_valor = TabularCPD(
        variable='Valor_Investimento', variable_card=3,
        values=[[0.33], [0.34], [0.33]],
        state_names={'Valor_Investimento': ['Alto', 'Medio', 'Baixo']}
    )

    # 2.2 CPTs dos Nós Intermediários (Lógica de Qualidade e Risco)
    cpd_dp = TabularCPD(
        variable='Disponibilidade_Pecas', variable_card=2,
        values=[[0.8, 0.5, 0.5, 0.5],  # P(DP=Alta)
                [0.2, 0.5, 0.5, 0.5]], # P(DP=Media)
        evidence=['Marca'], evidence_card=[4],
        state_names={'Disponibilidade_Pecas': ['Alta', 'Media'],
                     'Marca': ['Xiaomi', 'Roborock', 'Dreame', 'Kabum']}
    )
    cpd_qualidade = TabularCPD(
        variable='Qualidade_Limpeza', variable_card=3,
        values=[[0.20, 0.95, 0.05, 0.30], # P(Qualidade=Exc)
                [0.60, 0.04, 0.25, 0.50], # P(Qualidade=Boa)
                [0.20, 0.01, 0.70, 0.20]],# P(Qualidade=Raz)
        evidence=['Navegacao', 'Tipo_Mop'], evidence_card=[2, 2],
        state_names={'Qualidade_Limpeza': ['Exc', 'Boa', 'Raz'],
                     'Navegacao': ['LDS_LiDAR', 'Giro_Outro'], 
                     'Tipo_Mop': ['Normal', 'Giratorio']}
    )
    
    cpd_risco = TabularCPD(
    variable='Risco_Problema', variable_card=2,

    values=[
        [0.85, 0.90, 0.96,   0.20, 0.35, 0.70,   # Valor Alto
         0.80, 0.86, 0.94,   0.15, 0.30, 0.60,   # Valor Medio
         0.75, 0.82, 0.92,   0.10, 0.25, 0.55],  # Valor Baixo

        [0.15, 0.10, 0.04,   0.80, 0.65, 0.30,
         0.20, 0.14, 0.06,   0.85, 0.70, 0.40,
         0.25, 0.18, 0.08,   0.90, 0.75, 0.45]
    ],

    evidence=['Valor_Investimento', 'Mercado_Cinza', 'Avaliacao_Loja'],
    evidence_card=[3, 2, 3],

    state_names={
        'Risco_Problema': ['Alto', 'Baixo'],
        'Valor_Investimento': ['Alto', 'Medio', 'Baixo'],
        'Mercado_Cinza': ['S', 'N'],
        'Avaliacao_Loja': ['Alta', 'Media', 'Baixa']
    }
)


    # 2.3 CPTs dos Nós de Utilidade (Calibradas)
    cpd_sat_uso = TabularCPD(
        variable='Satisfacao_Uso', variable_card=2,
        values=[[0.99, 0.80, 0.90, 0.60, 0.20, 0.10], # P(SatUso=Alta)
                [0.01, 0.20, 0.10, 0.40, 0.80, 0.90]],# P(SatUso=Baixa)
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

    # 2.4 CPT do Nó Final (Calibrada)
    cpd_sat_final = TabularCPD(
        variable='Satisfacao_Final', variable_card=2,
        values=[[0.98, 0.30, 0.60, 0.01], # P(SatFinal=Alta)
                [0.02, 0.70, 0.40, 0.99]],# P(SatFinal=Baixa)
        evidence=['Satisfacao_Uso', 'Satisfacao_PosVenda'], evidence_card=[2, 2],
        state_names={'Satisfacao_Final': ['Alta', 'Baixa'],
                     'Satisfacao_Uso': ['Alta', 'Baixa'],
                     'Satisfacao_PosVenda': ['Alta', 'Baixa']}
    )
    
    print("3. Adicionando CPTs ao modelo...")
    modelo.add_cpds(
        cpd_marca, cpd_nav, cpd_mop, cpd_base, cpd_mc, cpd_loja, cpd_valor, # Nós raiz
        cpd_dp, cpd_qualidade, cpd_risco, # Nós intermediários
        cpd_sat_uso, cpd_sat_posvenda, # Nós de utilidade
        cpd_sat_final # Nó final
    )
    
    # 4. Verificando a estrutura e CPTs
    print("4. Verificando a consistência do modelo...")
    if modelo.check_model():
        print("Modelo refatorado e calibrado validado com sucesso!")
    else:
        print("!!! ERRO: O modelo falhou na verificação. Verifique as CPTs. !!!")
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

    # --- Mapeamento de Valor dos Robôs ---
    # Alto: Q8 Max, S20+, D10 Plus
    # Medio: S40c, Q7 L5, D9 Max
    # Baixo: Kabum 700

    # 5.1 Consulta 1: Roborock Q8 Max (Valor=Alto)
    print("\n--- Consulta 1: Roborock Q8 Max ---")
    evidencia = {
        'Marca': 'Roborock', 'Navegacao': 'LDS_LiDAR', 'Tipo_Mop': 'Normal',
        'Base_Limpeza': 'N', 'Mercado_Cinza': 'N', 'Avaliacao_Loja': 'Media',
        'Valor_Investimento': 'Alto' # <- NOVO
    }
    print(f"Evidências: {evidencia}")
    resultado_q8 = inferencia.query(['Satisfacao_Final'], evidence=evidencia)
    print(resultado_q8)
    resultados["Roborock Q8 Max"] = resultado_q8.values[0]

    # 5.2 Consulta 2: XIAOMI S20+ (Valor=Alto)
    print("\n--- Consulta 2: XIAOMI S20+ ---")
    evidencia = {
        'Marca': 'Xiaomi', 'Navegacao': 'LDS_LiDAR', 'Tipo_Mop': 'Giratorio',
        'Base_Limpeza': 'N', 'Mercado_Cinza': 'S', 'Avaliacao_Loja': 'Media',
        'Valor_Investimento': 'Alto' # <- NOVO
    }
    print(f"Evidências: {evidencia}")
    resultado_s20 = inferencia.query(['Satisfacao_Final'], evidence=evidencia)
    print(resultado_s20)
    resultados["XIAOMI S20+"] = resultado_s20.values[0]

    # 5.3 Consulta 3: DREAME D10 Plus (Valor=Alto)
    print("\n--- Consulta 3: DREAME D10 Plus ---")
    evidencia = {
        'Marca': 'Dreame', 'Navegacao': 'LDS_LiDAR', 'Tipo_Mop': 'Normal',
        'Base_Limpeza': 'S', 'Mercado_Cinza': 'N', 'Avaliacao_Loja': 'Media',
        'Valor_Investimento': 'Alto' # <- NOVO
    }
    print(f"Evidências: {evidencia}")
    resultado_d10 = inferencia.query(['Satisfacao_Final'], evidence=evidencia)
    print(resultado_d10)
    resultados["DREAME D10 Plus"] = resultado_d10.values[0]
    
    # 5.4 Consulta 4: KABUM SMART 700 (Valor=Baixo)
    print("\n--- Consulta 4: KABUM SMART 700 ---")
    evidencia = {
        'Marca': 'Kabum', 'Navegacao': 'Giro_Outro', 'Tipo_Mop': 'Normal',
        'Base_Limpeza': 'N', 'Mercado_Cinza': 'N', 'Avaliacao_Loja': 'Media',
        'Valor_Investimento': 'Baixo' # <- NOVO
    }
    print(f"Evidências: {evidencia}")
    resultado_k700 = inferencia.query(['Satisfacao_Final'], evidence=evidencia)
    print(resultado_k700)
    resultados["KABUM SMART 700"] = resultado_k700.values[0]
    
    # 5.5 Consulta 5: XIAOMI S40c (Valor=Medio)
    print("\n--- Consulta 5: XIAOMI S40c (Venda Oficial, Loja Alta) ---")
    evidencia = {
        'Marca': 'Xiaomi', 'Navegacao': 'LDS_LiDAR', 'Tipo_Mop': 'Normal',
        'Base_Limpeza': 'N', 'Mercado_Cinza': 'N', 'Avaliacao_Loja': 'Alta',
        'Valor_Investimento': 'Medio' # <- NOVO
    }
    print(f"Evidências: {evidencia}")
    resultado_s40c = inferencia.query(['Satisfacao_Final'], evidence=evidencia)
    print(resultado_s40c)
    resultados["XIAOMI S40c"] = resultado_s40c.values[0]
    
    # 5.6 Consulta 6: ROBOROCK Q7 L5 (Valor=Medio)
    print("\n--- Consulta 6: ROBOROCK Q7 L5 ---")
    evidencia = {
        'Marca': 'Roborock', 'Navegacao': 'LDS_LiDAR', 'Tipo_Mop': 'Normal',
        'Base_Limpeza': 'N', 'Mercado_Cinza': 'N', 'Avaliacao_Loja': 'Media',
        'Valor_Investimento': 'Medio' # <- NOVO
    }
    print(f"Evidências: {evidencia}")
    resultado_q7 = inferencia.query(['Satisfacao_Final'], evidence=evidencia)
    print(resultado_q7)
    resultados["ROBOROCK Q7 L5"] = resultado_q7.values[0]
    
    
    print("\n--- Análise das Consultas Fixas (Modelo Calibrado) ---")
    
    # Ordena os resultados do melhor para o pior
    resultados_ordenados = sorted(resultados.items(), key=lambda item: item[1], reverse=True)
    
    for nome, prob in resultados_ordenados:
        print(f"Prob. Satisfação Alta ({nome}): {prob*100:.2f}%")
    
    print(f"\nDecisão (Baseline): O modelo {resultados_ordenados[0][0]} oferece a maior probabilidade de satisfação.")

# --- FUNÇÃO DE EXPORTAÇÃO ---
def exportar_modelo(modelo, nome_arquivo="modelo_robos_final.bif"):
    """
    Exporta o modelo Bayesiano completo (DAG + CPTs) para um
    arquivo .bif (XML), que pode ser aberto por softwares visuais
    como o GeNIe.
    """
    print(f"\n" + "="*50)
    print("7. Exportando modelo para visualização (formato XMLBIF)...")
    print("="*50)
    
    if XMLBIFWriter is None:
        print("!!! EXPORTAÇÃO INTERROMPIDA !!!")
        print("O 'XMLBIFWriter' não pôde ser importado.")
        return

    try:
        try:
            script_dir = os.path.dirname(__file__)
        except NameError:
            script_dir = os.getcwd() 
            
        caminho_completo = os.path.join(script_dir, nome_arquivo)

        writer = XMLBIFWriter(model=modelo)
        writer.write_xmlbif(filename=caminho_completo)
        
        print(f"Sucesso! Modelo salvo como:")
        print(f"{caminho_completo}")
        print(f"\nAbra este arquivo no GeNIe Modeler (bayesfusion.com/genie) para explorá-lo interativamente.")
    
    except Exception as e:
        print(f"!!! ERRO AO EXPORTAR O MODELO !!!")
        print("A importação funcionou, mas a escrita falhou.")
        traceback.print_exc()


# --- Função Principal ---
if __name__ == "__main__":
    modelo_robos = criar_modelo_decisao_refatorado()
    
    if modelo_robos:
        print("\nCriando objeto de inferência (VariableElimination)...")
        inferencia_global = VariableElimination(modelo_robos)
        
        realizar_inferencias_fixas(inferencia_global)
        
        exportar_modelo(modelo_robos)
    else:
        print("\nO script não pôde continuar porque o modelo Bayesiano falhou na verificação.")