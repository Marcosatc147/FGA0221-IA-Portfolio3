# --- PROJETO 2: MODELOS MARKOVIANOS OCULTOS (HMM) ---
# Disciplina: FGA0221 - Inteligência Artificial
# Tema: Tratando Incerteza
#
# Objetivo: Diagnosticar o estado operacional real de um robô aspirador
# (Limpando, Preso, Base) a partir de uma sequência de logs de sensores
# ruidosos (Normal, Colisao, EmEspera), usando o Algoritmo de Viterbi.
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

class RoboHMM:
    def __init__(self):
        # 1. Definição dos Estados Ocultos (Q) [cite: 1553]
        self.estados = ['Limpando', 'Preso', 'Base']
        self.n_estados = len(self.estados)
        
        # 2. Definição das Observações Possíveis (O) [cite: 1556]
        # Ajuste solicitado: "Imovel" -> "EmEspera"
        self.observacoes = ['Normal', 'Colisao', 'EmEspera']
        self.mapa_obs = {obs: i for i, obs in enumerate(self.observacoes)}
        
        # 3. Probabilidades Iniciais (Pi) [cite: 1558]
        # Suposição: O robô quase sempre começa na Base ou Limpando.
        self.pi = np.array([0.4, 0.0, 0.6])  # [Limpando, Preso, Base]
        
        # 4. Matriz de Transição (A) - P(Estado_t | Estado_t-1) [cite: 1555]
        # Linhas: Estado Anterior -> Colunas: Estado Atual
        self.A = np.array([
            # Limp, Preso, Base
            [0.80, 0.10, 0.10], # Se estava Limpando
            [0.40, 0.60, 0.00], # Se estava Preso (tenta sair ou continua preso)
            [0.10, 0.00, 0.90]  # Se estava na Base (sai ou fica)
        ])
        
        # 5. Matriz de Emissão (B) - P(Observação | Estado) [cite: 1557]
        # Linhas: Estado Oculto -> Colunas: Observação [Normal, Colisao, EmEspera]
        self.B = np.array([
            # Norm, Col,  Esp
            [0.80, 0.15, 0.05], # Estado: Limpando (Geralmente normal, as vezes bate)
            [0.05, 0.55, 0.40], # Estado: Preso (Bate muito ou fica parado tentando)
            [0.10, 0.00, 0.90]  # Estado: Base (Quase sempre em espera)
        ])

    def viterbi(self, sequencia_obs):
        """
        Implementação do Algoritmo de Viterbi para decodificação.
        Encontra a sequência de estados ocultos mais provável dada a observação.
        Referência: Russell & Norvig / Jurafsky [cite: 1893-1900]
        """
        T = len(sequencia_obs)
        
        # Matriz de Probabilidades do Caminho (Viterbi Trellis) [cite: 1819]
        # Armazena a prob do caminho mais provável até o estado j no tempo t
        delta = np.zeros((T, self.n_estados))
        
        # Matriz de Backpointers (para reconstruir o caminho) [cite: 1864]
        psi = np.zeros((T, self.n_estados), dtype=int)
        
        # Converter observações (strings) para índices
        obs_indices = [self.mapa_obs[o] for o in sequencia_obs]
        
        # --- PASSO 1: Inicialização (t=0) [cite: 1894] ---
        # delta[0, s] = pi[s] * B[s, obs_0]
        primeira_obs = obs_indices[0]
        delta[0] = self.pi * self.B[:, primeira_obs]
        
        # --- PASSO 2: Recursão (t=1 a T-1) [cite: 1898] ---
        for t in range(1, T):
            obs_atual = obs_indices[t]
            for s in range(self.n_estados):
                # Probabilidade de transição de todos os estados anteriores para o estado s
                # multiplicada pela probabilidade acumulada anterior
                prob_transicao = delta[t-1] * self.A[:, s]
                
                # Encontrar o estado anterior que maximiza essa probabilidade
                estado_anterior_max = np.argmax(prob_transicao)
                valor_max = prob_transicao[estado_anterior_max]
                
                # Atualizar delta e backpointer
                delta[t, s] = valor_max * self.B[s, obs_atual]
                psi[t, s] = estado_anterior_max
                
        # --- PASSO 3: Terminação [cite: 1900] ---
        # Encontrar o melhor estado final
        melhor_caminho = np.zeros(T, dtype=int)
        melhor_caminho[T-1] = np.argmax(delta[T-1])
        probabilidade_caminho = np.max(delta[T-1])
        
        # --- PASSO 4: Backtracking (Reconstrução do Caminho) [cite: 1864] ---
        for t in range(T-2, -1, -1):
            melhor_caminho[t] = psi[t+1, melhor_caminho[t+1]]
            
        # Converter índices de volta para nomes
        sequencia_estados = [self.estados[i] for i in melhor_caminho]
        
        return sequencia_estados, delta

    def visualizar_resultado(self, obs_seq, estados_seq, delta_matrix):
        """
        Gera uma visualização gráfica do processo de Viterbi (Heatmap).
        """
        plt.figure(figsize=(10, 6))
        
        # Criar Heatmap das probabilidades (Trellis)
        # Transpor para ter Estados no eixo Y e Tempo no eixo X
        sns.heatmap(delta_matrix.T, annot=True, fmt=".1e", cmap="YlGnBu",
                    xticklabels=obs_seq, yticklabels=self.estados)
        
        plt.title('Probabilidades do Algoritmo de Viterbi (Trellis)')
        plt.xlabel('Sequência de Observações (Tempo)')
        plt.ylabel('Estados Ocultos')
        
        # Salvar imagem
        plt.tight_layout()
        plt.savefig('viterbi_robo.png')
        print("\nGráfico salvo como 'viterbi_robo.png'")
        #plt.show()

# --- EXECUÇÃO DO CENÁRIO ---
if __name__ == "__main__":
    robo = RoboHMM()
    
    # Cenário: O robô sai da base, começa a limpar, bate em algo,
    # fica preso tentando sair, para (espera) e desiste.
    sequencia_logs = ['EmEspera', 'Normal', 'Normal', 'Colisao', 'Colisao', 'EmEspera']
    
    print(f"Sequência de Logs (Observada): {sequencia_logs}")
    print("-" * 50)
    
    estados_provaveis, matriz_delta = robo.viterbi(sequencia_logs)
    
    print(f"Diagnóstico (Sequência de Estados Mais Provável):")
    print(estados_provaveis)
    print("-" * 50)
    
    # Explicação passo a passo
    print("\nInterpretação do Diagnóstico:")
    for t, (obs, estado) in enumerate(zip(sequencia_logs, estados_provaveis)):
        print(f"Tempo {t}: Sensor '{obs}' -> Robô estava '{estado}'")

    robo.visualizar_resultado(sequencia_logs, estados_provaveis, matriz_delta)