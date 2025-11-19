# --- PROJETO 2: HMM COM ANÁLISE DIDÁTICA COMPLETA ---
# Explicação passo-a-passo do por quê cada resultado faz sentido

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

class RoboHMM:
    def __init__(self):
        self.estados = ['Limpando', 'Preso', 'Base']
        self.n_estados = len(self.estados)
        
        self.observacoes = ['Normal', 'Colisao', 'EmEspera']
        self.mapa_obs = {obs: i for i, obs in enumerate(self.observacoes)}
        
        # Probabilidade inicial (robô começa na Base ou Limpando)
        self.pi = np.array([0.4, 0.0, 0.6])
        
        # Transição: P(estado_novo | estado_anterior)
        self.A = np.array([
            # Para: Limp, Preso, Base
            [0.80, 0.10, 0.10],  # De: Limpando (maioria fica limpando)
            [0.40, 0.60, 0.00],  # De: Preso (tenta sair ou fica preso)
            [0.10, 0.00, 0.90]   # De: Base (maioria fica na base)
        ])
        
        # Emissão: P(observação | estado)
        self.B = np.array([
            # Obs: Normal, Colisao, EmEspera
            [0.80, 0.15, 0.05],  # Estado: Limpando
            [0.05, 0.55, 0.40],  # Estado: Preso
            [0.10, 0.00, 0.90]   # Estado: Base
        ])
        
    def explicar_logica(self, sequencia_obs):
        """
        Explicação didática: Por que cada observação leva a qual conclusão
        """
        print("\n" + "="*80)
        print("EXPLICAÇÃO DIDÁTICA: Por que o HMM escolhe cada estado")
        print("="*80)
        
        for t, obs in enumerate(sequencia_obs):
            obs_idx = self.mapa_obs[obs]
            print(f"\nTempo {t}: Observação = '{obs}'")
            print("-" * 60)
            print("Probabilidade dessa observação vir de cada estado:")
            
            # P(obs | estado) = B[estado, obs_idx]
            for s, estado in enumerate(self.estados):
                prob = self.B[s, obs_idx]
                print(f"  P('{obs}' | {estado:10s}) = {prob:.2f}")
            
            # Qual estado mais provavelmente gerou essa obs (sem contexto)?
            estado_melhor_sem_contexto = self.estados[np.argmax(self.B[:, obs_idx])]
            print(f"→ Sem contexto, '{obs}' vem melhor de: {estado_melhor_sem_contexto}")

    def diagnostico_ingenuo(self, sequencia_obs):
        """
        Baseline: Ignora TUDO, olha só P(obs|estado)
        (Isso é errado em geral!)
        """
        caminho_ingenuo = []
        for obs in sequencia_obs:
            obs_idx = self.mapa_obs[obs]
            prob_emissao = self.B[:, obs_idx]
            estado_idx = np.argmax(prob_emissao)
            caminho_ingenuo.append(self.estados[estado_idx])
        return caminho_ingenuo

    def viterbi(self, sequencia_obs):
        """
        Algoritmo de Viterbi: Encontra o CAMINHO mais provável no tempo
        Leva em conta: (1) Probabilidade inicial
                       (2) Transições entre estados
                       (3) Observações
        """
        T = len(sequencia_obs)
        delta = np.zeros((T, self.n_estados))
        psi = np.zeros((T, self.n_estados), dtype=int)
        obs_indices = [self.mapa_obs[o] for o in sequencia_obs]
        
        # PASSO 1: Inicialização (t=0)
        delta[0] = self.pi * self.B[:, obs_indices[0]]
        
        # PASSO 2: Recursão (t=1 até T-1)
        for t in range(1, T):
            obs_atual = obs_indices[t]
            for s in range(self.n_estados):
                # Probabilidade de TODAS as transições anteriores para s
                prob_transicao = delta[t-1] * self.A[:, s]
                estado_anterior_max = np.argmax(prob_transicao)
                valor_max = prob_transicao[estado_anterior_max]
                
                # Multiplicar pela prob de emitir obs_atual no estado s
                delta[t, s] = valor_max * self.B[s, obs_atual]
                psi[t, s] = estado_anterior_max
            
            # Normalização para evitar underflow
            delta[t] = delta[t] / (np.sum(delta[t]) + 1e-10)

        # PASSO 3: Terminação
        melhor_caminho = np.zeros(T, dtype=int)
        melhor_caminho[T-1] = np.argmax(delta[T-1])
        
        # PASSO 4: Backtracking
        for t in range(T-2, -1, -1):
            melhor_caminho[t] = psi[t+1, melhor_caminho[t+1]]
            
        sequencia_estados = [self.estados[i] for i in melhor_caminho]
        return sequencia_estados, delta

    def explicar_transicoes(self, viterbi_seq, ingenuo_seq, obs_seq):
        """
        Explica POR QUÊ o HMM diverge do ingênuo
        """
        print("\n" + "="*80)
        print("EXPLICAÇÃO: Por quê HMM e Ingênuo diferem?")
        print("="*80)
        
        for t in range(1, len(obs_seq)):
            est_vit_ant = viterbi_seq[t-1]
            est_vit_agr = viterbi_seq[t]
            est_ing_agr = ingenuo_seq[t]
            obs_atual = obs_seq[t]
            
            if est_vit_agr != est_ing_agr:
                print(f"\nTempo {t}: '{obs_atual}'")
                print(f"  Ingênuo escolheria: {est_ing_agr}")
                print(f"  HMM escolhe:        {est_vit_agr}")
                print(f"  Contexto: Robô estava em '{est_vit_ant}' no tempo anterior")
                
                # Verificar probabilidade de transição
                idx_ant = {'Limpando': 0, 'Preso': 1, 'Base': 2}[est_vit_ant]
                idx_novo_hmm = {'Limpando': 0, 'Preso': 1, 'Base': 2}[est_vit_agr]
                idx_novo_ing = {'Limpando': 0, 'Preso': 1, 'Base': 2}[est_ing_agr]
                
                prob_transicao_hmm = self.A[idx_ant, idx_novo_hmm]
                prob_transicao_ing = self.A[idx_ant, idx_novo_ing]
                
                print(f"  Prob transição para {est_vit_agr}: {prob_transicao_hmm:.2f}")
                print(f"  Prob transição para {est_ing_agr}: {prob_transicao_ing:.2f}")
                print(f"  → HMM leva em conta que {est_vit_agr} é mais provável dado o histórico")

    def visualizar_completo(self, obs_seq, viterbi_seq, ingenuo_seq, delta):
        """Visualização melhorada com anotações"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Heatmap Viterbi
        sns.heatmap(delta.T, annot=True, fmt=".2e", cmap="YlGnBu", ax=axes[0],
                    xticklabels=obs_seq, yticklabels=self.estados, cbar_kws={'label': 'Probabilidade'})
        axes[0].set_title('1. TRELLIS DE VITERBI: Confiança em cada estado ao longo do tempo', 
                         fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Estados Ocultos')

        # Plot 2: Comparação de caminhos
        tempo = range(len(obs_seq))
        mapa_y = {est: i for i, est in enumerate(self.estados)}
        y_viterbi = [mapa_y[s] for s in viterbi_seq]
        y_ingenuo = [mapa_y[s] for s in ingenuo_seq]

        axes[1].plot(tempo, y_viterbi, 'o-', label='HMM (Viterbi) - COM histórico', 
                    color='green', linewidth=2.5, markersize=8)
        axes[1].plot(tempo, y_ingenuo, 'x--', label='Ingênuo - SEM histórico', 
                    color='red', linewidth=2, markersize=8)
        
        axes[1].set_yticks(range(len(self.estados)))
        axes[1].set_yticklabels(self.estados)
        axes[1].set_xticks(range(len(obs_seq)))
        axes[1].set_xticklabels(obs_seq)
        axes[1].set_title('2. COMPARAÇÃO: HMM vs Ingênuo', fontweight='bold', fontsize=12)
        axes[1].legend(loc='upper left', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylabel('Estado Inferido')

        # Plot 3: Observações como referência
        obs_idx = [self.mapa_obs[o] for o in obs_seq]
        colors = ['green' if o == 0 else 'orange' if o == 1 else 'blue' for o in obs_idx]
        axes[2].bar(tempo, obs_idx, color=colors, alpha=0.6, edgecolor='black', linewidth=1.5)
        axes[2].set_yticks([0, 1, 2])
        axes[2].set_yticklabels(self.observacoes)
        axes[2].set_xticks(range(len(obs_seq)))
        axes[2].set_xticklabels(obs_seq)
        axes[2].set_title('3. OBSERVAÇÕES RUIDOSAS (Ground truth = desconhecido)', 
                         fontweight='bold', fontsize=12)
        axes[2].set_ylabel('Tipo de Observação')
        axes[2].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('analise_hmm_completa.png', dpi=150)
        print("\nGráfico salvo como 'analise_hmm_completa.png'")


# --- EXECUÇÃO ---
if __name__ == "__main__":
    robo = RoboHMM()
    
    logs = ['EmEspera', 'Normal', 'Normal', 'Colisao', 'Colisao', 'EmEspera']
    
    print("\n" + "#"*80)
    print("# MODELO MARKOVIANO OCULTO (HMM) - DIAGNÓSTICO DE ROBÔ ASPIRADOR")
    print("#"*80)
    print(f"\nSequência Observada: {logs}")
    
    # --- EXPLICAÇÕES ---
    robo.explicar_logica(logs)
    
    # --- EXECUÇÃO DOS ALGORITMOS ---
    path_viterbi, matriz_delta = robo.viterbi(logs)
    path_ingenuo = robo.diagnostico_ingenuo(logs)
    
    robo.explicar_transicoes(path_viterbi, path_ingenuo, logs)
    
    # --- TABELA COMPARATIVA ---
    df_comp = pd.DataFrame({
        'Tempo': range(len(logs)),
        'Observação': logs,
        'Diagnóstico Ingênuo': path_ingenuo,
        'Diagnóstico HMM': path_viterbi,
        'Acorta?': ['Diferente' if path_ingenuo[i] != path_viterbi[i] else 'Igual' 
                    for i in range(len(logs))]
    })
    
    print("\n" + "="*80)
    print("TABELA COMPARATIVA")
    print("="*80)
    print(df_comp.to_string(index=False))
    
    # --- INSIGHTS ---
    print("\n" + "="*80)
    print("INSIGHTS PRINCIPAIS")
    print("="*80)
    print(f"""
 OBSERVAÇÃO AMBÍGUA: 'EmEspera' aparece no início (t=0) e no fim (t=5)
  - No início: Muito provavelmente = 'Base' (robô na estação de carga)
  - No fim:    Muito provavelmente = 'Preso' (robô travado após colisões)

 O HMM leva em conta o CONTEXTO TEMPORAL:
  - No tempo 0: π (Prior) diz que é provável estar em Base → 'Base'
  - No tempo 5: Histórico de colisões força conclusão de 'Preso'

 O diagnóstico ingênuo é ENGANADO porque:
  - Olha APENAS para P(obs|estado), ignora o histórico
  - Faz o mesmo erro em ambos os 'EmEspera'

 Valor de um HMM:
  - Distingue entre causas diferentes da MESMA observação
  - Realista: sensores são ruidosos, mas estado evolui com REGRAS
""")
    
    # --- VISUALIZAÇÃO ---
    robo.visualizar_completo(logs, path_viterbi, path_ingenuo, matriz_delta)
    
    print("\n Análise completa gerada!\n")