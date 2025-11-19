# --- PROJETO 3: FILTRO DE KALMAN (Rastreamento 2D) - VERSÃO FINAL v3 ---
# Disciplina: FGA0221 - Inteligência Artificial
# Tema: Tratando Incerteza
# -----------------------------------------------------------------

import numpy as np
import matplotlib
matplotlib.use('Agg') # Modo sem janela (para evitar erro GTK)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class KalmanFilter2D:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas, initial_x, initial_y):
        """
        Inicializa o Filtro de Kalman.
        """
        # Variáveis de controle
        self.u = np.matrix([[u_x],[u_y]])
        
        # --- CORREÇÃO DE INICIALIZAÇÃO ---
        # Define explicitamente o ponto de partida
        print(f"DEBUG: Inicializando Filtro em X={initial_x}, Y={initial_y}")
        self.x = np.matrix([[initial_x], [initial_y], [0], [0]])
        
        # Matrizes do Modelo
        self.A = np.matrix([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        
        self.B = np.matrix([[(dt**2)/2, 0],
                            [0, (dt**2)/2],
                            [dt, 0],
                            [0, dt]])
        
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        
        # Matrizes de Covariância
        self.Q = np.matrix([[(dt**4)/4, 0, (dt**3)/2, 0],
                            [0, (dt**4)/4, 0, (dt**3)/2],
                            [(dt**3)/2, 0, dt**2, 0],
                            [0, (dt**3)/2, 0, dt**2]]) * std_acc**2
        
        self.R = np.matrix([[x_std_meas**2, 0],
                            [0, y_std_meas**2]])
        
        self.P = np.eye(self.A.shape[1])
        
    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        # Acessando índices da matriz corretamente [linha, coluna]
        return self.x[0, 0], self.x[1, 0]

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Atualização do Estado
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        
        # Atualização da Covariância
        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P
        
        # --- CORREÇÃO DO WARNING DO NUMPY ---
        # Extração correta de escalar de uma matriz np.matrix
        return float(self.x[0, 0]), float(self.x[1, 0])


def gerar_trajetoria_circular(t):
    """Gera um círculo perfeito (Ground Truth)"""
    x = 10 * np.cos(t)
    y = 10 * np.sin(t)
    return x, y

# --- SIMULAÇÃO ---
if __name__ == "__main__":
    dt = 0.1
    tempo_total = np.arange(0, 20, dt)
    
    # 1. Pega a posição inicial real
    start_x, start_y = gerar_trajetoria_circular(0)
    
    # 2. Inicializa o filtro
    # std_acc=2.0: Balanço entre inércia e capacidade de fazer curvas
    kf = KalmanFilter2D(dt=dt, u_x=0, u_y=0, 
                        std_acc=1.55,      
                        x_std_meas=3.0, 
                        y_std_meas=3.0, 
                        initial_x=start_x, 
                        initial_y=start_y)
    
    real_track = []
    measurements = []
    kalman_track = []
    
    print("Iniciando Simulação...")
    
    for t in tempo_total:
        # Ground Truth
        rx, ry = gerar_trajetoria_circular(t)
        real_track.append((rx, ry))
        
        # Medição Ruidosa
        mx = rx + np.random.normal(0, 3)
        my = ry + np.random.normal(0, 3)
        measurements.append((mx, my))
        
        # Kalman
        kf.predict()
        kx, ky = kf.update(np.matrix([[mx], [my]]))
        kalman_track.append((kx, ky))

    # Métricas
    real_track = np.array(real_track)
    measurements = np.array(measurements)
    kalman_track = np.array(kalman_track)
    
    mse_medicao = mean_squared_error(real_track, measurements)
    mse_kalman = mean_squared_error(real_track, kalman_track)
    
    print(f"Erro Médio das Medições: {mse_medicao:.4f}")
    print(f"Erro Médio do Kalman:    {mse_kalman:.4f}")
    
    melhoria = (1 - mse_kalman/mse_medicao)*100
    print(f"Melhoria de Precisão: {melhoria:.2f}%")

    # Visualização
    plt.figure(figsize=(10, 8))
    plt.plot(real_track[:,0], real_track[:,1], label='Trajetória Real', color='green', linewidth=3, zorder=3)
    plt.scatter(measurements[:,0], measurements[:,1], label='Medições Ruidosas', color='red', alpha=0.3, s=20, zorder=1)
    plt.plot(kalman_track[:,0], kalman_track[:,1], label='Estimativa Kalman', color='blue', linewidth=2, zorder=2)
    
    plt.title('Rastreamento de Robô: Fusão de Sensores com Filtro de Kalman')
    plt.xlabel('Posição X (m)')
    plt.ylabel('Posição Y (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.savefig('kalman_trackingv1.png')
    print("Gráfico salvo como 'kalman_trackingv1.png'")