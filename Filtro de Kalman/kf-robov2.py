# --- EXTENDED KALMAN FILTER (EKF) PARA CÍRCULO ---
# Versão melhorada para rastreamento circular com aceleração centrípeta

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class ExtendedKalmanFilterCircle:
    def __init__(self, dt, std_acc, x_std_meas, y_std_meas):
        """
        EKF em coordenadas polares (r, θ).
        Estado: [r, theta, theta_dot] onde theta_dot é a velocidade angular
        """
        self.dt = dt
        
        # Estado inicial: raio 10, ângulo 0, velocidade angular 1 rad/s
        self.x = np.array([[10.0], [0.0], [1.0]])  # [r, theta, omega]
        
        # Covariância do estado inicial (bem calibrada)
        self.P = np.diag([0.1, 0.1, 0.1])
        
        # Ruído do processo (aceleração angular)
        self.Q = np.diag([0.01, 0.05, std_acc**2])
        
        # Ruído de medição (posição x, y)
        self.R = np.diag([x_std_meas**2, y_std_meas**2])
        
    def f(self, x):
        """Modelo dinâmico não-linear (em coordenadas polares)"""
        r = x[0, 0]
        theta = x[1, 0]
        omega = x[2, 0]
        
        # Raio é constante em um círculo perfeito
        r_new = r
        # Ângulo evolui com velocidade angular
        theta_new = theta + omega * self.dt
        # Velocidade angular é constante (ou com perturbação pequena)
        omega_new = omega
        
        return np.array([[r_new], [theta_new], [omega_new]])
    
    def h(self, x):
        """Função de observação: converte (r, theta) para (x, y)"""
        r = x[0, 0]
        theta = x[1, 0]
        
        x_cart = r * np.cos(theta)
        y_cart = r * np.sin(theta)
        
        return np.array([[x_cart], [y_cart]])
    
    def jacobian_F(self, x):
        """Jacobiana da função dinâmica f"""
        omega = x[2, 0]
        
        F = np.array([
            [1, 0, 0],
            [0, 1, self.dt],
            [0, 0, 1]
        ])
        return F
    
    def jacobian_H(self, x):
        """Jacobiana da função de observação h"""
        r = x[0, 0]
        theta = x[1, 0]
        
        H = np.array([
            [np.cos(theta), -r * np.sin(theta), 0],
            [np.sin(theta), r * np.cos(theta), 0]
        ])
        return H
    
    def predict(self):
        """Etapa de predição do EKF"""
        # Predição do estado
        self.x = self.f(self.x)
        
        # Linearização
        F = self.jacobian_F(self.x)
        
        # Predição da covariância
        self.P = F @ self.P @ F.T + self.Q
        
        # Retornar posição cartesiana
        x_cart = self.x[0, 0] * np.cos(self.x[1, 0])
        y_cart = self.x[0, 0] * np.sin(self.x[1, 0])
        
        return x_cart, y_cart
    
    def update(self, z):
        """Etapa de atualização do EKF"""
        # Observação esperada
        z_pred = self.h(self.x)
        
        # Jacobiana da observação
        H = self.jacobian_H(self.x)
        
        # Inovação (residual)
        y = z - z_pred
        
        # Covariância da inovação
        S = H @ self.P @ H.T + self.R
        
        # Ganho de Kalman
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Atualização do estado
        self.x = self.x + K @ y
        
        # Atualização da covariância
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P
        
        # Retornar posição atualizada em cartesiano
        x_cart = self.x[0, 0] * np.cos(self.x[1, 0])
        y_cart = self.x[0, 0] * np.sin(self.x[1, 0])
        
        return x_cart, y_cart


def gerar_trajetoria_circular(t, raio=10, omega=1):
    """Gera um círculo perfeito"""
    x = raio * np.cos(omega * t)
    y = raio * np.sin(omega * t)
    return x, y


# --- SIMULAÇÃO ---
if __name__ == "__main__":
    dt = 0.1
    tempo_total = np.arange(0, 20, dt)
    
    # Inicializa o filtro estendido
    ekf = ExtendedKalmanFilterCircle(dt=dt, 
                                      std_acc=0.3,
                                      x_std_meas=3.0, 
                                      y_std_meas=3.0)
    
    real_track = []
    measurements = []
    ekf_track = []
    
    print("Iniciando Simulação com EKF...")
    print(f"Estado inicial (polar): r={ekf.x[0, 0]:.2f}, θ={ekf.x[1, 0]:.2f}, ω={ekf.x[2, 0]:.2f}")
    
    for t in tempo_total:
        # Ground Truth
        rx, ry = gerar_trajetoria_circular(t)
        real_track.append((rx, ry))
        
        # Medição Ruidosa
        mx = rx + np.random.normal(0, 3)
        my = ry + np.random.normal(0, 3)
        measurements.append((mx, my))
        
        # EKF: Predição + Atualização
        ekf.predict()
        ex, ey = ekf.update(np.array([[mx], [my]]))
        ekf_track.append((ex, ey))

    # Conversão para arrays
    real_track = np.array(real_track)
    measurements = np.array(measurements)
    ekf_track = np.array(ekf_track)
    
    # Métricas
    mse_medicao = mean_squared_error(real_track, measurements)
    mse_ekf = mean_squared_error(real_track, ekf_track)
    
    print(f"\n{'='*50}")
    print(f"Erro Médio das Medições: {mse_medicao:.4f}")
    print(f"Erro Médio do EKF:       {mse_ekf:.4f}")
    
    melhoria = (1 - mse_ekf/mse_medicao) * 100
    print(f"Melhoria de Precisão:    {melhoria:.2f}%")
    print(f"{'='*50}")

    # Visualização
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico 1: Trajetória 2D
    ax1.plot(real_track[:,0], real_track[:,1], label='Trajetória Real', 
             color='green', linewidth=3, zorder=3)
    ax1.scatter(measurements[:,0], measurements[:,1], label='Medições Ruidosas', 
                color='red', alpha=0.3, s=20, zorder=1)
    ax1.plot(ekf_track[:,0], ekf_track[:,1], label='Estimativa EKF', 
             color='blue', linewidth=2, zorder=2)
    
    ax1.set_title('Rastreamento com Extended Kalman Filter', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Posição X (m)')
    ax1.set_ylabel('Posição Y (m)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Gráfico 2: Erro ao longo do tempo
    erro_medicao = np.sqrt(np.sum((real_track - measurements)**2, axis=1))
    erro_ekf = np.sqrt(np.sum((real_track - ekf_track)**2, axis=1))
    
    ax2.plot(tempo_total, erro_medicao, label='Erro Medição', 
             color='red', alpha=0.7, linewidth=1.5)
    ax2.plot(tempo_total, erro_ekf, label='Erro EKF', 
             color='blue', linewidth=2)
    ax2.fill_between(tempo_total, erro_medicao, alpha=0.2, color='red')
    ax2.fill_between(tempo_total, erro_ekf, alpha=0.2, color='blue')
    
    ax2.set_title('Evolução do Erro ao Longo do Tempo', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Erro Euclidiano (m)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kalman_trackingv2.png', dpi=150)
    print("\nGráfico salvo como 'kalman_trackingv2.png'")