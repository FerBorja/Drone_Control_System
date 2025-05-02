import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import gym
import numpy as np
import tensorflow as tf
from collections import deque
import random
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import cv2

class BalancedDroneEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        self.action_space = gym.spaces.Discrete(3)  # 0=Subir, 1=Bajar, 2=Mantener
        self.target_altitude = 10.0
        self.gravity = 0.05
        self.max_thrust = 0.2
        self.reset()
        
    def reset(self):
        self.state = np.array([0.0, self.target_altitude, 0.0, 0.0])  # [x, z, vx, vz]
        self.steps = 0
        return self.state.copy()
    
    def step(self, action):
        thrust = 0.0
        if action == 0:  # Subir
            thrust = self.max_thrust
        elif action == 1:  # Bajar
            thrust = -self.max_thrust * 0.4
            
        # Física del drone
        self.state[3] += thrust - self.gravity
        self.state[1] += self.state[3] * 0.1
        self.state[0] += self.state[2] * 0.1
        
        # Condiciones de terminación
        self.steps += 1
        crashed = self.state[1] <= 0
        out_of_bounds = abs(self.state[0]) > 20
        timeout = self.steps >= 300
        done = crashed or out_of_bounds or timeout
        
        # Sistema de recompensas mejorado
        alt_error = abs(self.state[1] - self.target_altitude)
        reward = 0.5 * (1.0 - (alt_error / self.target_altitude))
        
        if crashed:
            reward -= 5.0
        elif out_of_bounds:
            reward -= 3.0
        
        reward -= 0.1 * abs(self.state[3])
        
        if action != 2:
            reward -= 0.05
            
        return self.state.copy(), reward, done, {}

def build_drone_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                 loss='huber')
    return model

def generate_drone_gif(env, model, filename='drone_balance.gif'):
    frames = []
    state = env.reset()
    done = False
    
    for _ in range(400):  # Máximo 400 frames
        # Renderizar el entorno
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Dibujar el suelo
        cv2.line(img, (0, 350), (600, 350), (100, 100, 100), 2)
        
        # Dibujar altitud objetivo (línea roja)
        target_y = 350 - int(env.target_altitude * 20)
        cv2.line(img, (0, target_y), (600, target_y), (0, 0, 255), 1)
        
        # Dibujar dron (círculo verde)
        drone_x = 300 + int(state[0] * 15)
        drone_y = 350 - int(state[1] * 20)
        cv2.circle(img, (drone_x, drone_y), 8, (0, 255, 0), -1)
        
        # Información de estado
        cv2.putText(img, f"Altura: {state[1]:.2f}m", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, f"Velocidad: {state[3]:.2f}m/s", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        frames.append(Image.fromarray(img))
        
        # Tomar acción
        action = np.argmax(model.predict(state[np.newaxis], verbose=0)[0])
        state, _, done, _ = env.step(action)
        
        if done:
            break
    
    # Guardar GIF
    frames[0].save(filename,
                  save_all=True,
                  append_images=frames[1:],
                  duration=50,  # 50ms por frame (20fps)
                  loop=0)
    print(f"GIF guardado como {filename}")

def train_drone():
    env = BalancedDroneEnv()
    model = build_drone_model()
    target_model = build_drone_model()
    target_model.set_weights(model.get_weights())
    
    memory = deque(maxlen=50000)
    batch_size = 128
    gamma = 0.99
    epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay = 0.995
    
    rewards_history = []
    crash_count = 0
    
    for episode in range(500):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            if np.random.rand() <= epsilon:
                action = np.random.randint(3)
            else:
                action = np.argmax(model.predict(state[np.newaxis], verbose=0)[0])
            
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward
            
            if done and next_state[1] <= 0:
                crash_count += 1
            
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states = np.array([x[0] for x in batch])
                targets = model.predict(states, verbose=0)
                
                next_states = np.array([x[3] for x in batch])
                next_q = target_model.predict(next_states, verbose=0)
                
                for i, (_, a, r, _, d) in enumerate(batch):
                    targets[i][a] = r if d else r + gamma * np.max(next_q[i])
                
                model.train_on_batch(states, targets)
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        if episode % 10 == 0:
            target_model.set_weights(model.get_weights())
        
        rewards_history.append(episode_reward)
        print(f"Episodio {episode+1}, Recompensa: {episode_reward:.1f}, "
              f"Altura Final: {state[1]:.1f}, Choques: {crash_count}, ε: {epsilon:.2f}")
        
        if len(rewards_history) > 30 and np.mean(rewards_history[-30:]) > 25 and crash_count == 0:
            print("¡Modelo convergido y estable!")
            break
    
    return env, model, rewards_history

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    
    # Entrenamiento
    env, model, rewards = train_drone()
    
    # Guardar modelo
    model.save("drone_model_balanced.h5")
    
    # Gráfico de entrenamiento
    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    plt.title("Recompensas por Episodio (Sistema Balanceado)")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.grid(True)
    plt.savefig("recompensas_balanceadas.png")
    plt.show()
    
    # Generar GIF de demostración
    generate_drone_gif(env, model)
    
    print("Proceso completado. Resultados guardados:")
    print("- Modelo: drone_model_balanced.h5")
    print("- Gráfico: recompensas_balanceadas.png")
    print("- Animación: drone_balance.gif")