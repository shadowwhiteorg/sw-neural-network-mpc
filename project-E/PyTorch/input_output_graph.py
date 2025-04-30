import numpy as np
import matplotlib.pyplot as plt
from motor_model import generate_motor_data

# Veri üretimi
X_data, U_data, Y_data = generate_motor_data(Tfinal=5, Ts=0.01)

# Zaman vektörü
Ts = 0.01
N_steps = X_data.shape[0]
t_vec = np.arange(N_steps) * Ts

# Grafik çizimi
plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(t_vec, U_data)
plt.title('Uygulanan Giriş (Volt)')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(t_vec, X_data[:,1])  # omega açısal hız
plt.title('Çıkış (Açısal Hız)')
plt.grid(True)

plt.tight_layout()
plt.savefig("motor_input_output.png", dpi=300)
plt.show()
