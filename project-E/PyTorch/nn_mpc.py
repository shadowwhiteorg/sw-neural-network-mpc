import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import cvxpy as cp
from torch.autograd.functional import jacobian
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Ölçeklendirme için
from sklearn.metrics import mean_squared_error, r2_score # Metrikler için

from motor_model import generate_motor_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan Cihaz:", device)

# === Veri Üretme ve Hazırlama ===
Ts = 0.01
X_data_all, U_data_all = generate_motor_data(Tfinal=20, Ts=Ts, seed=42)

# Girdi (Z) ve Hedef (Y) oluşturma
# Z: k=0'dan N_steps-2'ye kadar olan [state(k), input(k)] çiftleri
# Y: k=1'den N_steps-1'e kadar olan state(k+1) değerleri
Z_data = np.hstack([X_data_all[:-1], U_data_all[:-1]]) 
Y_target = X_data_all[1:]                            

# Veriyi Eğitim ve Test Setlerine Ayırma
Z_train, Z_test, Y_train, Y_test = train_test_split(
    Z_data, Y_target, test_size=0.2, random_state=42, shuffle=False
)

print(f"Eğitim seti boyutları: Z={Z_train.shape}, Y={Y_train.shape}")
print(f"Test seti boyutları: Z={Z_test.shape}, Y={Y_test.shape}")

# === Veri Ölçeklendirme ===
input_scaler = StandardScaler()
Z_train_scaled = input_scaler.fit_transform(Z_train)
Z_test_scaled = input_scaler.transform(Z_test)

Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to(device)
Z_train_scaled_tensor = torch.tensor(Z_train_scaled, dtype=torch.float32).to(device)
Z_test_scaled_tensor = torch.tensor(Z_test_scaled, dtype=torch.float32).to(device)

# === NN Model Tanımı ===
class NeuralNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=2):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x_scaled):
        x = self.relu(self.fc1(x_scaled))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNet().to(device)

# === Yeniden Eğitim ===
print("\nÖlçeklenmiş Veriyle NN Eğitimi Başlatılıyor...")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
num_epochs = 1000
batch_size = 64
train_losses = []
test_losses = []

start_train_time = time.time()
for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(Z_train_scaled_tensor.size(0))
    epoch_train_loss = 0.0
    for i in range(0, Z_train_scaled_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_z, batch_y = Z_train_scaled_tensor[indices], Y_train_tensor[indices]
        optimizer.zero_grad()
        outputs = model(batch_z)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * batch_z.size(0)
    epoch_train_loss /= Z_train_scaled_tensor.size(0)
    train_losses.append(epoch_train_loss)

    model.eval()
    with torch.no_grad():
        test_outputs = model(Z_test_scaled_tensor)
        test_loss = criterion(test_outputs, Y_test_tensor).item()
        test_losses.append(test_loss)

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.6f}, Test Loss: {test_loss:.6f}')

end_train_time = time.time()
print(f"Eğitim {end_train_time - start_train_time:.2f} saniyede tamamlandı.")

# Eğitim ve Test Kayıp Grafikleri
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Eğitim Kaybı (Train Loss)')
plt.plot(test_losses, label='Test Kaybı (Test Loss)')
plt.title('Eğitim ve Test Kayıpları')
plt.xlabel('Epoch')
plt.ylabel('MSE Kaybı')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# NN Model Doğrulama
print("\nNN Model Doğrulaması (Açık Çevrim Tahminler)...")
model.eval()
with torch.no_grad():
    Y_pred_tensor = model(Z_test_scaled_tensor)
Y_pred_physical = Y_pred_tensor.cpu().numpy()

# --- Kantitatif Değerlendirme ---
mse_i = mean_squared_error(Y_test[:, 0], Y_pred_physical[:, 0])
mse_omega = mean_squared_error(Y_test[:, 1], Y_pred_physical[:, 1])
r2_i = r2_score(Y_test[:, 0], Y_pred_physical[:, 0])
r2_omega = r2_score(Y_test[:, 1], Y_pred_physical[:, 1])
print(f"\nTest Seti Performansı:")
print(f"Akım (i) - MSE: {mse_i:.6f}, R2 Skoru: {r2_i:.4f}")
print(f"Açısal Hız (ω) - MSE: {mse_omega:.6f}, R2 Skoru: {r2_omega:.4f}")


# === MPC SİMÜLASYONU ===
print("\nMPC Simülasyonu Başlatılıyor...")

# --- Hata Ayıklama ve Ayar Parametreleri ---
USE_TRUE_MODEL_FOR_MPC = True  # !!! True yaparsanız MPC NN yerine gerçek modeli kullanır !!!
PRINT_JACOBIANS = False        # !!! True ise ilk adımların Jacobianlarını yazdırır !!!
JACOBIAN_PRINT_STEPS = 5      # Yazdırılacak adım sayısı

Tfinal_sim = 5
N_sim = int(Tfinal_sim / Ts)
N_horizon = 30
umin, umax = -5.0, 5.0
r_target = 0.1    # !Hedef açısal hız (omega) değeri!

# Tutucu MPC Ayarları
Q_omega = 1000.0   # Durum hatası cezası
R_u = 1.0         # Kontrol eforu cezası
R_delta_u = 2.0   # Kontrol değişim oranı cezası

# Simülasyon başlangıç
state = np.array([0.0, 0.0], dtype=np.float32)
u_last = 0.0
sim_states = np.zeros((N_sim + 1, 2))
sim_states[0, :] = state
sim_controls = np.zeros((N_sim, 1))

# Gerçek motor modeli (Hem simülasyon hem de opsiyonel MPC modeli için)
R_mot, L_mot, Ke_mot, Kt_mot, J_mot, B_mot = 2.0, 0.5, 0.015, 0.015, 0.02, 0.2
A_mot_cont = np.array([[-R_mot/L_mot, -Ke_mot/L_mot], [Kt_mot/J_mot, -B_mot/J_mot]])
B_mot_cont = np.array([[1/L_mot], [0]])
Ad_mot = np.eye(2) + Ts * A_mot_cont
Bd_mot = Ts * B_mot_cont

model.eval()

sim_start_time = time.time()
for t in range(N_sim):
    # Jacobian Hesaplama 
    x_current = sim_states[t, :]

    if USE_TRUE_MODEL_FOR_MPC:
        # MPC için Gerçek Modeli Kullan
        A_t = Ad_mot
        B_t = Bd_mot
        c_t = np.zeros(2) # Gerçek lineer model için ofset sıfır
        if PRINT_JACOBIANS and t < JACOBIAN_PRINT_STEPS:
            print(f"\n--- t={t}: GERÇEK MODEL KULLANILIYOR ---")
            print(f"A_t (Ad_mot):\n{A_t}")
            print(f"B_t (Bd_mot):\n{B_t}")
    else:
        # NN Lineerleştirmesini Kullan
        z_lin_physical_np = np.array([[x_current[0], x_current[1], u_last]])
        try:
            z_lin_scaled_np = input_scaler.transform(z_lin_physical_np)
        except Exception as e:
            print(f"HATA: Input scaler transform sırasında hata t={t}: {e}")
            print("Simülasyon durduruluyor.")
            break
        z_lin_scaled = torch.tensor(z_lin_scaled_np, dtype=torch.float32).to(device)

        def predict_nn_wrapper_scaled(tensor_input_scaled):
             return model(tensor_input_scaled.view(1, -1))

        try:
            # Jacobian: dY_physical / dZ_scaled
            J_s = jacobian(predict_nn_wrapper_scaled, z_lin_scaled, create_graph=False).squeeze().detach().cpu().numpy()
            with torch.no_grad():
                y_lin_physical = model(z_lin_scaled).squeeze().detach().cpu().numpy()
        except Exception as e:
             print(f"HATA: Jacobian veya NN tahmini sırasında hata t={t}: {e}")
             print("Simülasyon durduruluyor.")
             break

        # Fiziksel Jacobian'ları Hesapla: dY_physical / dZ_physical = J_s / scale_factors
        if hasattr(input_scaler, 'scale_') and len(input_scaler.scale_) == 3:
            scale_factors = input_scaler.scale_
            # Bölme işleminde sıfıra bölme hatasını önle
            if np.any(np.abs(scale_factors) < 1e-9):
                print(f"UYARI: Çok küçük ölçek faktörü tespit edildi t={t} ({scale_factors}). Ölçeklemesiz Jacobian kullanılıyor.")
                A_t = J_s[:, :2]
                B_t = J_s[:, 2].reshape(-1, 1)
            else:
                A_t = J_s[:, :2] / scale_factors[:2] # Element-wise BÖLME
                B_t = (J_s[:, 2] / scale_factors[2]).reshape(-1, 1) # Element-wise BÖLME
        else:
            print(f"UYARI/HATA: Scaler uygun değil t={t}. Ölçeklemesiz Jacobian kullanılıyor.")
            A_t = J_s[:, :2]
            B_t = J_s[:, 2].reshape(-1, 1)

        # Ofset terimi (A_t ve B_t doğruysa bu da doğru olur)
        c_t = y_lin_physical - A_t @ x_current - B_t.flatten() * u_last

        # Adım 1: Jacobian Kontrolü için Yazdırma
        if PRINT_JACOBIANS and t < JACOBIAN_PRINT_STEPS:
             print(f"\n--- t={t}, x={x_current}, u={u_last:.3f} için NN Lineerleştirme ---")
             print(f"Scaler Faktörleri (i, w, u): {scale_factors}")
             print(f"Jacobian (Scaled Input) J_s:\n{J_s}")
             print(f"NN Tahmini y_lin: {y_lin_physical}")
             print(f"Fiziksel A_t:\n{A_t}")
             print(f"Gerçek Ad_mot:\n{Ad_mot}") # Karşılaştırma için
             print(f"Fiziksel B_t:\n{B_t}")
             print(f"Gerçek Bd_mot:\n{Bd_mot}") # Karşılaştırma için
             print(f"Ofset c_t: {c_t.flatten()}")


    # --- CVXPY Optimizasyon Problemi ---
    u_var = cp.Variable(N_horizon, name='u')
    x_var = cp.Variable((N_horizon + 1, 2), name='x')
    u_prev_param = cp.Parameter(value=u_last)

    cost = 0
    constraints = [x_var[0, :] == x_current]

    for k in range(N_horizon):
        constraints += [x_var[k+1, :] == A_t @ x_var[k, :] + B_t.flatten() * u_var[k] + c_t.flatten()]

        omega_pred = x_var[k+1, 1]
        stage_cost = Q_omega * cp.square(omega_pred - r_target) + R_u * cp.square(u_var[k])

        if k == 0:
            stage_cost += R_delta_u * cp.square(u_var[k] - u_prev_param)
        else:
            stage_cost += R_delta_u * cp.square(u_var[k] - u_var[k-1])

        cost += stage_cost
        constraints += [u_var[k] >= umin, u_var[k] <= umax]

    # --- Problemi Çöz ---
    problem = cp.Problem(cp.Minimize(cost), constraints)
    try:
        problem.solve(solver=cp.OSQP, verbose=False, warm_start=True, eps_abs=1e-4, eps_rel=1e-4)
    except cp.SolverError as e:
         print(f"!!! Solver Hatası t={t}: {e}. u=u_last uygulanıyor. !!!")
         u_apply = u_last # Hata durumunda önceki kontrolü kullan

    # --- Çözümü İşle ---
    if u_var.value is None or problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"⚠️ Optimizasyon çözülemedi veya başarısız oldu (status: {problem.status}) t={t}. u={u_last:.2f} uygulanıyor.")
        u_apply = u_last
    else:
         if problem.status == cp.OPTIMAL_INACCURATE:
             print(f"Uyarı: Çözücü OPTIMAL_INACCURATE döndü t={t}.")
         u_apply = float(u_var.value[0])

    u_apply = np.clip(u_apply, umin, umax)

    # --- Gerçek Sistemi Güncelle ---
    current_state_col = sim_states[t, :].reshape(-1, 1)
    next_state = Ad_mot @ current_state_col + Bd_mot * u_apply
    if t + 1 < N_sim + 1:
        sim_states[t+1, :] = next_state.flatten()
    sim_controls[t, 0] = u_apply
    u_last = u_apply

sim_end_time = time.time()
print(f"MPC simülasyonu {sim_end_time - sim_start_time:.2f} saniyede tamamlandı.")

# === GRAFİKLEME ===
t_vec_state = np.arange(N_sim + 1) * Ts
t_vec_control = np.arange(N_sim) * Ts

plt.figure(figsize=(12, 9)) 

# Açısal Hız Grafiği
plt.subplot(3, 1, 1) 
plt.plot(t_vec_state, sim_states[:, 1], label='Açısal Hız (ω) - NN-MPC')
plt.axhline(r_target, color='r', linestyle='--', label=f'Referans ({r_target})')
plt.title("True Model MPC Sonuçları")
plt.xlabel("Zaman (s)")
plt.ylabel("Açısal Hız (rad/s)")
plt.grid(True)
plt.legend()

# Akım Grafiği
plt.subplot(3, 1, 2)
plt.plot(t_vec_state, sim_states[:, 0], label='Akım (i)')
plt.title("Akım")
plt.xlabel("Zaman (s)")
plt.ylabel("Akım (A)")
plt.grid(True)
plt.legend()

# Kontrol Girişi Grafiği
plt.subplot(3, 1, 3)
plt.step(t_vec_control, sim_controls[:, 0], where='post', label='Uygulanan Voltaj (V)')
plt.axhline(umin, color='k', linestyle=':', alpha=0.7, label='Limitler')
plt.axhline(umax, color='k', linestyle=':', alpha=0.7)
plt.title("Kontrol Girişi (Voltaj)")
plt.xlabel("Zaman (s)")
plt.ylabel("Volt (V)")
plt.grid(True)
plt.legend()
plt.ylim(umin - 1, umax + 1)

plt.tight_layout()
plt.show()