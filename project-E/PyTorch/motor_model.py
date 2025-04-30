import numpy as np

def generate_motor_data(Tfinal=5, Ts=0.01, seed=0):
    """
    Lineer DC motor modeli için durum (state) ve girdi (input) verisi üretir.
    Dönüş Değerleri: states (N_steps, 2), inputs (N_steps, 1)
    """
    R, L = 2.0, 0.5
    Ke, Kt = 0.015, 0.015
    J, B = 0.02, 0.2

    # Sürekli zamanlı sistem matrisleri
    A = np.array([[-R/L, -Ke/L],
                  [ Kt/J, -B/J]])
    B_motor = np.array([[1/L],
                        [0]])
    # C matrisi burada doğrudan kullanılmıyor ama tanım olarak kalabilir
    # C = np.eye(2)

    # Ayrık zamanlı sistem matrisleri (Euler yöntemi)
    Ad = np.eye(2) + Ts * A
    Bd = Ts * B_motor

    N_steps = int(Tfinal / Ts)
    np.random.seed(seed)
    # Daha zengin veri için giriş sinyalini değiştirebilirsiniz
    # time_vec = np.arange(N_steps) * Ts
    # u_sequence = 2.0 * np.sin(time_vec * 2) + 3.0 * np.random.uniform(-1, 1, size=(N_steps, 1))
    u_sequence = np.random.uniform(-5, 5, size=(N_steps, 1)) # Orijinal rastgele giriş

    x = np.zeros((2, 1))  # Başlangıç durumu [i, omega] = [0, 0]
    state_list = []       # Durumları saklamak için liste
    input_list = []       # Girdileri saklamak için liste

    for k in range(N_steps):
        # Durumu güncelle (vektör/matris çarpımı için @ kullanıldı)
        # Bd'nin şekli (2,1), u_sequence[k]'nın (1,) veya skaler olması gerekir.
        # Eğer u_sequence[k] skaler ise Bd * u_sequence[k] kullanılmalı.
        # Eğer u_sequence[k] (1,1) ise Bd @ u_sequence[k] kullanılabilir.
        # Kodunuz Bd @ u_sequence[k] kullandığı için u_sequence'in (N_steps, 1) olduğunu varsayıyorum.
        x = Ad @ x + Bd * u_sequence[k, 0] # u_sequence[k,0] ile skaler çarpım daha güvenli

        # y = C @ x # y'yi ayrıca hesaplamaya gerek yok (eğer kullanılmıyorsa)

        # Anlık durumu ve girdiyi listelere ekle
        state_list.append(x.flatten())      # (2,) vektör olarak ekle
        input_list.append(u_sequence[k, 0]) # Skaler olarak ekle

    # Listeleri NumPy dizilerine dönüştür
    # Not: X_data_all N_steps+1 boyutunda bekleniyordu önceki kodda, burada N_steps boyutunda olacak.
    # Bunu çağıran kodda ele almak gerekebilir. Ancak NN eğitimi için bu format (N_steps) daha uygun.
    X_data = np.array(state_list)      # (N_steps, 2)
    U_data = np.array(input_list).reshape(-1, 1) # (N_steps, 1)

    # Sadece durumları ve girdileri döndür
    return X_data, U_data