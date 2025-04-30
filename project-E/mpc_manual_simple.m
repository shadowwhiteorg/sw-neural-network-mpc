%% Motor Modeli
motor_params
dc_motor_model
% Motor parametreleri

%% Diskretize Motor Modeli
Ts = 0.01;  % 10 ms
sys_d = c2d(sys, Ts);
[Ad, Bd, Cd, Dd] = ssdata(sys_d);

%% Simülasyon Setup
Tfinal = 3;  % 3 saniye
N_steps = Tfinal / Ts;

x = [0; 0];  % Başlangıç durumu
y = zeros(N_steps,1);
u = zeros(N_steps,1);
t = (0:N_steps-1)' * Ts;

r = 1;  % Hedef hız 1 rad/s
lambda = 1e-6;  % Kontrol hareket cezası

%% Simülasyon Döngüsü (Basit MPC - 1 adım ileri bakış)
for k = 1:N_steps
    % Gelecek adım tahmini:
    % x(k+1) = Ad*x(k) + Bd*u
    % y(k+1) = Cd*x(k+1) + Dd*u
    
    % Cost fonksiyonu:
    % J = (y_pred - r)^2 + lambda * u^2
    % Buradan J'yi minimize edecek u bulunacak.
    
    % J(u) = ((Cd*(Ad*x + Bd*u) + Dd*u) - r)^2 + lambda*u^2
    
    % Açalım:
    E = Cd*Bd + Dd;   % u katsayısı
    F = Cd*Ad*x;      % sabit terim
    
    % J(u) = (E*u + F - r)^2 + lambda*u^2
    % Parantez açılırsa:
    % J(u) = (E^2 + lambda)*u^2 + 2*E*(F - r)*u + (F - r)^2
    
    % Optimal u için türev alıp sıfıra eşitliyoruz:
    numerator = -2*E*(F - r);
    denominator = 2*(E'*E + lambda);
    
    u_opt = numerator / denominator;
    
    % Uygulanan u, kısıtlarla sınırlandıralım (örneğin ±5V)
    u_sat = min(max(u_opt, -5), 5);
    
    % Sisteme uygula
    x = Ad*x + Bd*u_sat;
    
    % Kayıt
    y(k) = Cd*x + Dd*u_sat;
    u(k) = u_sat;
end

%% Sonuçları Çiz
figure;
subplot(2,1,1)
plot(t, y)
title('Basit MPC (1 adım) ile Motor Açısal Hız')
xlabel('Zaman (s)')
ylabel('Açısal Hız (rad/s)')
grid on

subplot(2,1,2)
plot(t, u)
title('Basit MPC (1 adım) ile Motor Voltajı')
xlabel('Zaman (s)')
ylabel('Voltaj (V)')
grid on
