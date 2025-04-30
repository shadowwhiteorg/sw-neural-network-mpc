%% Motor Modeli
motor_params
dc_motor_model

%% Diskretize Motor Modeli
Ts = 0.01;
sys_d = c2d(sys, Ts);
[Ad, Bd, Cd, Dd] = ssdata(sys_d);

%% Simülasyon Setup
Tfinal = 3; 
N_steps = Tfinal / Ts;
Np = 5;  % Prediction horizon

x = [0; 0];
y = zeros(N_steps,1);
u = zeros(N_steps,1);
t = (0:N_steps-1)' * Ts;

r = 1;  % İstenen hız
lambda = 1e-6;  % Kontrol ceza ağırlığı

%% Prediction Matrislerini Oluştur
% Tahmin modelini matris formunda kuracağız:
% Y = F*x(k) + Phi*U

% F matrisini oluşturalım
F = zeros(Np, size(Cd,1) * size(Ad,1));
for i = 1:Np
    F(i,:) = Cd * (Ad^i);
end

% Phi matrisini oluşturalım
Phi = zeros(Np, Np);
for row = 1:Np
    for col = 1:row
        Phi(row,col) = Cd * (Ad^(row-col)) * Bd;
    end
end

%% Simülasyon Döngüsü
for k = 1:N_steps
    % 1. Prediction: 
    % Y = F*x(k) + Phi*U
    % Cost Function: J = (Y - R)'*(Y - R) + lambda*(U'*U)
    
    % R vektörü (Referans)
    R_vec = r * ones(Np,1);
    
    % Cost fonksiyonunu kur:
    H = Phi'*Phi + lambda*eye(Np);  % Quadratic cost matrix
    f = (F*x - R_vec)'*Phi;          % Linear cost term
    
    % Solve QP: minimize (1/2)*U'*H*U + f*U
    U_opt = quadprog(2*H, 2*f', [], [], [], [], -5*ones(Np,1), 5*ones(Np,1));  % Kısıtlar: -5V ile +5V arası
    
    % 2. Uygula: sadece ilk kontrol eylemini uygula
    u_sat = U_opt(1);
    
    % 3. Sistemi güncelle
    x = Ad*x + Bd*u_sat;
    
    % 4. Kayıt
    y(k) = Cd*x + Dd*u_sat;
    u(k) = u_sat;
end

%% Sonuçları Çiz
figure;
subplot(2,1,1)
plot(t, y)
title('Multi-Step MPC (5 adım) ile Motor Açısal Hız')
xlabel('Zaman (s)')
ylabel('Açısal Hız (rad/s)')
grid on

subplot(2,1,2)
plot(t, u)
title('Multi-Step MPC (5 adım) ile Motor Voltajı')
xlabel('Zaman (s)')
ylabel('Voltaj (V)')
grid on
