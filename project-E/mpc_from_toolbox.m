%% Motor Modeli
motor_params
dc_motor_model

%% MPC Controller
%% 1. Motor Modelini Discrete Yapıyoruz
Ts = 0.01;  % 10 ms
sys_d = c2d(sys, Ts);  % Discrete time model

%% 2. MPC Controller'ı Kuruyoruz
PredictionHorizon = 20;
ControlHorizon = 5;

mpc_controller = mpc(sys_d, Ts, PredictionHorizon, ControlHorizon);

% Ağırlıklar
mpc_controller.Weights.ManipulatedVariables = 0.01;
mpc_controller.Weights.ManipulatedVariablesRate = 0.1;
mpc_controller.Weights.OutputVariables = 1;

% Voltaj kısıtları
mpc_controller.MV.Min = -5;
mpc_controller.MV.Max = 5;

%% Simülasyon Ayarları
Tfinal = 3;  % 2 saniye simülasyon süresi

r = ones(Tfinal/Ts, 1);  % Hedef (reference) sinyali - 1 rad/s sabit hız istiyoruz

% Simülasyonu başlat
[y, t, u] = sim(mpc_controller, Tfinal, r);

% %% Sonuçları Plotlama
figure;
subplot(2,1,1)
plot(t, y)
title('Motor Açısal Hız (Çıkış)')
xlabel('Zaman (s)')
ylabel('Açısal Hız (rad/s)')
grid on

subplot(2,1,2)
plot(t, u)
title('Motor Voltajı (Giriş)')
xlabel('Zaman (s)')
ylabel('Uygulanan Voltaj (V)')
grid on
