%% Motor Modeli
motor_params
dc_motor_model

%% MPC Setup
Ts = 0.01;  % 10 ms
sys_d = c2d(sys, Ts);  % Sistem diskretize ediliyor
[Ad, Bd, Cd, Dd] = ssdata(sys_d);  % Diskretize matrisleri çıkarıyoruz

PredictionHorizon = 20;
ControlHorizon = 5;

mpc_controller = mpc(sys_d, Ts, PredictionHorizon, ControlHorizon);

mpc_controller.Weights.ManipulatedVariables = 0.01;
mpc_controller.Weights.ManipulatedVariablesRate = 0.1;
mpc_controller.Weights.OutputVariables = 1;

mpc_controller.MV.Min = -5;
mpc_controller.MV.Max = 5;

%% Simülasyon Setup
Tfinal = 3;  % 3 saniye
N_steps = Tfinal / Ts;
r = ones(N_steps, 1);

x = [0; 0];  % Başlangıç sistem durumu
y = zeros(N_steps, 1);
u = zeros(N_steps, 1);
t = (0:N_steps-1)' * Ts;

mpc_state = mpcstate(mpc_controller);  % MPC durumu oluşturuluyor

%% Simülasyon Döngüsü
for k = 1:N_steps
    y_measured = Cd*x;  % Diskretize çıkışı kullanıyoruz
    mv = mpcmove(mpc_controller, mpc_state, y_measured, r(k));
    
    % Sisteme uygula (Diskretize sistemde!)
    x = Ad*x + Bd*mv;
    
    % Çıkışı kaydet
    y(k) = Cd*x + Dd*mv;
    u(k) = mv;
end

%% Sonuçları Çiz
figure;
subplot(2,1,1)
plot(t, y)
title('MPC ile Motor Açısal Hız (Çıkış)')
xlabel('Zaman (s)')
ylabel('Açısal Hız (rad/s)')
grid on

subplot(2,1,2)
plot(t, u)
title('MPC ile Motor Voltajı (Giriş)')
xlabel('Zaman (s)')
ylabel('Uygulanan Voltaj (V)')
grid on
