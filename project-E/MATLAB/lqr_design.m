dc_motor_model

Q = diag([1, 100]);  % Akım ve hız için ağırlıklar
R = 0.01;            % Kontrol girişi (voltaj) için ağırlık

%% LQR Kazanç
K = lqr(A, B_motor, Q, R);

%% Kapalı Çevrim Sistemi
Acl = A - B_motor*K;
sys_cl = ss(Acl, B_motor, C, D);

%% Step Response
step(sys_cl)
title('LQR Kontrollü DC Motor Step Response')
xlabel('Zaman (s)')
ylabel('Açısal Hız (rad/s)')
grid on
