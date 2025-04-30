motor_params

%% Durum Uzayı Matrisleri
A = [-R/L, -Ke/L;
      Kt/J, -B/J];

B_motor = [1/L; 0];
C = [0 1];
D = 0;

%% Sistem Oluşturulması
sys = ss(A, B_motor, C, D);

% step(sys)
% title('DC Motor Step Response')
% xlabel('Zaman (s)')
% ylabel('Açısal Hız (rad/s)')
% grid on
