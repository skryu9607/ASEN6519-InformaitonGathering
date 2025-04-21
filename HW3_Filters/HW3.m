% Homework 3
% SeungKeol Ryu
% Mar 3rd
% Exploration for filters
global delta_t
%%% Description of the problem %%%%
% xk : two dimensional position of the target
% x_k+1 = ft(xk,wk) : where wk is process noise
% xki = [xki,yki,psii]^T : N total sensors
% uki = control input of the ith senstor : xik+1 = fs(xki,uki)
% zki = h(xk,xki,vki) : measurement model : measure model is a function of
% the target state and the sensor state.

%%% Sensor Motion Model %%%
% Kinematic unicycle model 
% delT : sample time, the sensor speed is constrained by two things
% turn rate is also bounded.

%%% Target Model %%%
% Constant velocity model : the process noise is the uncertain
% acceleration. The target model has "four" states.
% noise's dimension is two

%%% Measurement model %%%
% The range - bearing sensor : zki = [rki,betaki]

%% Problem 2 : simulate the following scenario
%%% Generate the "sensor" path, "target" path, and sensor "measurements" %%%
clc;clear all;close all;
delta_t = 1;FinalTime = 200;
x0 = [2000,0,12,5.5]';Q = 0.05 * eye(2)

x0s = [0,0,60*pi/180]';
sensorspeed = 20;
R = [5^2,0;0,(10*pi/180)^2];
mean_process_noise = [0,0]';
TargetPath = [x0];

for k = 1:200
    process_noise = mvnrnd(mean_process_noise,Q)';
    x0_new = ConstantVelocityTargetModel(x0,process_noise,1);
    TargetPath = [TargetPath,x0_new];
    x0 = x0_new;
end
figure(1);hold on;
plot(TargetPath(1,:),TargetPath(2,:),'r-.')
SensorPath = [x0s];
for k = 1:200
    if 89<=k && k<=110
        turnrate(k) = -0.05;
    else
        turnrate(k) = 0.0;
    end

    x0s_new = SensorMotionModel(x0s,[sensorspeed,turnrate(k)]',1);
    SensorPath = [SensorPath,x0s_new];
    x0s = x0s_new;
end
plot(SensorPath(1,:),SensorPath(2,:),'b-.');
xlabel("m");ylabel("m");
%axis equal;
title("Target and Sensor path")

ExactSensorPath = SensorPath;
ExactTargetPath = TargetPath;

%% Problem 3 : Extended Kalman Filter (EKF) %%
% R :  measurement noise
params.R = R;
% Q : process noise
params.Q = Q;

params.delta_t = 1;
% Initial measuremt of system and environment
InitialMeasurement.mean = [2400,-200,8,8.5]';
InitialMeasurement.cov = [100^2*eye(2),zeros(2,2);zeros(2,2),10^2*eye(2)];

xhat_old = InitialMeasurement.mean;
cov_old = InitialMeasurement.cov;

TargetPathEKF = [InitialMeasurement.mean];
TargetCOVEKF = cell(1,FinalTime+1);TargetCOVEKF{1} = InitialMeasurement.cov;

tic;
% Using EKF
for k = 1:200
    [xhat,P] = ExtendedKF(xhat_old,cov_old,ExactSensorPath(:,k),params,ExactTargetPath(:,k));
    TargetPathEKF = [TargetPathEKF,xhat];
    TargetCOVEKF{k+1} = P;
    xhat_old = xhat;
    cov_old = P;
end
elasped_time = toc;
disp(['EKF of elapsed time : ', num2str(elasped_time), ' seconds']);
plot(TargetPathEKF(1,:),TargetPathEKF(2,:),'g')

% =============================
% 📌 Figure 2: EKF Estimation Errors + 3-σ Bounds
% =============================
time = 0:1:200;
sigma_bounds = zeros(201, 4);

for k = 1:201
    P_k = TargetCOVEKF{k};
    sigma_bounds(k, :) = 3 * sqrt([P_k(1,1), P_k(2,2), P_k(3,3), P_k(4,4)]);
end

figure(2);
subplot(4,1,1);
plot(time, TargetPathEKF(1,:) - ExactTargetPath(1,:), 'r', time, sigma_bounds(:,1), 'b--', time, -sigma_bounds(:,1), 'b--');
title('X Position Error');
xlabel('Time (s)');
ylabel('Error (m)');

subplot(4,1,2);
plot(time, TargetPathEKF(2,:) - ExactTargetPath(2,:), 'g', time, sigma_bounds(:,2), 'b--', time, -sigma_bounds(:,2), 'b--');
title('Y Position Error');
xlabel('Time (s)');
ylabel('Error (m)');

subplot(4,1,3);
plot(time, TargetPathEKF(3,:) - ExactTargetPath(3,:), 'm', time, sigma_bounds(:,3), 'b--', time, -sigma_bounds(:,3), 'b--');
title('X Velocity Error');
xlabel('Time (s)');
ylabel('Error (m/s)');

subplot(4,1,4);
plot(time, TargetPathEKF(4,:) - ExactTargetPath(4,:), 'c', time, sigma_bounds(:,4), 'b--', time, -sigma_bounds(:,4), 'b--');
title('Y Velocity Error');
xlabel('Time (s)');
ylabel('Error (m/s)');

sgtitle('Estimation Errors with 3-sigma Bounds (EKF)');

%% Problem 4 : Unscented Kalman Filter (UKF) %%
%%% A sigma point Filter. 

% Initial measuremt of system and environment
InitialMeasurement.mean = [2400,-200,8,8.5]';
InitialMeasurement.cov = [100^2*eye(2),zeros(2,2);zeros(2,2),10^2*eye(2)];

xhat_old = InitialMeasurement.mean;
cov_old = InitialMeasurement.cov;

TargetPathUKF = [InitialMeasurement.mean];
TargetCOVUKF = cell(1,FinalTime+1);TargetCOVUKF{1} = InitialMeasurement.cov;
% Parameters for UKF
params.Lp = 4 + 2;
params.alpha = 10^-3;
params.beta = 2;
params.kappa = 0;
params.lambda = params.alpha^2*(params.Lp + params.kappa) - params.Lp;
% Loop
tic;
% Using UKF
for k = 1:200
    [xhat,P] = UnscentedKF(xhat_old,cov_old,ExactSensorPath(:,k),params,ExactTargetPath(:,k));
    TargetPathUKF = [TargetPathUKF,xhat];
    TargetCOVUKF{k+1} = P;
    xhat_old = xhat;
    cov_old = P;
end
elasped_time = toc;
disp(['UKF of elapsed time : ', num2str(elasped_time), ' seconds']);
figure(1);
plot(TargetPathUKF(1,:),TargetPathUKF(2,:),'m')
legend("Target","Sensor","EKF","UKF",'Location','best');figure(3);
time = 0:1:200;
sigma_bounds = zeros(200, 4);

for k = 1:201
    P_k = TargetCOVUKF{k};
    sigma_bounds(k, :) = 3 * sqrt([P_k(1,1), P_k(2,2), P_k(3,3), P_k(4,4)]);
end

figure(3);
subplot(4,1,1);
plot(time, TargetPathUKF(1,:) - ExactTargetPath(1,:), 'r', time, sigma_bounds(:,1), 'b--', time, -sigma_bounds(:,1), 'b--');
title('X Position Error');
xlabel('Time (s)');
ylabel('Error (m)');

subplot(4,1,2);
plot(time, TargetPathUKF(2,:) - ExactTargetPath(2,:), 'g', time, sigma_bounds(:,2), 'b--', time, -sigma_bounds(:,2), 'b--');
title('Y Position Error');
xlabel('Time (s)');
ylabel('Error (m)');

subplot(4,1,3);
plot(time, TargetPathUKF(3,:) - ExactTargetPath(3,:), 'm', time, sigma_bounds(:,3), 'b--', time, -sigma_bounds(:,3), 'b--');
title('X Velocity Error');
xlabel('Time (s)');
ylabel('Error (m/s)');

subplot(4,1,4);
plot(time, TargetPathUKF(4,:) - ExactTargetPath(4,:), 'c', time, sigma_bounds(:,4), 'b--', time, -sigma_bounds(:,4), 'b--');
title('Y Velocity Error');
xlabel('Time (s)');
ylabel('Error (m/s)');

sgtitle('Estimation Errors with 3-sigma Bounds (UKF)');

%% Problem 6 : optimizations to minimize the information cost %%
%%% Using fmincon function %%%
InitialMeasurement.mean = [2400,-200,8,8.5]';
InitialMeasurement.cov = [100^2*eye(2),zeros(2,2);zeros(2,2),10^2*eye(2)];
params.omega_max = 0.2;
xhat_old = InitialMeasurement.mean;
cov_old = InitialMeasurement.cov;
x0s = [0,0,60*pi/180]';
T_final = 100; delta_T = 1;
disp("Measuring elapsed time for each optimization method");

% Greedy Offline
tic;
[GreedyOfflineSensor, GreedyOfflineTarget,JGOffline] = OpenLoop(x0s, xhat_old, cov_old, ExactTargetPath, T_final, delta_T, 1, 1, params);
elapsed_greedy_off = toc;
disp(['Elapsed time for Greedy Offline: ', num2str(elapsed_greedy_off), ' seconds']);

% Receding Horizon Offline
tic;
[RecedingOfflineSensor, RecedingOfflineTarget,JROffline] = OpenLoop(x0s, xhat_old, cov_old, ExactTargetPath, T_final, delta_T, 10, 5, params);
elapsed_rh_off = toc;
disp(['Elapsed time for Receding Horizon Offline: ', num2str(elapsed_rh_off), ' seconds']);

% Greedy Online
tic;
[GreedyOnlineSensor, GreedyOnlineTarget, JGOnline] = ClosedLoop(x0s, xhat_old, cov_old, ExactTargetPath, T_final, delta_T, 1, 1, params);
elapsed_greedy_on = toc;
disp(['Elapsed time for Greedy Online: ', num2str(elapsed_greedy_on), ' seconds']);

% Receding Horizon Online
tic;
[RecedingOnlineSensor, RecedingOnlineTarget, JRonline] = ClosedLoop(x0s, xhat_old, cov_old, ExactTargetPath, T_final, delta_T, 10, 5, params);
elapsed_rh_on = toc;
disp(['Elapsed time for Receding Horizon Online: ', num2str(elapsed_rh_on), ' seconds']);

% % Full Optimal Offline
% tic;
% [FullOfflineSensor, FullOfflineTarget,JF] = OpenLoop(x0s, xhat_old, cov_old, ExactTargetPath, T_final, delta_T, 100, 1, params);
% elapsed_full_opt = toc;
% disp(['Elapsed time for Full Optimal Offline: ', num2str(elapsed_full_opt), ' seconds']);
%%
figure(4);hold on;

disp("Plotting")
plot(ExactTargetPath(1,1:100),ExactTargetPath(2,1:100),'g','LineWidth',2)

plot(GreedyOfflineSensor(1,:),GreedyOfflineSensor(2,:),'b:','LineWidth',2)
plot(RecedingOfflineSensor(1,:),RecedingOfflineSensor(2,:),'r:','LineWidth',2)

plot(GreedyOfflineTarget(1,:),GreedyOfflineTarget(2,:),'b','LineWidth',2)
plot(RecedingOfflineTarget(1,:),RecedingOfflineTarget(2,:),'r','LineWidth',2)


plot(GreedyOnlineSensor(1,:),GreedyOnlineSensor(2,:),'m:','LineWidth',2)
plot(RecedingOnlineSensor(1,:),RecedingOnlineSensor(2,:),'k:','LineWidth',2)

plot(GreedyOnlineTarget(1,:),GreedyOnlineTarget(2,:),'m','LineWidth',2)
plot(RecedingOnlineTarget(1,:),RecedingOnlineTarget(2,:),'k','LineWidth',2)

% plot(FullOfflineSensor(1,:),FullOfflineSensor(2,:),'cyan:','LineWidth',2)
% plot(FullOfflineTarget(1,:),FullOfflineTarget(2,:),'cyan','LineWidth',2)


hold off;
legend("ExactTargetPath", ...
    "Greedy(Off) sensor","RH(Off) sensor", ...
    "Greedy(Off) Target","RH(Off) Target", ...
    "Greedy(On) sensor","RH(On) sensor", ...
    "Greedy(On) Target","RH(On) Target", ...
    "Full offline Sensor","Full offline Target" ...
    ,'Location','best')
%title("Each path of each optimization of sensor movement : Dashed line (Sensor), Solid line (Target)")
title("RH 5 steps, Each path of each optimization of sensor movement : Dashed line (Sensor), Solid line (Target)")

xlabel("X [m]");
ylabel("Y [m] ")
% =============================
% UKF Performance Comparison Table
% =============================
%%
disp("Computing UKF performance metrics...");

% RMSE 계산
rmse_greedy_off = sqrt(mean(vecnorm(GreedyOfflineTarget - ExactTargetPath(:,1:101), 2, 1).^2));
rmse_rh_off = sqrt(mean(vecnorm(RecedingOfflineTarget - ExactTargetPath(:,1:101), 2, 1).^2));
rmse_greedy_on = sqrt(mean(vecnorm(GreedyOnlineTarget - ExactTargetPath(:,1:101), 2, 1).^2));
rmse_rh_on = sqrt(mean(vecnorm(RecedingOnlineTarget - ExactTargetPath(:,1:101), 2, 1).^2));
% rmse_full_opt = sqrt(mean(vecnorm(FullOfflineTarget - ExactTargetPath(:,1:101), 2, 1).^2));

% 정보 비용
J_info_greedy_off = sum(log(det(cov_old)));
J_info_rh_off = sum(log(det(cov_old)));
J_info_greedy_on = sum(log(det(cov_old)));
J_info_rh_on = sum(log(det(cov_old)));
% J_info_full_opt = sum(log(det(cov_old)));

% =============================
% 📌 Display Performance Table
% =============================

disp("Optimization Performance Summary:");
PerformanceTable = table(["Greedy Offline"; "Receding Horizon Offline"; "Greedy Online"; "Receding Horizon Online"; "Full Optimal"], ...
    [elapsed_greedy_off; elapsed_rh_off; elapsed_greedy_on; elapsed_rh_on; elapsed_full_opt], ...
    [rmse_greedy_off; rmse_rh_off; rmse_greedy_on; rmse_rh_on; rmse_full_opt], ...
    [J_info_greedy_off; J_info_rh_off; J_info_greedy_on; J_info_rh_on; J_info_full_opt], ...
    'VariableNames', {'Optimization Method', 'Elapsed Time (s)', 'RMSE', 'J_info'});

disp(PerformanceTable);

disp("UKF Performance Comparison (RMSE & Information Cost):");
PerformanceTable = table(["Greedy Offline"; "Greedy Online"; "RH Offline"; "RH Online"; "Full Optimal"], ...
    [rmse_greedy_off; rmse_greedy_on; rmse_rh_off; rmse_rh_on; rmse_full_opt], ...
    [JGOffline; JGOnline; JROffline; JRonline; JF], ...
    'VariableNames', {'Optimization Method', 'RMSE', 'J_info'});

disp(PerformanceTable);