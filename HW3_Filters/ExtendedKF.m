function [xhat_11,Pk1k1] = ExtendedKF(target_mean,target_cov, sensor_state,params,z_from_groundtruth)
%EXTENDEDKF Summary of this function goes here

% xhat_1 : xhat_(k+1)| k 
% xhat_11 : xhat_(k+1) | k+1

% initial condition
xhat = target_mean;
Pkk = target_cov;
delta_t = params.delta_t;

% Prediction step
Fk = [1,0,delta_t,0; 0,1,0,delta_t ;0,0,1,0 ;0,0,0,1];
GammaK = [delta_t^2/2,0;0,delta_t^2/2;delta_t,0;0,delta_t];
Qk = params.Q;

xhat_1 = Fk * xhat;
Pk1k = Fk*Pkk*Fk' + GammaK*Qk*GammaK';

% Update step (we know the location of the sensor)
H = RangeBearingSensorJacobian(xhat,sensor_state);
V = eye(2);
K = Pk1k * H' * inv(H * Pk1k * H' + V * params.R * V');

z_from_predictedstate = RangeBearingSensor(xhat_1,sensor_state,[0,0]');

sensor_noise = mvnrnd([0,0]',params.Q)';
z_from_groundtruth = RangeBearingSensor(z_from_groundtruth,sensor_state,sensor_noise);
error = z_from_groundtruth - z_from_predictedstate;

xhat_11 = xhat_1 + K * (error);

Pk1k1 = (eye(4) - K * H) * Pk1k;

end

