function [xhat_11,Pk1k1] = ExtendedKFcost(target_mean, target_cov, sensor_state, params)
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

Pk1k1 = (eye(4) - K * H) * Pk1k;
xhat_11 = xhat_1;
end

