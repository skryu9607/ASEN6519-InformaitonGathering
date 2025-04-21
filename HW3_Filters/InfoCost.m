function J = InfoCost(control_seq,sensor_state_curr,target_state_est,target_state_cov,params)
Np = length(control_seq);

% Initial values
sensor_state = sensor_state_curr;
target_state = target_state_est;
P = target_state_cov;
delta_t = 1;
F = [1 0 delta_t 0;0 1 0 delta_t;0 0 1 0;0 0 0 1];
Q = params.Q;
% using EKF without noise,  
% Do not update the state measuremnt, just update the covariance. 
J = log(det(P));
for i = 1:Np
    % Update sensor state using kinematic unicycle model
    sensor_control = [20; control_seq(i)];  % Assume constant speed 20m/s
    sensor_state = SensorMotionModel(sensor_state, sensor_control, delta_t);
    
    % Predict target state (without process noise)
    target_state = F * target_state;
    GammaK = [delta_t^2/2,0;0,delta_t^2/2;delta_t,0;0,delta_t];
    % Predict covariance matrix
    P = F * P * F' + GammaK * Q * GammaK';
    
    % Compute measurement Jacobian Hk
    H = RangeBearingSensorJacobian(target_state, sensor_state);
    
    V = eye(2);
    K = P * H' * inv(H * P * H' + V * params.R * V');
    P = (eye(4) - K * H) * P;
    
    % Accumulate information cost
    J = J + log(det(P));
end
J;
end