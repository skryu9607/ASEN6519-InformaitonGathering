function [sensor_trajectory, target_estimates,J] = OpenLoop(sensor_state, target_state, P, ExactTargetPath, T_final, delta_T, Np, Nc, params)
% Initialize trajectory storage
sensor_trajectory = [sensor_state];

% Define optimization options
options = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'off');
lb = -params.omega_max * ones(Np, 1);
ub = params.omega_max * ones(Np, 1);
J = 0;
for k = 1:(T_final / delta_T)
    % Use fmincon to find the optimal control sequence
    seq0 = zeros(Np,1); % Initial guess
    [opt_seq, cost] = fmincon(@(x) InfoCost(x, sensor_state, target_state, P, params), seq0, [], [], [], [], lb, ub, [], options);
    J = J + cost;
    % Apply only Nc steps from the optimized sequence
    for j = 1:Nc
        if j > length(opt_seq)
            break;
        end
        % Update sensor state based on the selected control input
        sensor_state = SensorMotionModel(sensor_state, [20,opt_seq(j)]', delta_T);
        sensor_trajectory = [sensor_trajectory, sensor_state];
    end
end
target_estimates = [target_state];
for k = 1:(T_final / delta_T)
% After the trajectory is completed, apply the filter for final estimation
    [xhat_11,~] = UnscentedKF(target_state,P, sensor_trajectory(:,k), params, ExactTargetPath(:,k));
    target_estimates = [target_estimates,xhat_11];
end

end
