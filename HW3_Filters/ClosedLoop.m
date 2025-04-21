function [sensor_trajectory, target_estimates,infocost] = ClosedLoop(sensor_state, target_state, P, ExactTargetPath, T_final, delta_T, Np, Nc, params)
% Initialize trajectory storage
    sensor_trajectory = [sensor_state];
    target_estimates = [target_state];

    % Define optimization options
    options = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'off');
    lb = -params.omega_max * ones(Np, 1);
    ub = params.omega_max * ones(Np, 1);
    J = 0;
    for k = 1:(T_final / delta_T)
        % Use fmincon to find the optimal control sequence
        seq0 = zeros(Np,1); % Initial guess
        [opt_seq, infocost] = fmincon(@(x) InfoCost(x, sensor_state, target_state, P, params), ...
                               seq0, [], [], [], [], lb, ub, [], options);
        J = J + infocost;
        % Apply only Nc steps from the optimized sequence
        for j = 1:Nc
            if j > length(opt_seq)
                break;
            end
            
            % Update sensor state based on the selected control input
            sensor_control = [20; opt_seq(j)]; % Assume constant speed 20m/s
            sensor_state = SensorMotionModel(sensor_state, sensor_control, delta_T);
            sensor_trajectory = [sensor_trajectory, sensor_state];

            % Run Sigma Point Filter after each step to refine target estimate
            [target_state, P] = UnscentedKF(target_state,P, sensor_trajectory(:,k), params, ExactTargetPath(:,k));
            target_estimates = [target_estimates, target_state];
        end
    end
end
