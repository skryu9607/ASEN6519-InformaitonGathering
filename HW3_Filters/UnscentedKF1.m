function [xhat_11, Pk1k1] = UnscentedKF1(target_mean, target_cov, sensor_state, params, z_from_groundtruth)
    % Initialize State and Covariance
    xhat = target_mean; 
    Pkk = target_cov;

    % Define augmented state and covariance matrices
    augx = [xhat; zeros(2,1)]; % Augmented state: [state; process noise]
    augP = [Pkk, zeros(4,2); zeros(2,4), params.Q]; % Augmented covariance

    eps = 1e-6; % Regularization constant
    delta_t = params.delta_t;
    Lp = params.Lp; % Number of sigma points
    n = 4; % State dimension
    q = 2; % Process noise dimension

    % Compute Sigma Points
    sigmapoints = ComputeSigmaPoints(augx, augP, Lp, params.lambda);

    % Propagate Sigma Points through Motion Model
    weightX = zeros(n, size(sigmapoints,2));
    for i = 1:size(sigmapoints,2)
        weightX(:,i) = ConstantVelocityTargetModel(sigmapoints(1:n,i), sigmapoints(n+1:n+q,i), delta_t);
    end

    % Compute Mean Prediction
    xhat_1 = ComputeWeightedMean(weightX, params.lambda, Lp);

    % Compute Predicted Covariance
    Pk1k = ComputeWeightedCovariance(weightX, xhat_1, params.lambda, Lp, params.alpha, params.beta);
    Pk1k = RegularizeCovariance(Pk1k, eps);

    % Measurement Update
    r = 2; % Measurement dimension
    Lu = n + r;
    augXnew = [xhat_1; zeros(2,1)]; % Augmented state
    augPnew = [Pk1k, zeros(4,2); zeros(2,4), params.R]; % Augmented covariance

    % Compute Sigma Points for Measurement
    sigmapointsnew = ComputeSigmaPoints(augXnew, augPnew, Lu, params.lambda);
    gammas = ComputeMeasurementModel(sigmapointsnew, sensor_state, n, r);

    % Compute Measurement Prediction
    z_predicted = ComputeWeightedMean(gammas, params.lambda, Lu);

    % Compute Measurement Covariance & Cross-Covariance
    [Pzz, Pxz] = ComputeMeasurementCovariances(gammas, z_predicted, sigmapointsnew, xhat_1, params.lambda, Lu, params.alpha, params.beta);
    Pzz = RegularizeCovariance(Pzz, eps);

    % Compute Kalman Gain
    K = Pxz / Pzz;

    % Measurement Update
    sensor_noise = mvnrnd([0,0]', params.Q)'; % Simulated measurement noise
    z_from_groundtruth = RangeBearingSensor(z_from_groundtruth, sensor_state, sensor_noise);
    xhat_11 = xhat_1 + K * (z_from_groundtruth - z_predicted);
    Pk1k1 = Pk1k - K * Pzz * K';
    Pk1k1 = RegularizeCovariance(Pk1k1, eps);
end

%% ----------------- Helper Functions -----------------

% Generate Sigma Points
function sigmapoints = ComputeSigmaPoints(mean_x, cov_x, L, lambda)
    [U, S, ~] = svd((L + lambda) * cov_x); % Square root using SVD
    sqrtCov = U * sqrt(S);

    sigmapoints = zeros(size(mean_x,1), 1 + 2*L);
    sigmapoints(:,1) = mean_x;

    for i = 1:L
        sigmapoints(:,i+1) = mean_x + sqrtCov(:,i);
        sigmapoints(:,i+L+1) = mean_x - sqrtCov(:,i);
    end
end

% Compute Weighted Mean
function mean_x = ComputeWeightedMean(sigmapoints, lambda, L)
    mean_x = lambda / (L + lambda) * sigmapoints(:,1);
    for i = 2:2*L+1
        mean_x = mean_x + 1/(2*(L + lambda)) * sigmapoints(:,i);
    end
end

% Compute Weighted Covariance
function cov_x = ComputeWeightedCovariance(sigmapoints, mean_x, lambda, L, alpha, beta)
    weightc0 = lambda/(L + lambda) + (1 - alpha^2 + beta);
    cov_x = weightc0 * (sigmapoints(:,1) - mean_x) * (sigmapoints(:,1) - mean_x)';

    for i = 2:2*L+1
        cov_x = cov_x + 1/(2*(L + lambda)) * (sigmapoints(:,i) - mean_x) * (sigmapoints(:,i) - mean_x)';
    end
end

% Regularize Covariance Matrix
function P = RegularizeCovariance(P, eps)
    P = (P + P') / 2; % Force symmetry
    P = P + eps * eye(size(P)); % Add small positive value for stability
end

% Compute Measurement Model
function gammas = ComputeMeasurementModel(sigmapoints, sensor_state, n, r)
    gammas = zeros(r, size(sigmapoints,2));
    for i = 1:size(sigmapoints,2)
        gammas(:,i) = RangeBearingSensor(sigmapoints(1:n,i), sensor_state, sigmapoints(n+1:n+r,i));
    end
end

% Compute Measurement Covariance & Cross-Covariance
function [Pzz, Pxz] = ComputeMeasurementCovariances(gammas, z_predicted, sigmapoints, xhat_1, lambda, Lu, alpha, beta)
    weightc0 = lambda / (Lu + lambda) + (1 - alpha^2 + beta);

    Pzz = weightc0 * (gammas(:,1) - z_predicted) * (gammas(:,1) - z_predicted)';
    Pxz = weightc0 * (sigmapoints(1:size(xhat_1,1),1) - xhat_1) * (gammas(:,1) - z_predicted)';

    for i = 2:2*Lu+1
        Pzz = Pzz + 1/(2*(Lu + lambda)) * (gammas(:,i) - z_predicted) * (gammas(:,i) - z_predicted)';
        Pxz = Pxz + 1/(2*(Lu + lambda)) * (sigmapoints(1:size(xhat_1,1),i) - xhat_1) * (gammas(:,i) - z_predicted)';
    end
end
