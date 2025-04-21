function [xhat_11,Pk1k1] = UnscentedKF(target_mean,target_cov, sensor_state, params, z_from_groundtruth)
%UNSCENTEDKF Summary of this function goes here

xhat = target_mean; 
% exp of wk =  zeros(1,2)
Pkk = target_cov;
% Define augmented state and covariance matrices
augx = [xhat',zeros(2,1)']'; % 6 * 1 column vector
augP = [Pkk, zeros(4,2);zeros(2,4),params.Q]; % 6 * 6 square matrix
eps = 1e-6;
delta_t = params.delta_t;

%%% Prediction step %%%

Lp = params.Lp;n=4;q = 2;
% Sigma points
sigmapoints = zeros(Lp,1+2*Lp);
sigmapoints(:,1) = augx;
weightX = zeros(n,1+2*Lp);

weightX(:,1) = ConstantVelocityTargetModel(augx(1:n),augx(n+1:n+q),delta_t);
% Derive the square root of matrix.
[U, S, ~] = svd((Lp + params.lambda) * augP); 
sqrtAugP = U * sqrt(S);

for i = 1 : 2*Lp
    if i < Lp+1
        sigmapoints(:,i+1) = augx + sqrtAugP(:,i);
    else
        sigmapoints(:,i+1) = augx - sqrtAugP(:,i - (Lp));
    end
    weightX(:,i+1) = ConstantVelocityTargetModel(sigmapoints(1:n,i+1), sigmapoints(n+1:n+q,i+1),delta_t);
end

xhat_1 = params.lambda/(Lp + params.lambda) * weightX(:,1);

for i = 1 : 2 * Lp
    xhat_1 = xhat_1 + 1/(2*(Lp + params.lambda)) * weightX(:,i+1);
end


weightc0 = params.lambda/(Lp + params.lambda) + (1 - params.alpha^2 + params.beta);

Pk1k = weightc0 * (weightX(:,1) - xhat_1) * (weightX(:,1)- xhat_1)';
for i = 1 : 2 * Lp
    Pk1k = Pk1k + 1/(2*(Lp + params.lambda)) * (weightX(:,i+1) - xhat_1) * (weightX(:,i+1) - xhat_1)';
end

% Force Symmetry & Regularization
Pk1k = (Pk1k + Pk1k') / 2 + eps * eye(size((Pk1k+Pk1k')/2));

%%% Update step %%%

r = 2;Lu = n + r;
augXnew = [xhat_1', [0;0]']';

augPnew = [Pk1k, zeros(4,2) ; zeros(2,4), params.R];

% Sigma points
sigmapointsnew = zeros(Lu,1+2*Lu);
sigmapointsnew(:,1) = augXnew;
gammas = zeros(2,1+2*Lu);

gammas(:,1) = RangeBearingSensor(augXnew(1:n),sensor_state,augXnew(n+1:n+r));
% sqrtAugPnew = chol((Lu + params.lambda) * augPnew, 'lower'); % Cholesky decomposition
[U, S, ~] = svd((Lp + params.lambda) * augPnew); 
sqrtAugPnew = U * sqrt(S); 

for i = 1 : 2*Lu
    if i < Lu+1
        sigmapointsnew(:,i+1) = augXnew + sqrtAugPnew(:,i);
    else
        sigmapointsnew(:,i+1) = augXnew - sqrtAugPnew(:,i - (Lu));
    end
    gammas(:,i+1) = RangeBearingSensor(sigmapointsnew(1:n,i+1), sensor_state,sigmapointsnew(n+1:n+r,i+1));   
end

z_predicted = params.lambda/(Lu + params.lambda) * gammas(:,1);

for i = 1 : 2*Lu
    z_predicted = z_predicted + 1/(2*(Lu + params.lambda)) * gammas(:,i+1);
end

Pzz = weightc0 * (gammas(:,1) - z_predicted) * (gammas(:,1) - z_predicted)';
Pxz = weightc0 * (sigmapointsnew(1:n,1) - xhat_1) * (gammas(:,1) - z_predicted)';

for i = 1 : 2 * Lu
    Pzz = Pzz + 1/(2*(Lu + params.lambda)) * (gammas(:,i + 1) - z_predicted) * (gammas(:,i + 1) - z_predicted)';
    Pxz = Pxz + 1/(2*(Lu + params.lambda)) * (sigmapointsnew(1:n,i + 1) - xhat_1) * (gammas(:,i + 1) - z_predicted)';
end
% Ensure Pzz is Positive Definite
Pzz = (Pzz + Pzz') / 2 + eps * eye(size((Pzz+Pzz')/2));
K = Pxz / Pzz;

sensor_noise = mvnrnd([0,0]',params.Q)';
z_from_groundtruth = RangeBearingSensor(z_from_groundtruth,sensor_state,sensor_noise);

% Getting outputs
xhat_11 = xhat_1 + K * (z_from_groundtruth - z_predicted);
Pk1k1 = Pk1k - K * Pzz * K';
Pk1k1 = (Pk1k1 + Pk1k1')/2 + eps*(eye(size((Pk1k1+Pk1k1')/2)));
% [V, D] = eig(Pk1k1);
% D = max(D, 1e-10 * eye(size(D))); % Ensure minimum eigenvalues
% Pk1k1 = V * D * V';

end



