function target_state_next = ConstantVelocityTargetModel(target_state_curr, process_noise,delta_t)
%CONSTANTVELOCITYTARGETMODEL Summary of this function goes here
%   Detailed explanation goes here

Fk = [1,0,delta_t,0;0,1,0,delta_t;0,0,1,0;0,0,0,1];
Gammak = [delta_t^2/2,0;0,delta_t^2/2;delta_t,0;0,delta_t];

target_state_next = Fk * target_state_curr + Gammak * process_noise;


end

