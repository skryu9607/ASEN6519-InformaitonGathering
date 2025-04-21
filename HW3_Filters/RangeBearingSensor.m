function sensor_meas = RangeBearingSensor(target_state,sensor_state,sensor_noise)
%RANGEBEARINGSENSOR Summary of this function goes here
%   Detailed explanation goes here
xk = target_state(1);yk = target_state(2);
xki = sensor_state(1);yki = sensor_state(2);

range = sqrt((xk - xki)^2 + (yk - yki)^2) + sensor_noise(1);
bearing = atan2(yk - yki,xk - xki) + sensor_noise(2);

sensor_meas = [range,bearing]';


end

