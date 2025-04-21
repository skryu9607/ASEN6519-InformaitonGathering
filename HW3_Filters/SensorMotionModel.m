function sensor_state_next = SensorMotionModel(sensor_state_curr,sensor_control,delta_t)
%%% Inputs : the current state of the sensor and the control input of the
%%% sensor and the time interval, delta_t.
%%% Outputs : the sensor state of the next time step
%%% SeungKeol Ryu
%%% Edited : March 3rd
xki = sensor_state_curr(1);yki = sensor_state_curr(2);psiki = sensor_state_curr(3);
sensorspeed = sensor_control(1);turnrate = sensor_control(2);
if ~isequal(size(sensor_state_curr), [3,1]) || ~isequal(size(sensor_control), [2,1])
    error("Dimension error: sensor_state_curr must be 3x1 and sensor_control must be 2x1 column vectors.");
end
xk1i = xki + sensorspeed * cos(psiki) * delta_t;
yk1i = yki + sensorspeed * sin(psiki) * delta_t;
psi1ki = psiki + turnrate * delta_t;

sensor_state_next = [xk1i,yk1i,psi1ki]';

end

