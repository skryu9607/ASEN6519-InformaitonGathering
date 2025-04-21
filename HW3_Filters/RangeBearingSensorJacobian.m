function H = RangeBearingSensorJacobian(target_state,sensor_state)
% The order : xk,yk,xki,yki
xk = target_state(1);yk = target_state(2);
xki = sensor_state(1);yki = sensor_state(2);
rki = sqrt((xk - xki)^2 + (yk - yki)^2);
rki_partial = 1/rki * [xk-xki,yk-yki,-(xk-xki),-(yk-yki)];
betaki_partial = 1/rki^2 * [-(yk-yki),xk-xki,yk-yki,-(xk-xki)];

H = [rki_partial;betaki_partial];

end

