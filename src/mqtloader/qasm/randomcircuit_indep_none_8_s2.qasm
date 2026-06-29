OPENQASM 2.0;
include "qelib1.inc";
gate rcccx q0,q1,q2,q3 { h q3; t q3; cx q2,q3; tdg q3; h q3; cx q0,q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; h q3; t q3; cx q2,q3; tdg q3; h q3; }
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; sdg q0; sx q0; s q0; s q1; cx q0,q1; ry(0.5*param0) q0; ry((-0.5)*param0) q1; cx q0,q1; sdg q1; sdg q0; sxdg q0; s q0; rz(param1) q1; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
qreg q[8];
creg meas[8];
c3sqrtx q[2],q[3],q[5],q[7];
c3sqrtx q[0],q[4],q[1],q[6];
rcccx q[5],q[7],q[3],q[2];
c3sqrtx q[4],q[1],q[0],q[6];
crx(3.2457565193355435) q[3],q[2];
cp(3.7287155914048777) q[1],q[4];
cs q[5],q[7];
tdg q[0];
c3sqrtx q[2],q[5],q[3],q[7];
cswap q[6],q[0],q[1];
c3sqrtx q[0],q[1],q[6],q[4];
c3sqrtx q[7],q[5],q[3],q[2];
rccx q[3],q[5],q[4];
c3sqrtx q[0],q[2],q[7],q[6];
sdg q[1];
rcccx q[1],q[4],q[7],q[5];
rcccx q[6],q[2],q[3],q[0];
rccx q[4],q[7],q[1];
sx q[6];
cy q[3],q[2];
csdg q[0],q[5];
c3sqrtx q[5],q[6],q[4],q[3];
u1(0.33132582847963304) q[2];
xx_minus_yy(6.136388065108867,3.8618212994884704) q[7],q[1];
ccz q[6],q[7],q[1];
rcccx q[3],q[0],q[2],q[5];
c3sqrtx q[6],q[0],q[4],q[3];
xx_plus_yy(1.2881895320595143,1.619922526778703) q[7],q[2];
rcccx q[4],q[5],q[0],q[1];
rcccx q[6],q[2],q[3],q[7];
cz q[4],q[3];
ccz q[7],q[5],q[1];
ccx q[0],q[6],q[2];
cz q[7],q[3];
rccx q[6],q[2],q[5];
cswap q[1],q[4],q[0];
rcccx q[4],q[7],q[5],q[0];
c3sqrtx q[1],q[2],q[6],q[3];
ccx q[4],q[1],q[6];
rccx q[5],q[7],q[3];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];