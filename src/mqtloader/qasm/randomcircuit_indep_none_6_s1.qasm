OPENQASM 2.0;
include "qelib1.inc";
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; sdg q0; sx q0; s q0; s q1; cx q0,q1; ry(0.5*param0) q0; ry((-0.5)*param0) q1; cx q0,q1; sdg q1; sdg q0; sxdg q0; s q0; rz(param1) q1; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate rcccx q0,q1,q2,q3 { h q3; t q3; cx q2,q3; tdg q3; h q3; cx q0,q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; h q3; t q3; cx q2,q3; tdg q3; h q3; }
qreg q[6];
creg meas[6];
cswap q[1],q[3],q[5];
rccx q[0],q[2],q[4];
ccx q[4],q[5],q[2];
cswap q[3],q[0],q[1];
cswap q[1],q[4],q[2];
xx_minus_yy(6.094221105379693,3.2425545542113827) q[3],q[5];
rz(0.7280051138837079) q[0];
rccx q[1],q[3],q[4];
h q[0];
rccx q[3],q[1],q[5];
cswap q[4],q[2],q[0];
rccx q[1],q[3],q[5];
swap q[2],q[4];
sdg q[0];
id q[4];
ccz q[3],q[5],q[0];
cswap q[5],q[4],q[1];
rccx q[2],q[0],q[3];
ccz q[1],q[2],q[5];
ccx q[3],q[0],q[4];
ccx q[1],q[4],q[0];
ccx q[2],q[5],q[3];
ccz q[5],q[2],q[3];
ccz q[1],q[4],q[0];
rcccx q[4],q[2],q[1],q[3];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];