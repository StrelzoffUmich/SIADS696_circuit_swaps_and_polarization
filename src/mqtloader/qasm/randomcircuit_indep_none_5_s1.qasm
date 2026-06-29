OPENQASM 2.0;
include "qelib1.inc";
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
qreg q[5];
creg meas[5];
cswap q[1],q[3],q[0];
cs q[2],q[4];
rccx q[2],q[3],q[0];
y q[1];
ccx q[1],q[4],q[2];
cry(4.880043932370009) q[1],q[4];
cswap q[3],q[0],q[2];
cswap q[2],q[1],q[0];
id q[4];
cswap q[1],q[2],q[4];
rccx q[2],q[1],q[3];
swap q[0],q[4];
id q[3];
ccz q[0],q[2],q[4];
c3sqrtx q[1],q[3],q[0],q[2];
z q[4];
ccz q[2],q[1],q[3];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];