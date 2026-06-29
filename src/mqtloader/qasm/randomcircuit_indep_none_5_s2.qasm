OPENQASM 2.0;
include "qelib1.inc";
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate rcccx q0,q1,q2,q3 { h q3; t q3; cx q2,q3; tdg q3; h q3; cx q0,q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; h q3; t q3; cx q2,q3; tdg q3; h q3; }
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
qreg q[5];
creg meas[5];
c3sqrtx q[3],q[0],q[1],q[4];
r(3.5328193511135773,0.9428690079622023) q[2];
ccz q[0],q[4],q[1];
rzx(1.1765425947969412) q[2],q[3];
rcccx q[2],q[3],q[4],q[1];
cs q[0],q[2];
cswap q[1],q[3],q[4];
rcccx q[1],q[4],q[2],q[3];
cswap q[2],q[1],q[4];
rcccx q[0],q[1],q[2],q[3];
rzz(5.233874286313884) q[2],q[1];
rzx(0.6246184755646657) q[3],q[0];
c3sqrtx q[3],q[2],q[4],q[0];
s q[1];
rcccx q[3],q[4],q[1],q[0];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];