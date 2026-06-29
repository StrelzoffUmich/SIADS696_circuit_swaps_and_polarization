OPENQASM 2.0;
include "qelib1.inc";
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; sdg q0; sx q0; s q0; s q1; cx q0,q1; ry(0.5*param0) q0; ry((-0.5)*param0) q1; cx q0,q1; sdg q1; sdg q0; sxdg q0; s q0; rz(param1) q1; }
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate rcccx q0,q1,q2,q3 { h q3; t q3; cx q2,q3; tdg q3; h q3; cx q0,q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; h q3; t q3; cx q2,q3; tdg q3; h q3; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
qreg q[6];
creg meas[6];
r(1.1896583411569202,0.3563202172462531) q[0];
z q[1];
xx_minus_yy(2.041763619419775,1.7490221364800458) q[2],q[5];
c3sqrtx q[4],q[5],q[3],q[0];
cs q[1],q[2];
cy q[2],q[1];
rcccx q[3],q[5],q[4],q[0];
h q[0];
iswap q[5],q[2];
u3(3.7029977474574,4.651897830403353,2.33211259936547) q[3];
rx(6.100027862439213) q[4];
cswap q[5],q[1],q[2];
cu(4.201196447767275,4.400549713290081,5.235503102986703,4.233686841774406) q[0],q[3];
sx q[4];
cx q[0],q[1];
dcx q[2],q[5];
c3sqrtx q[1],q[0],q[3],q[4];
cx q[5],q[2];
t q[0];
ry(1.9834554863361864) q[3];
ryy(2.467657301292213) q[1],q[2];
s q[5];
s q[1];
rzz(3.8194401645045537) q[2],q[5];
rccx q[0],q[4],q[3];
ecr q[4],q[3];
cu3(3.6147040829771084,4.878249562483391,3.3418638739229753) q[2],q[1];
u1(1.5424990013753146) q[5];
u(0.5824106678205935,5.803970985343431,5.362210836141355) q[2];
x q[1];
ccz q[5],q[3],q[0];
u1(2.194815975202933) q[2];
sx q[1];
xx_minus_yy(4.2383422890732,5.763165490840785) q[0],q[4];
sdg q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];