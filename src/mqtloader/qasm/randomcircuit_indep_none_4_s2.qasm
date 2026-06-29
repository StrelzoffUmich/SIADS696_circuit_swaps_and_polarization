OPENQASM 2.0;
include "qelib1.inc";
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
qreg q[4];
creg meas[4];
cry(2.940122020423439) q[3],q[2];
x q[0];
crx(3.4777264301172575) q[0],q[2];
ch q[3],q[1];
cswap q[0],q[2],q[3];
u1(3.848699841633905) q[1];
xx_plus_yy(3.1219478687923115,1.555182121389826) q[1],q[3];
rccx q[0],q[2],q[3];
y q[1];
tdg q[1];
cp(5.322801980939597) q[0],q[3];
ry(2.269889027609814) q[3];
iswap q[0],q[2];
h q[0];
cz q[3],q[2];
barrier q[0],q[1],q[2],q[3];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];