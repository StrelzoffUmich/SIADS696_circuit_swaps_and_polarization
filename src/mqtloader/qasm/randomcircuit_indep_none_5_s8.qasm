OPENQASM 2.0;
include "qelib1.inc";
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
qreg q[5];
creg meas[5];
cs q[4],q[1];
rzx(2.1124353781126395) q[2],q[3];
id q[0];
cu(5.080874631897661,2.4280995414969886,0.21425977507603577,4.716682593396704) q[2],q[1];
s q[4];
cp(4.29633354368644) q[0],q[3];
x q[2];
cu(5.79495243820984,3.319648984678104,4.429070412541983,0.029460191658368468) q[0],q[3];
cx q[1],q[4];
sx q[1];
xx_plus_yy(5.838250179191951,2.2110454694939286) q[0],q[3];
rzx(2.505236639618995) q[2],q[4];
z q[0];
y q[1];
r(5.3101303395347665,2.58460822743691) q[3];
z q[2];
z q[4];
cu1(6.0134053470929185) q[1],q[3];
ch q[2],q[4];
cs q[3],q[0];
iswap q[2],q[1];
s q[4];
swap q[2],q[1];
ry(3.813519017873472) q[4];
r(2.001730314533563,2.487624099372773) q[3];
u2(0.8285681096285292,5.38529374764059) q[0];
ryy(2.740267167634765) q[4],q[1];
ch q[0],q[2];
tdg q[4];
sx q[1];
rz(1.867957682821644) q[3];
ry(5.44517576970119) q[2];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];