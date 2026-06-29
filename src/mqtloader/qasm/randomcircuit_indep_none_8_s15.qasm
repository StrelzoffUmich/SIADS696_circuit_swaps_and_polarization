OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; sdg q0; sx q0; s q0; s q1; cx q0,q1; ry(0.5*param0) q0; ry((-0.5)*param0) q1; cx q0,q1; sdg q1; sdg q0; sxdg q0; s q0; rz(param1) q1; }
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
gate rcccx q0,q1,q2,q3 { h q3; t q3; cx q2,q3; tdg q3; h q3; cx q0,q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; h q3; t q3; cx q2,q3; tdg q3; h q3; }
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
qreg q[8];
creg meas[8];
cry(0.1974700954206846) q[0],q[5];
rzx(2.586348858544183) q[6],q[2];
iswap q[4],q[1];
cry(1.0507201913780568) q[7],q[3];
ccx q[3],q[0],q[5];
cry(5.359742365830875) q[4],q[1];
ryy(5.438830018694474) q[6],q[2];
u1(2.9096484865663093) q[7];
iswap q[5],q[2];
xx_minus_yy(3.3095154677832346,0.3574513307731107) q[1],q[6];
t q[7];
crz(3.065159831604048) q[0],q[4];
h q[3];
sx q[0];
cy q[2],q[1];
cu(1.5794139055909173,2.701683685794904,3.253112466982678,1.5954302464557133) q[3],q[4];
xx_minus_yy(3.589440048510018,3.746511914285776) q[5],q[7];
iswap q[6],q[3];
ryy(4.407402247615672) q[2],q[1];
csdg q[4],q[0];
csdg q[7],q[5];
rcccx q[1],q[4],q[6],q[0];
cz q[3],q[5];
iswap q[2],q[7];
z q[2];
ch q[5],q[0];
u(2.7113250923578405,4.64405891375159,4.045559487227495) q[7];
c3sqrtx q[4],q[6],q[3],q[1];
rzz(2.1639724367498183) q[2],q[0];
cp(0.16772232804260773) q[3],q[1];
dcx q[7],q[5];
t q[4];
rxx(3.09319987349959) q[4],q[1];
cx q[5],q[3];
swap q[2],q[6];
x q[7];
s q[7];
rxx(3.057858602799537) q[3],q[5];
xx_minus_yy(1.3978252929384303,5.405728516506422) q[2],q[0];
cz q[6],q[4];
c3sqrtx q[6],q[3],q[4],q[1];
swap q[7],q[5];
cry(0.6254058719535827) q[0],q[2];
ccx q[2],q[7],q[0];
crx(6.134037678883007) q[3],q[5];
ccz q[6],q[1],q[4];
cswap q[6],q[1],q[4];
u3(1.4359168395972879,2.1459141275298173,1.4831959858449213) q[0];
tdg q[3];
cswap q[5],q[7],q[2];
sxdg q[6];
xx_plus_yy(1.1101683672917966,0.11260466535371279) q[1],q[2];
rxx(4.492294557525396) q[3],q[7];
ccx q[5],q[4],q[0];
r(0.585341866227886,0.511354469395983) q[1];
iswap q[7],q[4];
ryy(2.8625184260395167) q[3],q[5];
rccx q[6],q[0],q[2];
cs q[5],q[0];
cry(2.703034022210769) q[1],q[3];
csdg q[2],q[4];
cp(0.8273518314751745) q[7],q[6];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];