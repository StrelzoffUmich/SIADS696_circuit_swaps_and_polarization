OPENQASM 2.0;
include "qelib1.inc";
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
qreg q[6];
creg meas[6];
cswap q[0],q[3],q[2];
ccz q[5],q[1],q[4];
crx(2.4241904553553666) q[5],q[0];
cu1(2.6353175098260864) q[3],q[1];
s q[2];
u3(1.675213695323451,6.198411878136848,3.45028777289559) q[4];
cs q[4],q[5];
cu1(6.283153012882725) q[0],q[1];
z q[2];
r(2.6407420362728518,5.06553005240474) q[3];
ry(6.0102361813189225) q[2];
z q[5];
crx(3.5142971764588995) q[1],q[4];
ryy(6.086266334934358) q[0],q[3];
sx q[3];
ryy(2.6049181115600035) q[5],q[4];
r(1.0381180536392487,1.0330187283752796) q[1];
ryy(5.143240999787738) q[2],q[0];
csdg q[0],q[1];
csdg q[2],q[3];
u(1.0099288913157531,1.8730830485757637,0.27004698948568484) q[5];
cz q[2],q[4];
ccz q[0],q[3],q[1];
t q[5];
rzx(4.6999311942040585) q[2],q[5];
p(4.198230582392688) q[1];
rzz(2.4600069893807497) q[3],q[4];
u(3.3673980233871923,1.2537017907693773,1.7839766686152574) q[0];
csdg q[2],q[3];
ccx q[5],q[1],q[0];
u1(0.6058753907790871) q[4];
cy q[2],q[4];
sxdg q[5];
ccx q[1],q[0],q[3];
u(4.1212782051549,5.226802666147904,0.8851181532802196) q[3];
ccx q[1],q[4],q[5];
tdg q[2];
h q[5];
xx_plus_yy(0.5723286596405267,2.795998697567813) q[4],q[1];
ccz q[2],q[0],q[3];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];