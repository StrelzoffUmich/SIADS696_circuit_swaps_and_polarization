OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; sdg q0; sx q0; s q0; s q1; cx q0,q1; ry(0.5*param0) q0; ry((-0.5)*param0) q1; cx q0,q1; sdg q1; sdg q0; sxdg q0; s q0; rz(param1) q1; }
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
qreg q[8];
creg meas[8];
crx(0.6713663549993827) q[2],q[5];
cx q[3],q[1];
ryy(1.6207788068011701) q[7],q[6];
sdg q[4];
dcx q[6],q[7];
u2(4.079297547670931,5.4761468829910225) q[1];
ryy(2.5641582983718583) q[3],q[0];
ccz q[4],q[5],q[2];
u3(2.035092632636993,1.0696888720905462,4.903289751060482) q[2];
sxdg q[4];
xx_plus_yy(5.746127449698812,4.579633957334016) q[5],q[7];
sx q[3];
ccz q[1],q[6],q[0];
swap q[5],q[3];
dcx q[1],q[2];
z q[6];
ccz q[0],q[7],q[4];
csx q[1],q[2];
rzz(2.541377787873183) q[6],q[4];
rxx(4.7965972589329855) q[7],q[5];
csx q[0],q[3];
h q[2];
y q[3];
u2(5.4739731074419264,4.370414956576545) q[0];
p(4.91447789155306) q[4];
rzz(3.810424660730481) q[7],q[5];
ch q[6],q[1];
sdg q[1];
s q[5];
rzz(4.872584858061494) q[2],q[4];
swap q[3],q[0];
csx q[7],q[6];
xx_minus_yy(4.364732429607092,1.539711366434543) q[1],q[3];
swap q[6],q[0];
h q[4];
rzx(4.728624039742595) q[5],q[7];
rx(0.04135731914063035) q[0];
u3(5.204705832680271,3.046684508298133,5.364598985991842) q[2];
r(3.167977202944766,0.4923783765857933) q[3];
id q[4];
sdg q[6];
iswap q[7],q[1];
crz(5.537315060316528) q[3],q[5];
dcx q[4],q[1];
dcx q[6],q[0];
ecr q[7],q[2];
ch q[0],q[2];
cu3(3.635556813115056,2.148855787121812,2.633444276051519) q[4],q[7];
cy q[6],q[3];
rz(0.25744138738022276) q[1];
ry(2.5192891144962344) q[5];
ch q[1],q[4];
u1(3.834943615749385) q[3];
cy q[0],q[2];
crx(3.0073992555598106) q[5],q[6];
sxdg q[7];
h q[3];
u2(0.7684977839819236,5.064269750503158) q[0];
id q[6];
u3(4.156789975785996,6.238749066917127,1.1871780320799896) q[7];
cz q[1],q[4];
cx q[5],q[2];
cu(5.67260410348416,5.045501599249377,5.643850364000483,5.5666665679324465) q[7],q[3];
csx q[6],q[5];
csdg q[4],q[1];
csdg q[2],q[0];
dcx q[4],q[0];
ch q[2],q[5];
ecr q[1],q[3];
swap q[6],q[7];
xx_plus_yy(0.07939585603467601,1.073458774361393) q[3],q[2];
z q[1];
csdg q[6],q[4];
xx_minus_yy(3.4118049683749083,0.6401969615681682) q[7],q[0];
p(0.33939583473585366) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];