OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
qreg q[4];
creg meas[4];
rzx(3.415696558991173) q[2],q[3];
cs q[0],q[1];
id q[1];
xx_plus_yy(1.1036768144935105,5.423513122375916) q[2],q[3];
sx q[0];
dcx q[3],q[0];
sx q[2];
p(4.066411630084061) q[1];
z q[0];
rccx q[3],q[1],q[2];
z q[3];
p(1.9493071941838678) q[2];
rxx(3.052593588320219) q[1],q[0];
y q[2];
csx q[3],q[0];
u2(2.1231588472374674,2.4606147501308975) q[1];
sxdg q[3];
cu1(4.945484520918815) q[1],q[0];
r(1.5040025672010786,5.507112841004416) q[2];
rx(0.32685947450826425) q[2];
ch q[3],q[1];
barrier q[0],q[1],q[2],q[3];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];