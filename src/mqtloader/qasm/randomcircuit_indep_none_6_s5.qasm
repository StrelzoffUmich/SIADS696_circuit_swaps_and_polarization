OPENQASM 2.0;
include "qelib1.inc";
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
qreg q[6];
creg meas[6];
dcx q[1],q[5];
cz q[2],q[4];
ecr q[3],q[0];
cu3(5.192227903120697,0.6980943301565483,0.1567134589172818) q[2],q[0];
cswap q[1],q[5],q[3];
u1(1.8203772161283145) q[4];
csx q[5],q[2];
u1(0.3499457784561316) q[4];
sxdg q[0];
cz q[3],q[1];
cry(4.7146018947572665) q[0],q[5];
cu(1.1027414969933367,0.900230546875808,2.49754331320993,3.0345368385860607) q[2],q[3];
iswap q[4],q[1];
rx(4.533039299630897) q[3];
tdg q[0];
cz q[2],q[4];
cy q[5],q[1];
csdg q[0],q[2];
ch q[5],q[3];
rzz(4.797147746170943) q[1],q[4];
csx q[4],q[0];
csx q[3],q[1];
ch q[2],q[5];
ccz q[4],q[1],q[0];
ccz q[3],q[2],q[5];
dcx q[5],q[1];
xx_plus_yy(0.9488255812144709,1.2732744598971972) q[0],q[3];
cu(3.0630157473555037,2.1858688468137073,0.5066759081619147,0.14339341216825127) q[4],q[2];
dcx q[4],q[1];
csx q[3],q[5];
csx q[2],q[0];
crz(4.7439307362805465) q[3],q[2];
cz q[5],q[0];
swap q[1],q[4];
swap q[0],q[4];
cx q[5],q[2];
crz(2.6342915674207976) q[1],q[3];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];