OPENQASM 2.0;
include "qelib1.inc";
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; sdg q0; sx q0; s q0; s q1; cx q0,q1; ry(0.5*param0) q0; ry((-0.5)*param0) q1; cx q0,q1; sdg q1; sdg q0; sxdg q0; s q0; rz(param1) q1; }
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
gate rcccx q0,q1,q2,q3 { h q3; t q3; cx q2,q3; tdg q3; h q3; cx q0,q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; h q3; t q3; cx q2,q3; tdg q3; h q3; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
qreg q[9];
creg meas[9];
c3sqrtx q[6],q[3],q[4],q[7];
cx q[2],q[8];
csdg q[0],q[5];
ry(1.0768745138035725) q[1];
cz q[2],q[0];
p(0.5625649602318823) q[5];
csdg q[4],q[6];
crz(4.317243454546385) q[1],q[7];
rxx(2.082001764601333) q[3],q[8];
cry(3.3657690428054505) q[7],q[6];
iswap q[5],q[1];
iswap q[8],q[3];
crx(2.0659827206207995) q[2],q[4];
cswap q[3],q[4],q[5];
sdg q[1];
u1(5.794797709149284) q[2];
iswap q[8],q[0];
rxx(4.766498996373667) q[6],q[7];
csx q[5],q[7];
swap q[0],q[6];
crz(1.815981943186897) q[1],q[3];
xx_minus_yy(1.689354297256685,4.745891677358445) q[8],q[2];
ryy(0.6139601760057403) q[2],q[0];
ryy(4.531070609085379) q[6],q[4];
ryy(0.5032879884138091) q[3],q[8];
cp(5.223379172772005) q[1],q[5];
cz q[0],q[3];
cz q[8],q[4];
crx(4.432646815568427) q[2],q[6];
cu(2.6222624932157963,4.955796226953902,4.2129618387662955,0.19968375852080614) q[1],q[7];
rxx(0.6514018072203254) q[8],q[2];
cu3(4.092349098852995,4.769509765272288,4.194352593698525) q[5],q[1];
ccz q[4],q[7],q[6];
cu3(3.562077946228642,2.8368076875365094,6.124344219577483) q[3],q[0];
swap q[4],q[7];
cu(1.8638021382171144,0.3540241972262331,6.243276942390347,2.4527398075876925) q[6],q[2];
cy q[3],q[0];
ryy(0.9711145392182988) q[1],q[5];
sdg q[5];
crx(3.8307019166935046) q[2],q[7];
iswap q[3],q[0];
ch q[4],q[8];
csdg q[1],q[6];
cu3(2.1646996602033055,1.4594056210934832,2.467470477152319) q[0],q[5];
crx(4.348451094620217) q[4],q[6];
iswap q[8],q[2];
dcx q[7],q[1];
cy q[2],q[4];
crx(6.048307138603669) q[6],q[8];
cp(0.606395108958728) q[7],q[3];
csdg q[0],q[1];
rcccx q[0],q[1],q[3],q[2];
csdg q[8],q[5];
id q[7];
rzz(2.7799474529968196) q[4],q[6];
ch q[4],q[3];
crx(2.4548330561607856) q[6],q[7];
ccz q[0],q[8],q[5];
xx_minus_yy(4.159117462995512,3.752071994212636) q[1],q[2];
cry(2.9283396266049375) q[4],q[5];
cp(3.5807782863235786) q[6],q[8];
cry(5.594726038872753) q[3],q[2];
ccx q[7],q[1],q[0];
iswap q[5],q[7];
swap q[1],q[6];
xx_plus_yy(3.455140021414297,2.911946713544212) q[8],q[2];
crz(6.0883565606384265) q[4],q[3];
cs q[1],q[3];
ryy(4.022633619280714) q[8],q[0];
cz q[4],q[2];
cu1(2.4561414643512505) q[7],q[6];
csx q[7],q[4];
ryy(6.003242566735445) q[6],q[5];
ryy(4.036288347204874) q[0],q[8];
ryy(4.35867517649564) q[1],q[3];
id q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];