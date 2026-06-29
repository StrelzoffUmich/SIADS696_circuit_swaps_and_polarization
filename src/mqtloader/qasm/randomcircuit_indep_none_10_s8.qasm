OPENQASM 2.0;
include "qelib1.inc";
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
gate rcccx q0,q1,q2,q3 { h q3; t q3; cx q2,q3; tdg q3; h q3; cx q0,q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; h q3; t q3; cx q2,q3; tdg q3; h q3; }
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; sdg q0; sx q0; s q0; s q1; cx q0,q1; ry(0.5*param0) q0; ry((-0.5)*param0) q1; cx q0,q1; sdg q1; sdg q0; sxdg q0; s q0; rz(param1) q1; }
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
qreg q[10];
creg meas[10];
ecr q[0],q[1];
x q[3];
rcccx q[7],q[6],q[2],q[4];
cswap q[8],q[5],q[9];
cry(4.5860018160117075) q[3],q[9];
xx_minus_yy(4.234716435965865,5.038692040761055) q[0],q[5];
cu3(0.9357639297380729,6.12295855687831,5.451628021923314) q[6],q[4];
u2(5.674155223636202,6.0440791945016) q[1];
cu1(3.902919985908311) q[8],q[7];
cu1(4.207478486107242) q[2],q[3];
ry(0.9037103512763025) q[1];
rcccx q[7],q[8],q[6],q[4];
swap q[5],q[9];
rcccx q[8],q[6],q[3],q[9];
sdg q[1];
z q[0];
c3sqrtx q[5],q[4],q[2],q[7];
xx_minus_yy(2.0096339137696453,3.3784973358441155) q[8],q[3];
sdg q[0];
swap q[9],q[5];
cry(5.353636372177599) q[7],q[6];
cp(4.194020095989464) q[1],q[2];
csx q[7],q[8];
cy q[3],q[1];
cry(4.377387647939181) q[6],q[9];
crz(3.942644625147732) q[5],q[2];
csdg q[0],q[4];
cu(6.213047004415373,2.226180104237326,4.568854710054939,5.989348772764583) q[4],q[0];
rzz(3.2340354203351223) q[2],q[8];
rcccx q[7],q[5],q[9],q[6];
p(2.0579660975333973) q[3];
p(4.289841070534182) q[7];
xx_minus_yy(0.915806682765929,3.7966832187840454) q[1],q[9];
cy q[3],q[8];
csdg q[5],q[6];
cry(4.547960356529325) q[2],q[4];
rxx(5.531022527102457) q[1],q[2];
cu3(2.365856492591085,0.9687781637021848,1.744045623327983) q[6],q[7];
rcccx q[4],q[0],q[8],q[3];
y q[9];
x q[5];
ryy(1.1724314218863816) q[6],q[5];
cry(4.071522159431019) q[4],q[8];
csdg q[3],q[1];
sdg q[0];
xx_plus_yy(1.2031009783582733,0.6291425191311164) q[1],q[9];
sxdg q[2];
r(0.8630839493764444,5.08816418274372) q[4];
crz(0.41053657062907634) q[7],q[0];
ryy(0.4677806273646717) q[6],q[8];
ch q[3],q[5];
u1(4.984998867119877) q[5];
cp(2.279129903528802) q[8],q[4];
rcccx q[6],q[1],q[0],q[7];
rz(3.0079701712311775) q[3];
ryy(5.7195738488942105) q[2],q[9];
rzx(3.3740953118803683) q[6],q[4];
rx(5.97472859193449) q[5];
sdg q[0];
rcccx q[8],q[9],q[3],q[2];
crz(3.333452787169752) q[7],q[1];
xx_minus_yy(3.998410262529986,4.227894597095535) q[6],q[0];
swap q[8],q[3];
rcccx q[7],q[5],q[9],q[2];
cs q[1],q[4];
c3sqrtx q[2],q[4],q[3],q[1];
cy q[5],q[0];
swap q[6],q[9];
c3sqrtx q[4],q[6],q[7],q[8];
dcx q[5],q[2];
p(5.88528917890705) q[0];
crx(1.3535115861893774) q[3],q[1];
u(5.058740770436585,4.915719421984012,4.760291634984493) q[7];
csdg q[1],q[2];
swap q[4],q[0];
iswap q[8],q[9];
rccx q[3],q[6],q[5];
rzx(5.215518625156226) q[5],q[3];
c3sqrtx q[0],q[9],q[1],q[2];
crx(1.1729642027487035) q[4],q[7];
z q[3];
iswap q[0],q[1];
cu1(3.3065669514289824) q[4],q[2];
c3sqrtx q[6],q[5],q[8],q[9];
r(5.105805796776039,0.8773758951772235) q[7];
c3sqrtx q[4],q[1],q[2],q[7];
c3sqrtx q[8],q[6],q[0],q[3];
cs q[5],q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];