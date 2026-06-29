OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; sdg q0; sx q0; s q0; s q1; cx q0,q1; ry(0.5*param0) q0; ry((-0.5)*param0) q1; cx q0,q1; sdg q1; sdg q0; sxdg q0; s q0; rz(param1) q1; }
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
gate rcccx q0,q1,q2,q3 { h q3; t q3; cx q2,q3; tdg q3; h q3; cx q0,q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; h q3; t q3; cx q2,q3; tdg q3; h q3; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
qreg q[9];
creg meas[9];
cry(1.0507201913780568) q[2],q[8];
rzx(5.379320756128239) q[3],q[5];
iswap q[0],q[7];
cry(1.6065431931193521) q[4],q[6];
r(2.7299055326400135,2.80331739930483) q[1];
crz(1.196429905907207) q[6],q[2];
sdg q[1];
rzx(0.1884033189236258) q[7],q[4];
csx q[8],q[5];
ccz q[0],q[5],q[1];
rx(1.4363216440432562) q[8];
csx q[6],q[7];
iswap q[3],q[4];
cu(1.5954302464557133,3.589440048510018,3.746511914285776,1.9368555889809949) q[1],q[6];
xx_minus_yy(2.570694167634441,5.537476912436928) q[3],q[4];
rxx(4.9126922258385815) q[8],q[7];
swap q[0],q[5];
u(3.519569908449362,4.547296896947313,4.505112616210665) q[2];
cu(5.39279081868582,1.4465702350547007,6.244026493204514,2.0797818481569292) q[4],q[2];
swap q[1],q[7];
ccx q[8],q[6],q[5];
cz q[3],q[0];
ch q[3],q[1];
u(4.045559487227495,4.105065149841366,3.994319411686153) q[8];
c3sqrtx q[7],q[5],q[4],q[0];
rz(2.1955026882808113) q[6];
swap q[1],q[0];
cu3(0.6570470190859857,3.3091757012015917,1.3968222687844276) q[7],q[4];
cry(4.77450559753736) q[5],q[2];
x q[3];
ryy(5.869687040045388) q[6],q[8];
cu(1.0889289769124677,3.6885323150362797,1.559216770625479,3.057858602799537) q[5],q[3];
ccx q[4],q[2],q[1];
ecr q[8],q[7];
s q[0];
s q[6];
xx_minus_yy(0.6254058719535827,0.3079233828542554) q[0],q[6];
c3sqrtx q[8],q[1],q[2],q[5];
swap q[4],q[7];
ccx q[8],q[5],q[2];
crx(1.647010051725865) q[3],q[6];
rcccx q[4],q[0],q[1],q[7];
cswap q[2],q[8],q[5];
u3(2.1459141275298173,1.4831959858449213,3.123913653350043) q[4];
tdg q[7];
c3sqrtx q[3],q[0],q[1],q[6];
xx_plus_yy(4.492294557525396,1.1402832202335076) q[8],q[2];
rxx(1.0858886213821521) q[1],q[0];
ccx q[4],q[7],q[6];
cy q[3],q[5];
rccx q[2],q[5],q[8];
u3(5.226469529422574,4.436433498319035,2.084139081764859) q[0];
cswap q[4],q[6],q[7];
s q[3];
rx(4.561549626738646) q[1];
u3(2.7444965420646454,5.640708872913764,1.267273382625566) q[2];
cu1(6.278895344124908) q[0],q[1];
t q[7];
xx_minus_yy(2.1031583270050955,3.9973641275720997) q[3],q[5];
cu3(1.949369565946397,4.901779732315261,2.531204850611388) q[8],q[4];
id q[6];
z q[3];
ecr q[1],q[6];
iswap q[5],q[2];
u3(4.205015177957589,4.497219218640723,0.7211605767663397) q[0];
cs q[4],q[7];
cu3(3.7317391492541074,1.6580247354522009,6.218517253144742) q[0],q[2];
crz(5.858117961614186) q[3],q[6];
cs q[4],q[7];
rccx q[1],q[8],q[5];
crz(3.5553745213100925) q[8],q[0];
iswap q[7],q[2];
rccx q[5],q[6],q[1];
csdg q[3],q[4];
r(1.449987618868569,2.3589869911331047) q[3];
csx q[0],q[5];
cswap q[1],q[2],q[8];
ch q[6],q[4];
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