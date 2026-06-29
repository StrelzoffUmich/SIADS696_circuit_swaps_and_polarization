OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; sdg q0; sx q0; s q0; s q1; cx q0,q1; ry(0.5*param0) q0; ry((-0.5)*param0) q1; cx q0,q1; sdg q1; sdg q0; sxdg q0; s q0; rz(param1) q1; }
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
qreg q[9];
creg meas[9];
cx q[5],q[6];
ryy(4.9894933075929435) q[2],q[8];
ccx q[4],q[0],q[3];
crz(6.185753501980357) q[1],q[7];
swap q[5],q[6];
dcx q[3],q[2];
cu(3.48351101186116,4.230491747414989,3.5533662426254042,2.4569869419786525) q[4],q[8];
cry(0.13435634986315936) q[1],q[7];
x q[0];
cswap q[3],q[0],q[8];
cu1(4.449393224201746) q[1],q[2];
ccz q[4],q[6],q[7];
cs q[4],q[5];
cu1(4.704476763164805) q[2],q[0];
rxx(3.6591310092331693) q[3],q[6];
ryy(0.720167693853525) q[7],q[8];
ccx q[4],q[5],q[8];
rxx(6.225548293668903) q[0],q[7];
y q[1];
cu1(6.2048685765019) q[6],q[3];
cz q[8],q[2];
csx q[6],q[5];
crx(4.800268327126565) q[4],q[0];
sx q[1];
cs q[3],q[7];
rzz(4.855497266950871) q[5],q[7];
cu3(5.462191986474673,4.92170989360168,4.556215178294383) q[6],q[8];
ccx q[4],q[2],q[0];
swap q[3],q[1];
cz q[0],q[1];
iswap q[8],q[7];
ecr q[6],q[2];
cswap q[5],q[3],q[4];
xx_minus_yy(0.8557709574016537,0.37358245090469727) q[0],q[7];
ccx q[2],q[8],q[1];
csx q[4],q[5];
cu3(5.065312421719555,0.0238068889666226,4.253264435614723) q[3],q[6];
cu3(4.839877146915379,4.0491963386114,3.4052683242309154) q[1],q[6];
iswap q[3],q[8];
csx q[7],q[0];
cu(2.845259244236378,5.007026937327072,0.41659070687389455,1.4766025283436628) q[5],q[4];
ryy(1.3320945768673755) q[5],q[3];
csdg q[8],q[7];
iswap q[2],q[1];
cs q[6],q[4];
cry(6.057492616616773) q[0],q[6];
rzz(6.195764269330564) q[7],q[2];
ryy(2.9129536927616813) q[3],q[4];
cry(2.89687969703454) q[5],q[8];
cy q[3],q[7];
t q[6];
rzz(1.5781707820246358) q[0],q[4];
xx_minus_yy(4.081892831188119,2.71733690167052) q[2],q[1];
cp(4.244549381017819) q[8],q[5];
cp(3.499496426794547) q[1],q[8];
cy q[3],q[4];
dcx q[5],q[6];
rzx(5.067133433601671) q[0],q[7];
cz q[7],q[5];
cs q[3],q[4];
swap q[6],q[1];
ecr q[0],q[8];
xx_plus_yy(2.132937767835479,0.8128420739886287) q[3],q[6];
cu(3.3512818357549423,5.363850051051716,6.004138243184126,6.275252791157256) q[2],q[4];
swap q[5],q[7];
cu3(2.307635541816659,0.7843809634136427,1.6501274888789392) q[8],q[0];
iswap q[3],q[5];
cp(4.399980593639267) q[7],q[1];
csx q[6],q[2];
crx(1.8072367086122487) q[0],q[8];
ccz q[7],q[3],q[5];
ccx q[0],q[2],q[8];
crx(2.150386568920867) q[4],q[6];
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