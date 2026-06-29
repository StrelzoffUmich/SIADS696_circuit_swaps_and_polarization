OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
qreg q[6];
creg meas[6];
rzx(5.126159064066656) q[3],q[0];
cs q[5],q[4];
cu(0.017206504032783308,5.3872299529679575,0.21102439329246717,4.584560380312186) q[2],q[1];
dcx q[1],q[4];
sx q[2];
p(2.4107171716328644) q[0];
cy q[5],q[3];
cu3(1.9493071941838678,3.052593588320219,5.588816891696628) q[3],q[2];
cy q[4],q[1];
z q[5];
p(5.868768495722669) q[0];
u2(3.9156003111145408,0.5278839723744851) q[5];
cu(5.231657474644899,4.945484520918815,1.5040025672010786,5.507112841004416) q[4],q[0];
u3(0.36799381575837975,2.1118857763128847,0.9442337383644338) q[3];
z q[2];
cx q[1],q[4];
sdg q[3];
r(1.2535925060964899,5.919471253635685) q[5];
csx q[2],q[0];
cu1(3.896916049018153) q[2],q[0];
csdg q[4],q[5];
dcx q[3],q[1];
ryy(4.468242482290601) q[1],q[2];
crx(5.856303728343588) q[0],q[4];
cry(0.7221430327460431) q[5],q[3];
iswap q[5],q[1];
id q[3];
xx_plus_yy(5.591630008039875,5.1671271502276594) q[4],q[2];
sdg q[0];
rzx(4.599330573313728) q[3],q[1];
csdg q[5],q[0];
tdg q[2];
tdg q[4];
rxx(0.4190256832211293) q[2],q[4];
id q[3];
ryy(2.1633633999373942) q[0],q[5];
s q[1];
t q[4];
ecr q[1],q[0];
sxdg q[3];
p(3.481453302102835) q[2];
u(5.087562850280148,3.521574266672758,1.8122039367930747) q[5];
swap q[4],q[3];
y q[2];
crz(2.554179775072455) q[5],q[0];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];