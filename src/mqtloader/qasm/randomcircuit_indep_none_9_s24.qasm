OPENQASM 2.0;
include "qelib1.inc";
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; sdg q0; sx q0; s q0; s q1; cx q0,q1; ry(0.5*param0) q0; ry((-0.5)*param0) q1; cx q0,q1; sdg q1; sdg q0; sxdg q0; s q0; rz(param1) q1; }
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
qreg q[9];
creg meas[9];
iswap q[0],q[7];
cswap q[3],q[5],q[1];
cu1(5.822367474462155) q[6],q[8];
cs q[4],q[2];
xx_plus_yy(5.031462681562515,6.225396910722311) q[4],q[3];
ccz q[8],q[7],q[1];
p(5.8706352736658145) q[2];
cu3(0.8766382229228101,5.324568837511969,1.6940266702871425) q[5],q[6];
u1(0.06264946828728012) q[0];
sx q[0];
x q[5];
csx q[4],q[1];
c3sqrtx q[6],q[2],q[7],q[3];
r(0.833585548236889,3.781337239894039) q[8];
t q[3];
cry(3.069031727274617) q[4],q[7];
dcx q[8],q[2];
rz(2.6831129020414375) q[0];
p(1.3267560150768072) q[6];
u(4.729404860554826,4.765853592649405,5.402282201229015) q[5];
xx_minus_yy(5.191703598987795,3.82877739192867) q[7],q[2];
cx q[3],q[6];
xx_plus_yy(5.0059253074565,0.802998321481407) q[1],q[8];
p(1.1316790895094992) q[4];
crz(0.0634671491724168) q[5],q[0];
ry(3.1759202525382078) q[1];
u(5.734553203632922,0.7247159050197148,0.3213785108360022) q[2];
c3sqrtx q[8],q[7],q[3],q[6];
csx q[4],q[0];
cy q[7],q[3];
cx q[8],q[4];
rzx(3.129709410922436) q[6],q[0];
u3(2.788037005190842,3.0147560752608555,1.756342115418957) q[5];
xx_minus_yy(4.468310844755931,4.142278556673747) q[0],q[2];
ch q[6],q[5];
dcx q[8],q[7];
cu(2.996058084326774,0.6889555939866889,5.137397428765332,0.02251664521680029) q[3],q[4];
csx q[5],q[3];
ccz q[2],q[4],q[0];
cry(4.705200238793866) q[7],q[8];
cx q[6],q[1];
rzx(4.009694315812785) q[5],q[3];
crx(4.381892070913886) q[6],q[2];
csx q[4],q[0];
z q[7];
rz(4.9855390133579975) q[8];
ccz q[4],q[3],q[8];
rz(3.574981863372123) q[5];
cswap q[2],q[0],q[6];
r(5.242518574474614,2.8440384937119645) q[1];
c3sqrtx q[4],q[8],q[7],q[2];
rz(3.155730473496591) q[3];
swap q[0],q[1];
cy q[6],q[5];
y q[2];
rzz(2.429474381935527) q[8],q[3];
iswap q[6],q[7];
rzz(5.159804894829157) q[1],q[4];
ryy(1.8885192090959724) q[3],q[4];
iswap q[5],q[0];
cswap q[1],q[7],q[8];
u2(5.840390786118086,4.490116278992565) q[2];
crz(2.7577508550203347) q[5],q[6];
ccx q[7],q[2],q[4];
crz(2.2458176140231307) q[3],q[0];
cy q[8],q[1];
ecr q[5],q[3];
sx q[2];
cy q[4],q[7];
cu(4.681880068402451,1.1887302874633685,4.900563561731561,1.3980580435792762) q[1],q[8];
ch q[8],q[2];
rzz(0.9819794631752691) q[4],q[7];
ry(5.062314824370894) q[6];
cx q[0],q[3];
cy q[2],q[4];
u1(2.477277006999034) q[7];
rxx(3.268500341221862) q[6],q[0];
id q[5];
csdg q[8],q[1];
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