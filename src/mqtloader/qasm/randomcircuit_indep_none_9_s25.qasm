OPENQASM 2.0;
include "qelib1.inc";
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; sdg q0; sx q0; s q0; s q1; cx q0,q1; ry(0.5*param0) q0; ry((-0.5)*param0) q1; cx q0,q1; sdg q1; sdg q0; sxdg q0; s q0; rz(param1) q1; }
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate rcccx q0,q1,q2,q3 { h q3; t q3; cx q2,q3; tdg q3; h q3; cx q0,q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; h q3; t q3; cx q2,q3; tdg q3; h q3; }
qreg q[9];
creg meas[9];
cp(0.34100369047617385) q[4],q[8];
csdg q[3],q[6];
dcx q[1],q[5];
sx q[2];
csx q[7],q[0];
xx_minus_yy(2.9414951272721104,1.4074238850022727) q[5],q[6];
cx q[4],q[0];
x q[2];
swap q[3],q[7];
xx_minus_yy(5.920137317533596,5.489041974120298) q[8],q[1];
ecr q[7],q[2];
csx q[1],q[0];
cu(1.1455427272475724,0.15086508925216804,2.900542078781012,4.58054411291565) q[6],q[5];
crx(2.49198856182218) q[3],q[4];
rzz(4.543058689199608) q[7],q[4];
csx q[5],q[6];
cx q[8],q[2];
rzx(0.08426018267778762) q[1],q[0];
xx_plus_yy(5.944072695880025,5.197458861405819) q[8],q[0];
p(0.05416268559455427) q[4];
ch q[3],q[7];
rzz(2.3234971354110363) q[6],q[2];
z q[5];
r(6.138226429807445,1.4833899528201413) q[3];
cu3(2.2123822577976804,1.6609999776793416,2.785794996133839) q[7],q[2];
cz q[0],q[1];
tdg q[6];
ryy(6.014284180217201) q[8],q[4];
cy q[4],q[6];
dcx q[2],q[8];
y q[7];
iswap q[5],q[1];
csdg q[3],q[0];
ccz q[8],q[7],q[3];
rzz(0.15558602610521605) q[2],q[6];
rcccx q[4],q[0],q[1],q[5];
dcx q[6],q[8];
csx q[7],q[3];
ccx q[4],q[1],q[0];
x q[5];
ryy(4.65942692394969) q[2],q[8];
crx(3.300720175058269) q[7],q[0];
cry(0.5033275193594593) q[1],q[4];
rzx(0.6848844484238203) q[6],q[5];
cu1(4.473277761742836) q[1],q[5];
crx(4.720732254976119) q[3],q[4];
c3sqrtx q[2],q[7],q[6],q[0];
c3sqrtx q[1],q[0],q[6],q[3];
t q[4];
csx q[8],q[7];
csx q[2],q[5];
xx_minus_yy(5.462659814697061,5.60424636042043) q[0],q[5];
cu1(0.8304087356103618) q[6],q[7];
crz(4.918473355751992) q[3],q[4];
rzz(5.254316305821917) q[8],q[2];
ch q[2],q[6];
cu(0.38891107298414856,5.330374472753089,3.693463808798516,6.254696522323478) q[1],q[4];
cz q[7],q[0];
dcx q[5],q[3];
tdg q[6];
rxx(2.870462802674135) q[0],q[7];
cy q[8],q[4];
crx(0.6078466873190411) q[3],q[5];
cz q[2],q[1];
ry(3.4400815020509357) q[2];
crz(5.6028751813284225) q[1],q[3];
crz(3.163373151675996) q[8],q[4];
xx_plus_yy(6.076185795290006,2.3474799151060752) q[7],q[6];
ry(3.1284677867496167) q[0];
cry(0.9299709104322591) q[2],q[7];
cp(0.057923421172547354) q[6],q[3];
ryy(1.6896119256362845) q[5],q[1];
cu3(5.687665966419219,0.13082618760601147,5.67404643197523) q[8],q[4];
rxx(1.76075108531627) q[4],q[5];
crx(4.6258116662273485) q[0],q[1];
rzz(2.529033570342499) q[2],q[8];
rzz(5.607349352282309) q[7],q[6];
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