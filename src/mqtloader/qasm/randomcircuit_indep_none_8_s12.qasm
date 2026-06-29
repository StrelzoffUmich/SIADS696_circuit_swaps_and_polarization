OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
gate rcccx q0,q1,q2,q3 { h q3; t q3; cx q2,q3; tdg q3; h q3; cx q0,q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; h q3; t q3; cx q2,q3; tdg q3; h q3; }
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; sdg q0; sx q0; s q0; s q1; cx q0,q1; ry(0.5*param0) q0; ry((-0.5)*param0) q1; cx q0,q1; sdg q1; sdg q0; sxdg q0; s q0; rz(param1) q1; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
qreg q[8];
creg meas[8];
rzx(0.13670246674230233) q[2],q[7];
u1(5.568097355077936) q[3];
cs q[4],q[1];
csx q[0],q[6];
r(2.00794541656628,6.10165239878709) q[5];
ecr q[4],q[0];
u(1.9330353989700109,2.5162048065242772,2.2704232732190732) q[7];
cu1(0.05445542629718607) q[3],q[6];
cswap q[1],q[2],q[5];
t q[2];
cry(5.918200429047135) q[6],q[4];
u2(4.05959720144576,5.720077363287227) q[5];
rzz(0.9850588168167487) q[7],q[1];
cu1(1.2280123793582822) q[3],q[0];
rcccx q[6],q[7],q[4],q[5];
y q[3];
x q[1];
x q[0];
sx q[2];
rxx(1.0982238305052325) q[2],q[7];
y q[1];
s q[6];
csdg q[5],q[3];
rx(1.8310226001161238) q[4];
cry(0.6406153321722844) q[3],q[6];
h q[0];
tdg q[2];
ch q[4],q[1];
ry(4.40841697368812) q[7];
rcccx q[6],q[3],q[5],q[7];
u1(2.9104411021804744) q[1];
cu(4.227101981865178,3.008928054625208,5.163269450078994,5.121488225353055) q[4],q[0];
sdg q[2];
crx(2.2971263780723676) q[7],q[1];
xx_minus_yy(1.2270762685344139,6.1878388805046605) q[0],q[6];
xx_plus_yy(0.5080985685130993,1.2420240770844644) q[5],q[4];
cu1(1.3287817743557302) q[3],q[2];
csdg q[4],q[1];
ry(3.334292317661256) q[3];
x q[2];
h q[6];
sdg q[5];
rzz(1.5009005686397547) q[7],q[0];
csx q[6],q[0];
swap q[2],q[7];
u2(6.1167894612601525,2.141761245631009) q[5];
cu3(3.956837872971264,4.600614523539536,6.007260327174721) q[1],q[4];
u2(1.5016874796971944,0.43488307282409094) q[3];
rcccx q[1],q[3],q[7],q[4];
cu(0.4694478279995398,0.8791660365812587,4.351147879787088,0.5135027987453625) q[6],q[5];
r(2.051689315215335,3.5636011746077227) q[0];
c3sqrtx q[0],q[4],q[6],q[2];
sdg q[5];
cx q[7],q[1];
cu1(3.3754889303570073) q[7],q[6];
tdg q[0];
cry(0.48020413626313724) q[2],q[5];
tdg q[4];
u2(2.5493479087119844,3.0683576203465353) q[3];
rz(3.281918559888721) q[7];
rz(5.2306537769803585) q[0];
y q[6];
c3sqrtx q[1],q[4],q[5],q[2];
dcx q[3],q[2];
crx(6.152720242375343) q[6],q[7];
ryy(1.565807690174062) q[0],q[4];
sx q[5];
p(4.562509030697591) q[1];
x q[4];
ryy(5.8090478932285) q[1],q[5];
y q[2];
h q[0];
sdg q[3];
rx(0.6840861572923178) q[6];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];