OPENQASM 2.0;
include "qelib1.inc";
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; sdg q0; sx q0; s q0; s q1; cx q0,q1; ry(0.5*param0) q0; ry((-0.5)*param0) q1; cx q0,q1; sdg q1; sdg q0; sxdg q0; s q0; rz(param1) q1; }
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
gate rcccx q0,q1,q2,q3 { h q3; t q3; cx q2,q3; tdg q3; h q3; cx q0,q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; h q3; t q3; cx q2,q3; tdg q3; h q3; }
qreg q[9];
creg meas[9];
ecr q[2],q[7];
dcx q[3],q[4];
cp(5.39049097912428) q[8],q[5];
csx q[1],q[0];
r(1.7230189211714217,5.7287099080786925) q[6];
ccz q[1],q[5],q[8];
sx q[7];
ccz q[0],q[2],q[6];
cp(3.098214568844268) q[4],q[3];
dcx q[5],q[3];
cz q[1],q[8];
swap q[2],q[6];
cs q[4],q[0];
rxx(5.537467835878169) q[3],q[2];
ccz q[0],q[7],q[8];
cu1(0.8881539390135246) q[5],q[1];
cz q[6],q[4];
rzx(4.746576328052876) q[1],q[0];
rzx(4.779632135278802) q[8],q[6];
ch q[4],q[5];
dcx q[3],q[7];
ch q[7],q[5];
crz(4.738909164049312) q[8],q[4];
crz(2.8186697498901134) q[2],q[6];
cp(3.3898346889082713) q[1],q[0];
crz(0.388012451936937) q[6],q[0];
dcx q[5],q[4];
ccz q[7],q[1],q[8];
swap q[3],q[2];
cz q[8],q[5];
xx_minus_yy(0.4444717690387993,6.21410560834441) q[4],q[1];
xx_minus_yy(0.3815699339695386,6.220995927402712) q[2],q[7];
t q[0];
cs q[3],q[6];
ryy(4.382309725388623) q[6],q[0];
cs q[4],q[3];
swap q[5],q[2];
rzz(2.0451812972848127) q[7],q[8];
cp(3.317919042073649) q[0],q[6];
csdg q[2],q[5];
dcx q[8],q[3];
cx q[4],q[1];
s q[7];
ryy(5.003663892442385) q[7],q[6];
cy q[8],q[2];
ryy(0.9818081038210587) q[4],q[3];
csdg q[1],q[0];
cx q[6],q[4];
rxx(1.5892721949271043) q[0],q[7];
cz q[8],q[2];
cry(1.7103034688675764) q[5],q[3];
cu1(3.9461657027749766) q[6],q[5];
cu3(4.2684380469984164,3.2087150614448667,5.838661734061429) q[7],q[2];
ryy(1.253739914002637) q[8],q[0];
csx q[3],q[4];
x q[1];
ch q[3],q[5];
xx_minus_yy(5.041539728310238,4.588802011766603) q[6],q[2];
p(5.563601098326938) q[0];
cz q[1],q[8];
csdg q[0],q[3];
cu3(3.904931514741761,6.048446683831083,4.797104444084305) q[1],q[8];
rcccx q[2],q[4],q[6],q[5];
cu1(3.881818408794219) q[7],q[8];
cswap q[0],q[1],q[5];
cp(0.9468843523162511) q[4],q[2];
r(2.5223653826160843,2.6656054756078187) q[6];
csx q[1],q[4];
cswap q[3],q[6],q[5];
cu3(5.926335792903968,4.257445496015684,1.560285538630851) q[0],q[7];
rxx(3.053566085119466) q[2],q[8];
z q[1];
cs q[6],q[3];
u1(0.9940675620433592) q[7];
x q[0];
csdg q[5],q[4];
u1(4.148055207493592) q[8];
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