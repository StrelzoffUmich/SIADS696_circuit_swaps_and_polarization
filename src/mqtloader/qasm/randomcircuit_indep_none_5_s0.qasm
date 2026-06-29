OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
qreg q[5];
creg meas[5];
rzx(5.126159064066656) q[2],q[1];
cs q[3],q[4];
sxdg q[0];
sx q[4];
rxx(1.8831453470115125) q[3],q[2];
ry(2.6558221377616955) q[1];
cy q[2],q[4];
cu3(4.3073873243438365,4.086956167564611,4.325638382299154) q[0],q[1];
u2(2.4436653767928673,0.8488363754081273) q[3];
cp(2.0223650288392) q[1],q[4];
cs q[2],q[3];
p(3.7340972178071192) q[0];
cs q[3],q[2];
sxdg q[0];
cu1(4.945484520918815) q[4],q[1];
x q[2];
u3(1.4491677387649573,0.32685947450826425,2.5418741759590953) q[1];
rx(1.2472942445440403) q[4];
ch q[0],q[3];
cy q[4],q[3];
sdg q[2];
swap q[0],q[1];
dcx q[1],q[3];
swap q[2],q[4];
ch q[4],q[2];
ryy(2.6053595402076226) q[3],q[0];
y q[1];
h q[3];
rxx(5.426410725722254) q[1],q[0];
csdg q[4],q[2];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];