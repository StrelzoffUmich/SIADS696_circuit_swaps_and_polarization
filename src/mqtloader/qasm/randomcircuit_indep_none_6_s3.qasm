OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; sdg q0; sx q0; s q0; s q1; cx q0,q1; ry(0.5*param0) q0; ry((-0.5)*param0) q1; cx q0,q1; sdg q1; sdg q0; sxdg q0; s q0; rz(param1) q1; }
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
qreg q[6];
creg meas[6];
crx(0.017762767097894937) q[1],q[5];
cx q[0],q[3];
ryy(3.402132231646454) q[2],q[4];
csx q[0],q[2];
cu(5.947780728402538,5.79819173389371,5.530773866164002,0.4043587630605736) q[3],q[5];
iswap q[1],q[4];
z q[0];
ecr q[5],q[3];
ryy(4.797997943405111) q[1],q[2];
ry(0.12710081959821112) q[4];
cp(4.470196504596145) q[2],q[4];
u(3.3683440557772792,3.507684593123973,5.6930928599962325) q[1];
xx_plus_yy(1.7758820747473298,1.39621110079262) q[3],q[5];
xx_minus_yy(4.315041058026253,3.8734381098314867) q[2],q[3];
rccx q[5],q[4],q[1];
rx(2.742838604165215) q[0];
cry(4.366000516157721) q[3],q[0];
xx_plus_yy(0.4285332307834998,1.3234200494475383) q[1],q[2];
swap q[5],q[4];
rx(1.212762773874118) q[3];
csdg q[2],q[0];
rzz(5.9210402002405464) q[5],q[4];
tdg q[1];
rzz(4.872584858061494) q[2],q[1];
swap q[5],q[0];
csx q[4],q[3];
u2(4.529729624612599,4.886182663122709) q[3];
id q[0];
p(4.280277449481933) q[1];
xx_minus_yy(3.135331656776588,3.0506000080173377) q[2],q[5];
u2(0.0643031752929269,0.7791491092568377) q[1];
rz(5.788995413931423) q[0];
iswap q[5],q[3];
rx(4.370141767575751) q[2];
u3(2.5448697221725025,0.04135731914063035,5.204705832680271) q[4];
p(1.6430708165352044) q[4];
p(1.7387861174023131) q[5];
ecr q[2],q[1];
z q[3];
tdg q[0];
ry(2.5505749328244165) q[4];
ch q[2],q[1];
swap q[3],q[0];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];