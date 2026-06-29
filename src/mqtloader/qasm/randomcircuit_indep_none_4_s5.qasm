OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
qreg q[4];
creg meas[4];
cu3(2.889077969057205,3.3839323400613304,6.001567737656112) q[1],q[3];
rx(4.152655957978368) q[2];
sdg q[3];
y q[1];
y q[3];
ryy(3.92114943331845) q[1],q[2];
cp(3.8950338279952477) q[1],q[2];
h q[3];
cz q[1],q[0];
y q[2];
u1(3.8825966600259765) q[0];
p(6.072626074478666) q[3];
ry(2.713323100911148) q[3];
u(5.789071847736802,4.832205520718357,4.219743055095117) q[0];
dcx q[1],q[2];
ry(0.32533754848778496) q[3];
ecr q[0],q[1];
barrier q[0],q[1],q[2],q[3];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];