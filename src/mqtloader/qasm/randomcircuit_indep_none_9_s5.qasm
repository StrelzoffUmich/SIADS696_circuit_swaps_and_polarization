OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; sdg q0; sx q0; s q0; s q1; cx q0,q1; ry(0.5*param0) q0; ry((-0.5)*param0) q1; cx q0,q1; sdg q1; sdg q0; sxdg q0; s q0; rz(param1) q1; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate rcccx q0,q1,q2,q3 { h q3; t q3; cx q2,q3; tdg q3; h q3; cx q0,q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; h q3; t q3; cx q2,q3; tdg q3; h q3; }
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
qreg q[9];
creg meas[9];
crx(1.6207788068011701) q[3],q[6];
cx q[2],q[8];
ryy(2.6194350771272545) q[7],q[1];
sdg q[5];
swap q[0],q[4];
ryy(1.3784676130458824) q[1],q[6];
ccz q[0],q[4],q[7];
iswap q[8],q[2];
swap q[3],q[5];
xx_plus_yy(5.746127449698812,4.579633957334016) q[2],q[3];
sx q[8];
ccz q[7],q[1],q[6];
sxdg q[4];
cz q[0],q[5];
dcx q[5],q[7];
z q[0];
ccz q[8],q[4],q[3];
cswap q[2],q[6],q[1];
csx q[1],q[6];
ch q[3],q[4];
iswap q[0],q[8];
cs q[2],q[7];
ry(1.5124939549959957) q[5];
rzz(1.212762773874118) q[7],q[0];
ch q[3],q[2];
rzz(5.9210402002405464) q[8],q[5];
rx(0.9138481445950156) q[4];
csdg q[1],q[6];
csdg q[0],q[7];
cu3(5.6757376273698465,0.35974100775245765,4.529729624612599) q[5],q[3];
xx_plus_yy(4.886182663122709,4.280277449481933) q[8],q[2];
ch q[4],q[1];
tdg q[6];
xx_minus_yy(5.788995413931423,4.370141767575751) q[2],q[6];
u2(2.5448697221725025,0.04135731914063035) q[4];
rz(5.204705832680271) q[8];
iswap q[3],q[0];
rx(3.046684508298133) q[7];
u3(5.364598985991842,3.167977202944766,0.4923783765857933) q[1];
r(0.46337233955317175,4.962549063524577) q[5];
cu(5.566963099946367,1.97119549726753,3.433382919524903,3.547254911618815) q[7],q[1];
rcccx q[3],q[5],q[4],q[2];
ry(0.2721320458271034) q[6];
ch q[8],q[0];
rz(3.5557262507172633) q[4];
cry(2.3203483949345935) q[3],q[5];
cs q[7],q[2];
rz(0.8177790678296957) q[1];
ryy(1.8159370555575554) q[8],q[0];
cy q[6],q[1];
rcccx q[5],q[2],q[3],q[7];
cry(5.253733979494137) q[4],q[0];
h q[8];
u2(5.386959489599745,5.634974179032015) q[2];
xx_plus_yy(0.14169233763014327,6.255644309136386) q[6],q[5];
cp(4.312170002840389) q[0],q[8];
u3(5.67260410348416,5.045501599249377,5.643850364000483) q[3];
dcx q[1],q[4];
dcx q[2],q[1];
ch q[5],q[3];
ecr q[0],q[8];
swap q[4],q[6];
csdg q[5],q[4];
xx_minus_yy(0.6401969615681682,0.33939583473585366) q[8],q[1];
p(0.8601296188182305) q[6];
crz(5.182735198027271) q[7],q[2];
iswap q[0],q[3];
p(6.178861500560326) q[8];
swap q[5],q[7];
xx_minus_yy(3.0457260717831067,2.592667905154751) q[2],q[1];
rzz(5.872766147009556) q[3],q[6];
cu(4.53176556954921,4.691063671214756,5.0538250743020825,5.244638817648402) q[0],q[4];
xx_plus_yy(3.917467173068713,4.493978250948157) q[1],q[2];
rx(3.414343052828012) q[0];
rzx(5.585308633361838) q[7],q[6];
rzx(5.609914063304702) q[4],q[3];
cu1(5.717176012818502) q[5],q[8];
cu1(2.527435464075844) q[3],q[8];
id q[1];
t q[0];
c3sqrtx q[5],q[2],q[4],q[6];
iswap q[7],q[8];
x q[5];
crz(5.571290596694074) q[4],q[0];
cz q[6],q[1];
crz(0.6341117338227138) q[2],q[3];
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