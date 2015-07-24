`timescale 1ps / 1ps


/////////////////////// 
//// THCOMP NCL-UDP
//// 26. Z = AC + BC + AD + BD              LDD notation
//// {[A, C], [B, C], [A, D], [B, D]} -> Z  flow notation
/////////////////////// 
module THCOMP(Z,A,B,C,D);
output Z;
input A, B, C, D;
   THCOMPP #50 (Z,A,B,C,D);
endmodule
primitive THCOMPP(Z,A,B,C,D);
output Z;
input A, B, C, D;
reg Z;
initial Z = 1'b0;
table
// A B C D : Z : Z+         
0  0  0  0 : ? : 0;
1  0  0  0 : 0 : 0;
0  1  0  0 : 0 : 0;
1  1  0  0 : 0 : 0;
0  0  1  0 : 0 : 0;
1  0  1  0 : 0 : 1;
0  1  1  0 : 0 : 1;
1  1  1  0 : 0 : 1;
0  0  0  1 : 0 : 0;
1  0  0  1 : 0 : 1;
0  1  0  1 : 0 : 1;
1  1  0  1 : 0 : 1;
0  0  1  1 : 0 : 0;
1  0  1  1 : 0 : 1;
0  1  1  1 : 0 : 1;
1  1  1  1 : ? : 1;
0  0  0  0 : ? : 0;
1  0  0  0 : 1 : 1;
0  1  0  0 : 1 : 1;
1  1  0  0 : 1 : 1;
0  0  1  0 : 1 : 1;
1  0  1  0 : 1 : 1;
0  1  1  0 : 1 : 1;
1  1  1  0 : 1 : 1;
0  0  0  1 : 1 : 1;
1  0  0  1 : 1 : 1;
0  1  0  1 : 1 : 1;
1  1  0  1 : 1 : 1;
0  0  1  1 : 1 : 1;
1  0  1  1 : 1 : 1;
0  1  1  1 : 1 : 1;
1  1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// THAND NCL-UDP
//// 26. Z = AB + BC + AD           LDD notation
//// {[A, B], [B, C], [A, D]} -> Z  flow notation
/////////////////////// 
module THAND(Z,A,B,C,D);
output Z;
input A, B, C, D;
   THANDP #50 (Z,A,B,C,D);
endmodule
primitive THANDP(Z,A,B,C,D);
output Z;
input A, B, C, D;
reg Z;
initial Z = 1'b0;
table
// A B C D : Z : Z+         
0  0  0  0 : ? : 0;
1  0  0  0 : 0 : 0;
0  1  0  0 : 0 : 0;
1  1  0  0 : 0 : 1;
0  0  1  0 : 0 : 0;
1  0  1  0 : 0 : 0;
0  1  1  0 : 0 : 1;
1  1  1  0 : 0 : 1;
0  0  0  1 : 0 : 0;
1  0  0  1 : 0 : 1;
0  1  0  1 : 0 : 0;
1  1  0  1 : 0 : 1;
0  0  1  1 : 0 : 0;
1  0  1  1 : 0 : 1;
0  1  1  1 : 0 : 1;
1  1  1  1 : ? : 1;
0  0  0  0 : ? : 0;
1  0  0  0 : 1 : 1;
0  1  0  0 : 1 : 1;
1  1  0  0 : 1 : 1;
0  0  1  0 : 1 : 1;
1  0  1  0 : 1 : 1;
0  1  1  0 : 1 : 1;
1  1  1  0 : 1 : 1;
0  0  0  1 : 1 : 1;
1  0  0  1 : 1 : 1;
0  1  0  1 : 1 : 1;
1  1  0  1 : 1 : 1;
0  0  1  1 : 1 : 1;
1  0  1  1 : 1 : 1;
0  1  1  1 : 1 : 1;
1  1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// THXOR NCL-UDP
//// 26. Z = AB + CD        LDD notation
//// {[A, B], [C, D]} -> Z  flow notation
/////////////////////// 
module THXOR(Z,A,B,C,D);
output Z;
input A, B, C, D;
   THXORP #50 (Z,A,B,C,D);
endmodule
primitive THXORP(Z,A,B,C,D);
output Z;
input A, B, C, D;
reg Z;
initial Z = 1'b0;
table
// A B C D : Z : Z+         
0  0  0  0 : ? : 0;
1  0  0  0 : 0 : 0;
0  1  0  0 : 0 : 0;
1  1  0  0 : 0 : 1;
0  0  1  0 : 0 : 0;
1  0  1  0 : 0 : 0;
0  1  1  0 : 0 : 0;
1  1  1  0 : 0 : 1;
0  0  0  1 : 0 : 0;
1  0  0  1 : 0 : 0;
0  1  0  1 : 0 : 0;
1  1  0  1 : 0 : 1;
0  0  1  1 : 0 : 1;
1  0  1  1 : 0 : 1;
0  1  1  1 : 0 : 1;
1  1  1  1 : ? : 1;
0  0  0  0 : ? : 0;
1  0  0  0 : 1 : 1;
0  1  0  0 : 1 : 1;
1  1  0  0 : 1 : 1;
0  0  1  0 : 1 : 1;
1  0  1  0 : 1 : 1;
0  1  1  0 : 1 : 1;
1  1  1  0 : 1 : 1;
0  0  0  1 : 1 : 1;
1  0  0  1 : 1 : 1;
0  1  0  1 : 1 : 1;
1  1  0  1 : 1 : 1;
0  0  1  1 : 1 : 1;
1  0  1  1 : 1 : 1;
0  1  1  1 : 1 : 1;
1  1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH54W322 NCL-UDP
//// A has a weight of 3
//// B has a weight of 2
//// C has a weight of 2
//// 25. Z = AB + AC + BCD              LDD notation
//// {[A, B], [A, C], [B, C, D]} -> Z  flow notation
/////////////////////// 
module TH54W322(Z,A,B,C,D);
output Z;
input A, B, C, D;
   TH54W322P #50 (Z,A,B,C,D);
endmodule
primitive TH54W322P(Z,A,B,C,D);
output Z;
input A, B, C, D;
reg Z;
initial Z = 1'b0;
table
// A B C D : Z : Z+         
0  0  0  0 : ? : 0;
1  0  0  0 : 0 : 0;
0  1  0  0 : 0 : 0;
1  1  0  0 : 0 : 1;
0  0  1  0 : 0 : 0;
1  0  1  0 : 0 : 1;
0  1  1  0 : 0 : 0;
1  1  1  0 : 0 : 1;
0  0  0  1 : 0 : 0;
1  0  0  1 : 0 : 0;
0  1  0  1 : 0 : 0;
1  1  0  1 : 0 : 1;
0  0  1  1 : 0 : 0;
1  0  1  1 : 0 : 1;
0  1  1  1 : 0 : 1;
1  1  1  1 : ? : 1;
0  0  0  0 : ? : 0;
1  0  0  0 : 1 : 1;
0  1  0  0 : 1 : 1;
1  1  0  0 : 1 : 1;
0  0  1  0 : 1 : 1;
1  0  1  0 : 1 : 1;
0  1  1  0 : 1 : 1;
1  1  1  0 : 1 : 1;
0  0  0  1 : 1 : 1;
1  0  0  1 : 1 : 1;
0  1  0  1 : 1 : 1;
1  1  0  1 : 1 : 1;
0  0  1  1 : 1 : 1;
1  0  1  1 : 1 : 1;
0  1  1  1 : 1 : 1;
1  1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH44W322 NCL-UDP
//// A has a weight of 3
//// B has a weight of 2
//// C has a weight of 2
//// 24. Z = AB + AC + AD + BC              LDD notation
//// {[A, B], [A, C], [A, D], [B, C]} -> Z  flow notation
/////////////////////// 
module TH44W322(Z,A,B,C,D);
output Z;
input A, B, C, D;
   TH44W322P #50 (Z,A,B,C,D);
endmodule
primitive TH44W322P(Z,A,B,C,D);
output Z;
input A, B, C, D;
reg Z;
initial Z = 1'b0;
table
// A B C D : Z : Z+         
0  0  0  0 : ? : 0;
1  0  0  0 : 0 : 0;
0  1  0  0 : 0 : 0;
1  1  0  0 : 0 : 1;
0  0  1  0 : 0 : 0;
1  0  1  0 : 0 : 1;
0  1  1  0 : 0 : 1;
1  1  1  0 : 0 : 1;
0  0  0  1 : 0 : 0;
1  0  0  1 : 0 : 1;
0  1  0  1 : 0 : 0;
1  1  0  1 : 0 : 1;
0  0  1  1 : 0 : 0;
1  0  1  1 : 0 : 1;
0  1  1  1 : 0 : 1;
1  1  1  1 : ? : 1;
0  0  0  0 : ? : 0;
1  0  0  0 : 1 : 1;
0  1  0  0 : 1 : 1;
1  1  0  0 : 1 : 1;
0  0  1  0 : 1 : 1;
1  0  1  0 : 1 : 1;
0  1  1  0 : 1 : 1;
1  1  1  0 : 1 : 1;
0  0  0  1 : 1 : 1;
1  0  0  1 : 1 : 1;
0  1  0  1 : 1 : 1;
1  1  0  1 : 1 : 1;
0  0  1  1 : 1 : 1;
1  0  1  1 : 1 : 1;
0  1  1  1 : 1 : 1;
1  1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH54W32 NCL-UDP
//// A has a weight of 3
//// B has a weight of 2
//// 23. Z = AB + ACD       LDD notation
//// {[A, B], [A, C, D]} -> Z  flow notation
/////////////////////// 
module TH54W32(Z,A,B,C,D);
output Z;
input A, B, C, D;
   TH54W32P #50 (Z,A,B,C,D);
endmodule
primitive TH54W32P(Z,A,B,C,D);
output Z;
input A, B, C, D;
reg Z;
initial Z = 1'b0;
table
// A B C D : Z : Z+         
0  0  0  0 : ? : 0;
1  0  0  0 : 0 : 0;
0  1  0  0 : 0 : 0;
1  1  0  0 : 0 : 1;
0  0  1  0 : 0 : 0;
1  0  1  0 : 0 : 0;
0  1  1  0 : 0 : 0;
1  1  1  0 : 0 : 1;
0  0  0  1 : 0 : 0;
1  0  0  1 : 0 : 0;
0  1  0  1 : 0 : 0;
1  1  0  1 : 0 : 1;
0  0  1  1 : 0 : 0;
1  0  1  1 : 0 : 1;
0  1  1  1 : 0 : 0;
1  1  1  1 : ? : 1;
0  0  0  0 : ? : 0;
1  0  0  0 : 1 : 1;
0  1  0  0 : 1 : 1;
1  1  0  0 : 1 : 1;
0  0  1  0 : 1 : 1;
1  0  1  0 : 1 : 1;
0  1  1  0 : 1 : 1;
1  1  1  0 : 1 : 1;
0  0  0  1 : 1 : 1;
1  0  0  1 : 1 : 1;
0  1  0  1 : 1 : 1;
1  1  0  1 : 1 : 1;
0  0  1  1 : 1 : 1;
1  0  1  1 : 1 : 1;
0  1  1  1 : 1 : 1;
1  1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH34W32 NCL-UDP
//// A has a weight of 3
//// B has a weight of 2
//// 22. Z = A + BC + BD       LDD notation
//// {A, [B, C], [B, D]} -> Z  flow notation
/////////////////////// 
module TH34W32(Z,A,B,C,D);
output Z;
input A, B, C, D;
   TH34W32P #50 (Z,A,B,C,D);
endmodule
primitive TH34W32P(Z,A,B,C,D);
output Z;
input A, B, C, D;
reg Z;
initial Z = 1'b0;
table
// A B C D : Z : Z+         
0  0  0  0 : ? : 0;
1  0  0  0 : 0 : 1;
0  1  0  0 : 0 : 0;
1  1  0  0 : 0 : 1;
0  0  1  0 : 0 : 0;
1  0  1  0 : 0 : 1;
0  1  1  0 : 0 : 1;
1  1  1  0 : 0 : 1;
0  0  0  1 : 0 : 0;
1  0  0  1 : 0 : 1;
0  1  0  1 : 0 : 1;
1  1  0  1 : 0 : 1;
0  0  1  1 : 0 : 0;
1  0  1  1 : 0 : 1;
0  1  1  1 : 0 : 1;
1  1  1  1 : ? : 1;
0  0  0  0 : ? : 0;
1  0  0  0 : 1 : 1;
0  1  0  0 : 1 : 1;
1  1  0  0 : 1 : 1;
0  0  1  0 : 1 : 1;
1  0  1  0 : 1 : 1;
0  1  1  0 : 1 : 1;
1  1  1  0 : 1 : 1;
0  0  0  1 : 1 : 1;
1  0  0  1 : 1 : 1;
0  1  0  1 : 1 : 1;
1  1  0  1 : 1 : 1;
0  0  1  1 : 1 : 1;
1  0  1  1 : 1 : 1;
0  1  1  1 : 1 : 1;
1  1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH54W22 NCL-UDP
//// A has a weight of 2
//// B has a weight of 2
//// 21. Z = AB + ACD + BCD       LDD notation
//// {[A, B, C], [A, B, D]} -> Z  flow notation
/////////////////////// 
module TH54W22(Z,A,B,C,D);
output Z;
input A, B, C, D;
   TH54W22P #50 (Z,A,B,C,D);
endmodule
primitive TH54W22P(Z,A,B,C,D);
output Z;
input A, B, C, D;
reg Z;
initial Z = 1'b0;
table
// A B C D : Z : Z+         
0  0  0  0 : ? : 0;
1  0  0  0 : 0 : 0;
0  1  0  0 : 0 : 0;
1  1  0  0 : 0 : 0;
0  0  1  0 : 0 : 0;
1  0  1  0 : 0 : 0;
0  1  1  0 : 0 : 0;
1  1  1  0 : 0 : 1;
0  0  0  1 : 0 : 0;
1  0  0  1 : 0 : 0;
0  1  0  1 : 0 : 0;
1  1  0  1 : 0 : 1;
0  0  1  1 : 0 : 0;
1  0  1  1 : 0 : 0;
0  1  1  1 : 0 : 0;
1  1  1  1 : ? : 1;
0  0  0  0 : ? : 0;
1  0  0  0 : 1 : 1;
0  1  0  0 : 1 : 1;
1  1  0  0 : 1 : 1;
0  0  1  0 : 1 : 1;
1  0  1  0 : 1 : 1;
0  1  1  0 : 1 : 1;
1  1  1  0 : 1 : 1;
0  0  0  1 : 1 : 1;
1  0  0  1 : 1 : 1;
0  1  0  1 : 1 : 1;
1  1  0  1 : 1 : 1;
0  0  1  1 : 1 : 1;
1  0  1  1 : 1 : 1;
0  1  1  1 : 1 : 1;
1  1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH44W22 NCL-UDP
//// A has a weight of 2
//// B has a weight of 2
//// 20. Z = AB + ACD + BCD               LDD notation
//// {[A, B], [A, C, D], [B, C, D]} -> Z  flow notation
/////////////////////// 
module TH44W22(Z,A,B,C,D);
output Z;
input A, B, C, D;
   TH44W22P #50 (Z,A,B,C,D);
endmodule
primitive TH44W22P(Z,A,B,C,D);
output Z;
input A, B, C, D;
reg Z;
initial Z = 1'b0;
table
// A B C D : Z : Z+         
0  0  0  0 : ? : 0;
1  0  0  0 : 0 : 0;
0  1  0  0 : 0 : 0;
1  1  0  0 : 0 : 1;
0  0  1  0 : 0 : 0;
1  0  1  0 : 0 : 0;
0  1  1  0 : 0 : 0;
1  1  1  0 : 0 : 0;
0  0  0  1 : 0 : 0;
1  0  0  1 : 0 : 0;
0  1  0  1 : 0 : 0;
1  1  0  1 : 0 : 0;
0  0  1  1 : 0 : 0;
1  0  1  1 : 0 : 1;
0  1  1  1 : 0 : 1;
1  1  1  1 : ? : 1;
0  0  0  0 : ? : 0;
1  0  0  0 : 1 : 1;
0  1  0  0 : 1 : 1;
1  1  0  0 : 1 : 1;
0  0  1  0 : 1 : 1;
1  0  1  0 : 1 : 1;
0  1  1  0 : 1 : 1;
1  1  1  0 : 1 : 1;
0  0  0  1 : 1 : 1;
1  0  0  1 : 1 : 1;
0  1  0  1 : 1 : 1;
1  1  0  1 : 1 : 1;
0  0  1  1 : 1 : 1;
1  0  1  1 : 1 : 1;
0  1  1  1 : 1 : 1;
1  1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH34W22 NCL-UDP
//// A has a weight of 2
//// B has a weight of 2
//// 19. Z = AB + AC + AD + BC + BD                 LDD notation
//// {[A, B], [A, C], [A, D], [B, C], [B, D]} -> Z  flow notation
/////////////////////// 
module TH34W22(Z,A,B,C,D);
output Z;
input A, B, C, D;
   TH34W22P #50 (Z,A,B,C,D);
endmodule
primitive TH34W22P(Z,A,B,C,D);
output Z;
input A, B, C, D;
reg Z;
initial Z = 1'b0;
table
// A B C D : Z : Z+         
0  0  0  0 : ? : 0;
1  0  0  0 : 0 : 0;
0  1  0  0 : 0 : 0;
1  1  0  0 : 0 : 1;
0  0  1  0 : 0 : 0;
1  0  1  0 : 0 : 1;
0  1  1  0 : 0 : 1;
1  1  1  0 : 0 : 1;
0  0  0  1 : 0 : 0;
1  0  0  1 : 0 : 1;
0  1  0  1 : 0 : 1;
1  1  0  1 : 0 : 1;
0  0  1  1 : 0 : 0;
1  0  1  1 : 0 : 1;
0  1  1  1 : 0 : 1;
1  1  1  1 : ? : 1;
0  0  0  0 : ? : 0;
1  0  0  0 : 1 : 1;
0  1  0  0 : 1 : 1;
1  1  0  0 : 1 : 1;
0  0  1  0 : 1 : 1;
1  0  1  0 : 1 : 1;
0  1  1  0 : 1 : 1;
1  1  1  0 : 1 : 1;
0  0  0  1 : 1 : 1;
1  0  0  1 : 1 : 1;
0  1  0  1 : 1 : 1;
1  1  0  1 : 1 : 1;
0  0  1  1 : 1 : 1;
1  0  1  1 : 1 : 1;
0  1  1  1 : 1 : 1;
1  1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH24W22 NCL-UDP
//// A has a weight of 2
//// B has a weight of 2
//// 18. Z = A + B + CD      LDD notation
//// {A,  B, [C, D]} -> Z    flow notation
/////////////////////// 
module TH24W22(Z,A,B,C,D);
output Z;
input A, B, C, D;
   TH24W22P #50 (Z,A,B,C,D);
endmodule
primitive TH24W22P(Z,A,B,C,D);
output Z;
input A, B, C, D;
reg Z;
initial Z = 1'b0;
table
// A B C D : Z : Z+         
0  0  0  0 : ? : 0;
1  0  0  0 : 0 : 1;
0  1  0  0 : 0 : 1;
1  1  0  0 : 0 : 1;
0  0  1  0 : 0 : 0;
1  0  1  0 : 0 : 1;
0  1  1  0 : 0 : 1;
1  1  1  0 : 0 : 1;
0  0  0  1 : 0 : 0;
1  0  0  1 : 0 : 1;
0  1  0  1 : 0 : 1;
1  1  0  1 : 0 : 1;
0  0  1  1 : 0 : 1;
1  0  1  1 : 0 : 1;
0  1  1  1 : 0 : 1;
1  1  1  1 : ? : 1;
0  0  0  0 : ? : 0;
1  0  0  0 : 1 : 1;
0  1  0  0 : 1 : 1;
1  1  0  0 : 1 : 1;
0  0  1  0 : 1 : 1;
1  0  1  0 : 1 : 1;
0  1  1  0 : 1 : 1;
1  1  1  0 : 1 : 1;
0  0  0  1 : 1 : 1;
1  0  0  1 : 1 : 1;
0  1  0  1 : 1 : 1;
1  1  0  1 : 1 : 1;
0  0  1  1 : 1 : 1;
1  0  1  1 : 1 : 1;
0  1  1  1 : 1 : 1;
1  1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH44W3 NCL-UDP
//// A has a weight of 3
//// 17. Z = AB + AC + AD          LDD notation
//// {[A, B] [A, C], [A, D]} -> Z  flow notation
/////////////////////// 
module TH44W3(Z,A,B,C,D);
output Z;
input A, B, C, D;
   TH44W3P #50 (Z,A,B,C,D);
endmodule
primitive TH44W3P(Z,A,B,C,D);
output Z;
input A, B, C, D;
reg Z;
initial Z = 1'b0;
table
// A B C D : Z : Z+         
0  0  0  0 : ? : 0;
1  0  0  0 : 0 : 0;
0  1  0  0 : 0 : 0;
1  1  0  0 : 0 : 1;
0  0  1  0 : 0 : 0;
1  0  1  0 : 0 : 1;
0  1  1  0 : 0 : 0;
1  1  1  0 : 0 : 1;
0  0  0  1 : 0 : 0;
1  0  0  1 : 0 : 1;
0  1  0  1 : 0 : 0;
1  1  0  1 : 0 : 1;
0  0  1  1 : 0 : 0;
1  0  1  1 : 0 : 1;
0  1  1  1 : 0 : 1;
1  1  1  1 : ? : 1;
0  0  0  0 : ? : 0;
1  0  0  0 : 1 : 1;
0  1  0  0 : 1 : 1;
1  1  0  0 : 1 : 1;
0  0  1  0 : 1 : 1;
1  0  1  0 : 1 : 1;
0  1  1  0 : 1 : 1;
1  1  1  0 : 1 : 1;
0  0  0  1 : 1 : 1;
1  0  0  1 : 1 : 1;
0  1  0  1 : 1 : 1;
1  1  0  1 : 1 : 1;
0  0  1  1 : 1 : 1;
1  0  1  1 : 1 : 1;
0  1  1  1 : 1 : 1;
1  1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH34W3 NCL-UDP
//// A has a weight of 3
//// 16. Z = A + BCD      LDD notation
//// {A, [B, C, D]} -> Z  flow notation
/////////////////////// 
module TH34W3(Z,A,B,C,D);
output Z;
input A, B, C, D;
   TH34W3P #50 (Z,A,B,C,D);
endmodule
primitive TH34W3P(Z,A,B,C,D);
output Z;
input A, B, C, D;
reg Z;
initial Z = 1'b0;
table
// A B C D : Z : Z+         
0  0  0  0 : ? : 0;
1  0  0  0 : 0 : 1;
0  1  0  0 : 0 : 0;
1  1  0  0 : 0 : 1;
0  0  1  0 : 0 : 0;
1  0  1  0 : 0 : 1;
0  1  1  0 : 0 : 0;
1  1  1  0 : 0 : 1;
0  0  0  1 : 0 : 0;
1  0  0  1 : 0 : 1;
0  1  0  1 : 0 : 0;
1  1  0  1 : 0 : 1;
0  0  1  1 : 0 : 0;
1  0  1  1 : 0 : 1;
0  1  1  1 : 0 : 1;
1  1  1  1 : ? : 1;
0  0  0  0 : ? : 0;
1  0  0  0 : 1 : 1;
0  1  0  0 : 1 : 1;
1  1  0  0 : 1 : 1;
0  0  1  0 : 1 : 1;
1  0  1  0 : 1 : 1;
0  1  1  0 : 1 : 1;
1  1  1  0 : 1 : 1;
0  0  0  1 : 1 : 1;
1  0  0  1 : 1 : 1;
0  1  0  1 : 1 : 1;
1  1  0  1 : 1 : 1;
0  0  1  1 : 1 : 1;
1  0  1  1 : 1 : 1;
0  1  1  1 : 1 : 1;
1  1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH44W2 NCL-UDP
//// A has a weight of 2
//// 15. Z = ABC + ABD + ACD                 LDD notation
//// {[A, B, C], [A, B, D], [A, C, D]} -> Z  flow notation
/////////////////////// 
module TH44W2(Z,A,B,C,D);
output Z;
input A, B, C, D;
   TH44W2P #50 (Z,A,B,C,D);
endmodule
primitive TH44W2P(Z,A,B,C,D);
output Z;
input A, B, C, D;
reg Z;
initial Z = 1'b0;
table
// A B C D : Z : Z+         
0  0  0  0 : ? : 0;
1  0  0  0 : 0 : 0;
0  1  0  0 : 0 : 0;
1  1  0  0 : 0 : 0;
0  0  1  0 : 0 : 0;
1  0  1  0 : 0 : 0;
0  1  1  0 : 0 : 0;
1  1  1  0 : 0 : 1;
0  0  0  1 : 0 : 0;
1  0  0  1 : 0 : 0;
0  1  0  1 : 0 : 0;
1  1  0  1 : 0 : 1;
0  0  1  1 : 0 : 0;
1  0  1  1 : 0 : 1;
0  1  1  1 : 0 : 0;
1  1  1  1 : ? : 1;
0  0  0  0 : ? : 0;
1  0  0  0 : 1 : 1;
0  1  0  0 : 1 : 1;
1  1  0  0 : 1 : 1;
0  0  1  0 : 1 : 1;
1  0  1  0 : 1 : 1;
0  1  1  0 : 1 : 1;
1  1  1  0 : 1 : 1;
0  0  0  1 : 1 : 1;
1  0  0  1 : 1 : 1;
0  1  0  1 : 1 : 1;
1  1  0  1 : 1 : 1;
0  0  1  1 : 1 : 1;
1  0  1  1 : 1 : 1;
0  1  1  1 : 1 : 1;
1  1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH34W2 NCL-UDP
//// A has a weight of 2
//// 14. Z = AB + AC + AD + BCD                 LDD notation
//// {[A, B], [A, C], [A, D], [B, C, D]} -> Z   flow notation
/////////////////////// 
module TH34W2(Z,A,B,C,D);
output Z;
input A, B, C, D;
   TH34W2P #50 (Z,A,B,C,D);
endmodule
primitive TH34W2P(Z,A,B,C,D);
output Z;
input A, B, C, D;
reg Z;
initial Z = 1'b0;
table
// A B C D : Z : Z+         
0  0  0  0 : ? : 0;
1  0  0  0 : 0 : 0;
0  1  0  0 : 0 : 0;
1  1  0  0 : 0 : 1;
0  0  1  0 : 0 : 0;
1  0  1  0 : 0 : 1;
0  1  1  0 : 0 : 0;
1  1  1  0 : 0 : 0;
0  0  0  1 : 0 : 0;
1  0  0  1 : 0 : 1;
0  1  0  1 : 0 : 0;
1  1  0  1 : 0 : 0;
0  0  1  1 : 0 : 0;
1  0  1  1 : 0 : 0;
0  1  1  1 : 0 : 1;
1  1  1  1 : ? : 1;
0  0  0  0 : ? : 0;
1  0  0  0 : 1 : 1;
0  1  0  0 : 1 : 1;
1  1  0  0 : 1 : 1;
0  0  1  0 : 1 : 1;
1  0  1  0 : 1 : 1;
0  1  1  0 : 1 : 1;
1  1  1  0 : 1 : 1;
0  0  0  1 : 1 : 1;
1  0  0  1 : 1 : 1;
0  1  0  1 : 1 : 1;
1  1  0  1 : 1 : 1;
0  0  1  1 : 1 : 1;
1  0  1  1 : 1 : 1;
0  1  1  1 : 1 : 1;
1  1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH24W2 NCL-UDP
//// A has a weight of 2
//// 13. Z = A + BC + BD + CD            LDD notation
//// {[A, [B, C], [B, D], [B, C]} -> Z   flow notation
/////////////////////// 
module TH24W2(Z,A,B,C,D);
output Z;
input A, B, C, D;
   TH24W2P #50 (Z,A,B,C,D);
endmodule
primitive TH24W2P(Z,A,B,C,D);
output Z;
input A, B, C, D;
reg Z;
initial Z = 1'b0;
table
// A B C D : Z : Z+         
0  0  0  0 : ? : 0;
1  0  0  0 : 0 : 0;
0  1  0  0 : 0 : 0;
1  1  0  0 : 0 : 0;
0  0  1  0 : 0 : 0;
1  0  1  0 : 0 : 0;
0  1  1  0 : 0 : 0;
1  1  1  0 : 0 : 0;
0  0  0  1 : 0 : 0;
1  0  0  1 : 0 : 0;
0  1  0  1 : 0 : 0;
1  1  0  1 : 0 : 0;
0  0  1  1 : 0 : 0;
1  0  1  1 : 0 : 0;
0  1  1  1 : 0 : 0;
1  1  1  1 : ? : 1;
0  0  0  0 : ? : 0;
1  0  0  0 : 1 : 1;
0  1  0  0 : 1 : 1;
1  1  0  0 : 1 : 1;
0  0  1  0 : 1 : 1;
1  0  1  0 : 1 : 1;
0  1  1  0 : 1 : 1;
1  1  1  0 : 1 : 1;
0  0  0  1 : 1 : 1;
1  0  0  1 : 1 : 1;
0  1  0  1 : 1 : 1;
1  1  0  1 : 1 : 1;
0  0  1  1 : 1 : 1;
1  0  1  1 : 1 : 1;
0  1  1  1 : 1 : 1;
1  1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH44 NCL-UDP
//// 12. Z = ABCD          LDD notation
//// [A, B, C, D] -> Z     flow notation
/////////////////////// 
module TH44(Z,A,B,C,D);
output Z;
input A, B, C, D;
   TH44P #50 (Z,A,B,C,D);
endmodule
primitive TH44P(Z,A,B,C,D);
output Z;
input A, B, C, D;
reg Z;
initial Z = 1'b0;
table
// A B C D : Z : Z+         
0  0  0  0 : ? : 0;
1  0  0  0 : 0 : 0;
0  1  0  0 : 0 : 0;
1  1  0  0 : 0 : 0;
0  0  1  0 : 0 : 0;
1  0  1  0 : 0 : 0;
0  1  1  0 : 0 : 0;
1  1  1  0 : 0 : 0;
0  0  0  1 : 0 : 0;
1  0  0  1 : 0 : 0;
0  1  0  1 : 0 : 0;
1  1  0  1 : 0 : 0;
0  0  1  1 : 0 : 0;
1  0  1  1 : 0 : 0;
0  1  1  1 : 0 : 0;
1  1  1  1 : ? : 1;
0  0  0  0 : ? : 0;
1  0  0  0 : 1 : 1;
0  1  0  0 : 1 : 1;
1  1  0  0 : 1 : 1;
0  0  1  0 : 1 : 1;
1  0  1  0 : 1 : 1;
0  1  1  0 : 1 : 1;
1  1  1  0 : 1 : 1;
0  0  0  1 : 1 : 1;
1  0  0  1 : 1 : 1;
0  1  0  1 : 1 : 1;
1  1  0  1 : 1 : 1;
0  0  1  1 : 1 : 1;
1  0  1  1 : 1 : 1;
0  1  1  1 : 1 : 1;
1  1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH34 NCL-UDP
//// 11. Z = ABC + ABD + ACD + BCD                       LDD notation
//// {[A, B, C], [A, B, D], [A, C, D], [B, C, D]} -> Z   flow notation
/////////////////////// 
module TH34(Z,A,B,C,D);
output Z;
input A, B, C, D;
   TH34P #50 (Z,A,B,C,D);
endmodule
primitive TH34P(Z,A,B,C,D);
output Z;
input A, B, C, D;
reg Z;
initial Z = 1'b0;
table
// A B C D : Z : Z+         
0  0  0  0 : ? : 0;
1  0  0  0 : 0 : 0;
0  1  0  0 : 0 : 0;
1  1  0  0 : 0 : 0;
0  0  1  0 : 0 : 0;
1  0  1  0 : 0 : 0;
0  1  1  0 : 0 : 0;
1  1  1  0 : 0 : 1;
0  0  0  1 : 0 : 0;
1  0  0  1 : 0 : 0;
0  1  0  1 : 0 : 0;
1  1  0  1 : 0 : 1;
0  0  1  1 : 0 : 0;
1  0  1  1 : 0 : 1;
0  1  1  1 : 0 : 1;
1  1  1  1 : ? : 1;
0  0  0  0 : ? : 0;
1  0  0  0 : 1 : 1;
0  1  0  0 : 1 : 1;
1  1  0  0 : 1 : 1;
0  0  1  0 : 1 : 1;
1  0  1  0 : 1 : 1;
0  1  1  0 : 1 : 1;
1  1  1  0 : 1 : 1;
0  0  0  1 : 1 : 1;
1  0  0  1 : 1 : 1;
0  1  0  1 : 1 : 1;
1  1  0  1 : 1 : 1;
0  0  1  1 : 1 : 1;
1  0  1  1 : 1 : 1;
0  1  1  1 : 1 : 1;
1  1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH24 NCL-UDP
//// 10. Z = AB + AC + AD + BC + BD + CD                      LDD notation
//// {[A, B], [A, C], [A, D], [B, C], [B, D], [C, D]} -> Z    flow notation
/////////////////////// 
module TH24(Z,A,B,C,D);
output Z;
input A, B, C, D;
   TH24P #50 (Z,A,B,C,D);
endmodule
primitive TH24P(Z,A,B,C,D);
output Z;
input A, B, C, D;
reg Z;
initial Z = 1'b0;
table
// A B C D : Z : Z+         
0  0  0  0 : ? : 0;
1  0  0  0 : 0 : 0;
0  1  0  0 : 0 : 0;
1  1  0  0 : 0 : 1;
0  0  1  0 : 0 : 0;
1  0  1  0 : 0 : 1;
0  1  1  0 : 0 : 1;
1  1  1  0 : 0 : 1;
0  0  0  1 : 0 : 0;
1  0  0  1 : 0 : 1;
0  1  0  1 : 0 : 1;
1  1  0  1 : 0 : 1;
0  0  1  1 : 0 : 1;
1  0  1  1 : 0 : 1;
0  1  1  1 : 0 : 1;
1  1  1  1 : ? : 1;
0  0  0  0 : ? : 0;
1  0  0  0 : 1 : 1;
0  1  0  0 : 1 : 1;
1  1  0  0 : 1 : 1;
0  0  1  0 : 1 : 1;
1  0  1  0 : 1 : 1;
0  1  1  0 : 1 : 1;
1  1  1  0 : 1 : 1;
0  0  0  1 : 1 : 1;
1  0  0  1 : 1 : 1;
0  1  0  1 : 1 : 1;
1  1  0  1 : 1 : 1;
0  0  1  1 : 1 : 1;
1  0  1  1 : 1 : 1;
0  1  1  1 : 1 : 1;
1  1  1  1 : ? : 1;
endtable
endprimitive


/////////////////////// 
//// TH14 NCL-UDP
//// 9. Z = A + B + C + D    LDD notation
//// {A, B, C, D} -> Z       flow notation
/////////////////////// 
module TH14(Z,A,B,C,D);
output Z;
input A, B, C, D;
  TH14P #50 (Z,A,B,C,D);
endmodule
primitive TH14P(Z,A,B,C,D);
output Z;
input A, B, C, D;
table
// A B C D : Z          
0  0  0  0 : 0;
1  0  0  0 : 1;
0  1  0  0 : 1;
1  1  0  0 : 1;
0  0  1  0 : 1;
1  0  1  0 : 1;
0  1  1  0 : 1;
1  1  1  0 : 1;
0  0  0  1 : 1;
1  0  0  1 : 1;
0  1  0  1 : 1;
1  1  0  1 : 1;
0  0  1  1 : 1;
1  0  1  1 : 1;
0  1  1  1 : 1;
1  1  1  1 : 1;
endtable
endprimitive

/////////////////////// 
//// TH33W2 NCL-UDP
//// A has a weight of 2
//// 8. Z = AB + AC          LDD notation
//// {[A, B], [A, C]} -> Z   flow notation
/////////////////////// 
/////////////////////// 
module TH33W2(Z,A,B,C);
output Z;
input A, B, C;
   TH33W2P #40 (Z,A,B,C);
endmodule
primitive TH33W2P(Z,A,B,C);
output Z;
input A, B, C;
reg Z;
initial Z = 1'b0;
table
// A B C : Z : Z+         
0  0  0 : ? : 0;
1  0  0 : 0 : 0;
0  1  0 : 0 : 0;
1  1  0 : 0 : 1;
0  0  1 : 0 : 0;
1  0  1 : 0 : 1;
0  1  1 : 0 : 0;
1  1  1 : ? : 1;
0  0  0 : ? : 0;
1  0  0 : 1 : 1;
0  1  0 : 1 : 1;
1  1  0 : 1 : 1;
0  0  1 : 1 : 1;
1  0  1 : 1 : 1;
0  1  1 : 1 : 1;
1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH23W2 NCL-UDP
//// A has a weight of 2
//// 7. Z = A + BC        LDD notation
//// {A, [B, C]} -> Z     flow notation
/////////////////////// 
/////////////////////// 
module TH23W2(Z,A,B,C);
output Z;
input A, B, C;
   TH23W2P #40 (Z,A,B,C);
endmodule
primitive TH23W2P(Z,A,B,C);
output Z;
input A, B, C;
reg Z;
initial Z = 1'b0;
table
// A B C : Z : Z+         
0  0  0 : ? : 0;
1  0  0 : 0 : 1;
0  1  0 : 0 : 0;
1  1  0 : 0 : 1;
0  0  1 : 0 : 0;
1  0  1 : 0 : 1;
0  1  1 : 0 : 1;
1  1  1 : ? : 1;
0  0  0 : ? : 0;
1  0  0 : 1 : 1;
0  1  0 : 1 : 1;
1  1  0 : 1 : 1;
0  0  1 : 1 : 1;
1  0  1 : 1 : 1;
0  1  1 : 1 : 1;
1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH33 NCL-UDP
//// 6. Z = ABC         LDD notation
//// [A, B, C] -> Z     flow notation
/////////////////////// 
/////////////////////// 
module TH33(Z,A,B,C);
output Z;
input A, B, C;
   TH33P #40 (Z,A,B,C);
endmodule
primitive TH33P(Z,A,B,C);
output Z;
input A, B, C;
reg Z;
initial Z = 1'b0;
table
// A B C : Z : Z+         
0  0  0 : ? : 0;
1  0  0 : 0 : 0;
0  1  0 : 0 : 0;
1  1  0 : 0 : 0;
0  0  1 : 0 : 0;
1  0  1 : 0 : 0;
0  1  1 : 0 : 0;
1  1  1 : ? : 1;
0  0  0 : ? : 0;
1  0  0 : 1 : 1;
0  1  0 : 1 : 1;
1  1  0 : 1 : 1;
0  0  1 : 1 : 1;
1  0  1 : 1 : 1;
0  1  1 : 1 : 1;
1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH23 NCL-UDP
//// . Z = AB + BC + AC             LDD notation
//// {[A, B], [B, C], [A, C]} -> Z   flow notation
/////////////////////// 
module TH23(Z,A,B,C);
output Z;
input A, B, C;
  TH23P #40 (Z,A,B,C);
endmodule
primitive TH23P(Z,A,B,C);
output Z;
input A, B, C;
reg Z;
initial Z = 1'b0;
table
// A B C : Z  
0  0  0 : ? : 0;
1  0  0 : 0 : 0;
0  1  0 : 0 : 0;
1  1  0 : 0 : 1;
0  0  1 : 0 : 0;
1  0  1 : 0 : 1;
0  1  1 : 0 : 1;
1  1  1 : 0 : 1;
0  0  0 : ? : 0;
1  0  0 : 1 : 1;
0  1  0 : 1 : 1;
1  1  0 : ? : 1;
0  0  1 : 1 : 1;
1  0  1 : 1 : 1;
0  1  1 : 1 : 1;
1  1  1 : ? : 1;
endtable
endprimitive

/////////////////////// 
//// TH13 NCL-UDP
//// 4. Z = A + B + C     LDD notation
//// {A, B, C} -> Z       flow notation
/////////////////////// 
module TH13(Z,A,B,C);
output Z;
input A, B, C;
  TH13P #40 (Z,A,B,C);
endmodule
primitive TH13P(Z,A,B,C);
output Z;
input A, B, C;
table
// A B C : Z          
0  0  0 : 0;
1  0  0 : 1;
0  1  0 : 1;
1  1  0 : 1;
0  0  1 : 1;
1  0  1 : 1;
0  1  1 : 1;
1  1  1 : 1;
endtable
endprimitive

/////////////////////// 
//// TH22N NCL-UDP
//// initialize to null
//// 3. Z = AB       LDD notation
//// [A, B] -> Z     flow notation
/////////////////////// 
module TH22N(Z,A,B,C);
output Z;
input A,B;
input C;  // init rail
   TH22NP #30 (Z,A,B,C);
endmodule
primitive TH22NP(Z,A,B,C);
output Z;
input A,B;
input C;  // init rail
reg Z;
initial Z = 1'b0;
table
// A B C : Z : Z+         
?  ?  1 : ? : 0;
?  ?  1 : ? : 0;
?  ?  1 : ? : 0;
?  ?  1 : ? : 0;
?  ?  1 : ? : 0;
?  ?  1 : ? : 0;
?  ?  1 : ? : 0;
?  ?  1 : ? : 0;
0  0  0 : 0 : 0;
0  1  0 : 0 : 0;
1  0  0 : 0 : 0;
1  1  0 : 0 : 1;
0  0  0 : 1 : 0;
0  1  0 : 1 : 1;
1  0  0 : 1 : 1;
1  1  0 : 1 : 1;
endtable
endprimitive

/////////////////////// 
//// TH22D NCL-UDP
//// initialize to data
//// 3. Z = AB       LDD notation
//// [A, B] -> Z     flow notation
/////////////////////// 
module TH22D(Z,A,B,C);
output Z;
input A, B;
input C;  // init rail
   TH22DP #30 (Z,A,B,C);
endmodule
primitive TH22DP(Z,A,B,C);
output Z;
input A, B;
input C;  // init rail
reg Z;
initial Z = 1'b0;
table
// A B C : Z : Z+         
?  ?  1 : ? : 1;
?  ?  1 : ? : 1;
?  ?  1 : ? : 1;
?  ?  1 : ? : 1;
?  ?  1 : ? : 1;
?  ?  1 : ? : 1;
?  ?  1 : ? : 1;
?  ?  1 : ? : 1;
0  0  0 : 0 : 0;
1  0  0 : 0 : 0;
0  1  0 : 0 : 0;
1  1  0 : 0 : 1;
0  0  0 : 1 : 0;
1  0  0 : 1 : 1;
0  1  0 : 1 : 1;
1  1  0 : 1 : 1;
endtable
endprimitive

/////////////////////// 
//// TH22 NCL-UDP
//// 3. Z = AB       LDD notation
//// [A, B] -> Z     flow notation
/////////////////////// 
module TH22(Z,A,B);
output Z;
input A,B;
   TH22P #30 (Z,A,B);
endmodule
primitive TH22P(Z,A,B);
output Z;
input A, B;
reg Z;
initial Z = 1'b0;
table
// A B : Z : Z+         
0  0 : ? : 0;
0  1 : 0 : 0;
1  0 : 0 : 0;
1  1 : 0 : 1;
0  0 : ? : 0;
0  1 : 1 : 1;
1  0 : 1 : 1;
1  1 : 1 : 1;
endtable
endprimitive

/////////////////////// 
//// TH12 NCL-UDP
//// 2. Z = A + B    LDD notation
//// {A, B} -> Z     flow notation
/////////////////////// 
module TH12(Z,A,B);
output Z;
input A, B;
  TH12P #30 (Z,A,B);
endmodule
primitive TH12P(Z,A,B);
output Z;
input A, B;
table
// A B : Z          
0  0 : 0;
1  0 : 1;
0  1 : 1;
1  1 : 1;
endtable
endprimitive

/////////////////////// 
//// THnotN NCL-UDP
/////////////////////// 
module THnotN(Z, A, B);
output Z;
input A;
input B;  // init rail
   THnotNP #20 (Z, A, B);
endmodule
primitive THnotNP(Z,A,B);
output Z;
input A;
input B;  // init rail
table
// A B : Z          
?  1 : 0;
?  1 : 0;
0  0 : 1;
1  0 : 0;
endtable
endprimitive

/////////////////////// 
//// not NCL-UDP
/////////////////////// 
module THnot(Z, A);
output Z;
input A;
   THnotP #20 (Z, A);
endmodule
primitive THnotP(Z,A);
output Z;
input A;
table
// A B : Z          
0  : 1;
1  : 0;
endtable
endprimitive

/// library components////////////
/*
module Pipecomponent(output [rails-1:0] Z, input ZCOMP, input [rails-1:0] A, output ACOMP, input init);
parameter rails=2;
wire [rails-1:0] t0;
wire enable, ACOMPtemp;
genvar i:
THnotN  u0(enable, ZCOMP, init);
generate
for (i=0, i<rails, i=i+1) begin : pipestage
TH22  ps (t0[i], A[i], enable);
end
endgenerate
assign Z = t0[rails:0];
begin
   case (rails)
     1 : assign ACOMPtemp = t0[0]
     2 : TH12 u5 (ACOMPtemp, t0[0], t0[1]);
     3 : TH13 u6 (ACOMPtemp, t0[0], t0[1], t0[2]);
     4 : TH14 u7 (ACOMPtemp, t0[0], t0[1], t0[2], t0[3]);
   endcase
end

assign ACOMP = ACOMPtemp;
endmodule
*/