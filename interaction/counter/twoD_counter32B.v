/////////////////////////////
// NCL sandbox
// 32 bit counter with two D completeness pipelining
// Karl Fant July 2015
/////////////////////////////
`timescale 10ps / 1ps

module twoD_counter32B;

 /* Make an init that pulses once. */
  reg init = 1;
  initial begin
     # 1200 init = 0;
     # 5000 $stop;
  end
initial
 begin
    $dumpfile("twoD_counter32B.vcd");
    $dumpvars(0, twoD_counter32B);
 end

////// test bench
//////////////////////////////
////// circuit under test

wire [32:0] countcomp;
wire [31:0] sumcomp;
wire [1:0] sum [31:0];
wire [1:0] carry [32:0];
genvar i;

// 4 stage pipeline
THnotN  u0(carry[0][1], countcomp[0], init); // auto produce carryin
assign carry[0][0] = 1'b0; // auto produce carryin

// two D digit pipelined counter
for (i=0; i<32; i=i+1) begin
  twoD_counter_ringB ci (sum[i][1:0], carry[i+1][1:0], sumcomp[i], countcomp[i+1], carry[i][1:0], countcomp[i], init);
  TH12 u3 (sumcomp[i], sum[i][1], sum[i][0]);  // auto consume sum
end

TH12 u33 (countcomp[32], carry[32][0], carry[32][1]);  // auto consume carryout

////// circuit under test
//////////////////////////////
////// test bench

wire [63:0] displaysum;
genvar k;
for (k=0; k<32; k=k+1) begin
assign displaysum[2*k+1] = sum[k][1];
assign displaysum[2*k] = sum[k][0];
end

endmodule