////////////////////////////////////
// Write the loss function        
////////////////////////////////////
//Calculate the loss against a set of data
// Int Input: X0: arraySize, X1: array address
// Single Input: S0: m, S1: c

loss:
	// LDA X1, array
	LDA X6, inverseN
	ADD X2, X1, XZR // X2 holdes array address
	LDA X7, zeroFloat // load in 0 float value
	LDURS S7, [X7, #0] 
	ADDI X5, XZR, #0 // counter

loop2:	LDURS S3, [X2, #0] // loads x-value dataset[i][0]
	LDURS S4, [X2, #4] // loads y-value dataset[i][1]
	FMULS  S4, S4, S0   // m*xi 
	FADDS S4, S4, S1   // +c
	FSUBS  S4, S3, S4   // y-yexpected
	FMULS  S4, S4, S4   // pow(base,2) 
	ADDI X5, X5, #1
	ADDI X2, X2, #8
	FADDS  S7, S7, S4
	B check

check: SUBS XZR, X5, X0
       B.LT loop2
       LDURS S4, [X6,#0]
       FMULS  S7, S7, S4
       BR LR


////////////////////////////////////
// Write the train function //
////////////////////////////////////
// Int Input: X0: arraySize, X1: array address, X2: epoch number
// Single Input: S0: m, S1: c, S2: learning rate,  S3: -2/arraySize
// Output: S0: m, S1: c, S2: loss
train:
    //Initialize stack frame for needed variables

	//LDURS S7, [X3, #0] 
	LDURS S0, [X3, #0] // set M to 0.0
	LDURS S1, [X3, #0] // set c to 0.0
	LDA   X4, epsilon
	LDURS S4, [X4, #0] // setting epsilon to S4
	ADDI  X5, XZR, #0  // counter


loop1: 
	
	LDURS S4, [X3, #0] // set D_m = 0
	LDURS S5, [X3, #0] // set D_c = 0
	ADDI  X6, XZR, #0  // counter for loop 3 
	ADDI  X7, X1, #0   // array address counter
loop3: 
	LDURS S6, [X7, #0] // loads dataset [j][0]
	LDURS S8, [X7, #4] // loads dataset[j][1]

	ADDI  X7, X7, #8   // sets up X7 for next iteration. 
	FMULS S9, S0, S6   // M*dataset [j][0]
	FADDS S9, S9, S1   //  M*dataset [j][0]+C
	FSUBS S9, S8, S9   //  ( dataset[j][1] - (M*dataset[j][0] + C ) ) 
	FMULS S9, S9, S6   // dataset[j][0] * ( dataset[j][1] - (M*dataset[j][0] + C ) ) 
	FADDS S4, S4, S9   // D_m += dataset[j][0] * ( dataset[j][1] - (M*dataset[j][0] + C ) ) 
	FMULS S9, S0, S6   // M*dataset[j][0] 
	FADDS S9, S9, S1   // M*dataset[j][0] + c
	FSUBS S9, S8, S9   // dataset[j][1] - ( M*dataset[j][0] + C ) 
	FADDS S5, S5, S9   // D_C += dataset[j][1] - ( M*dataset[j][0] + C ) 
	ADDI  X6, X6, #1   // COUNTER +1
	SUBS  XZR, X0, X6  // checker 
	B.GT   loop3
	 
	
	FMULS S4, S3, S4   // D_m *= -2/dataset.size()
	FMULS S5, S3, S5   // D_c *= -2/dataset.size() 
	FMULS S9, S4, S2   // lr *D_m  
	FSUBS S0, S0, S9   // M - lr *D_m /
	FMULS S9, S5, S2   // lr*D_c 
	FSUBS S1, S1, S9   // C = C - lr*D_c  
	
	SUBI SP, SP, #80 // the number depends on how many instructions we wanna save (40 is temp for 5)
	STUR FP [SP, #72]
	STUR LR [SP, #64] 
	STUR X6 [SP, #8]
	STUR X7 [SP, #16]
	STURS S4 [SP, #24]
	STUR X5 [SP, #32]
	STURS S2 [SP, #40]
	STURS S3 [SP, #48] 
	STUR X2 [SP, #56] 
	ADDI FP, SP, #80
	
	BL loss 
	
	LDUR FP [SP, #72]
	LDUR LR [SP, #64]
	LDUR X6 [SP, #8]
	LDUR X7 [SP, #16]
	LDURS S4 [SP, #24]
	LDUR X5 [SP, #32]
	LDURS S2 [SP, #40]
	LDURS S3 [SP, #48] 
	LDUR X2 [SP, #56] 
	ADDI SP, SP, #80	

	//SUBIS XZR, X5, #0
	//B.EQ  past
	//LDURS S9 [SP, #0] 
	//FSUBS S9, S7, S9
	//FCMPS S9, S4
	//B.LT end
	
	SUBS XZR, X7, X10
	B.GT end

past:  
	STURS S7  [SP, #0] // store last loss value 
	ADDI  X5, X5, #1   // add to counter
	SUBS  XZR, X2, X5 	// checker
	B.GT   loop1
	
    //Call loss function at end of each epoch
    //Call loss function at the end of the function return the loss





FCMPS S14, S7
SUBIS XZR, X5, #0
	B.EQ  past
	LDURS S9 [SP, #0] 
	FSUBS S9, S7, S9
	FCMPS S9, S4
	B.LT end

past:  
	STURS S7  [SP, #0] // store last loss value 
	ADDI  X5, X5, #1   // add to counter
	SUBS  XZR, X2, X5 	// checker
	B.GT   loop1
B.GT end





M: NAN (7FC00000)
C: NAN (7FC00000)
Loss: NAN (7FC00000)
Normal: 

	LDURS S6, [X15, #0] // loads dataset [j][0]
	LDURS S8, [X15, #4] // loads dataset[j][1]
	
	FSUBS S6, S6, S10 
	FDIVS S6, S6, S12
	FSUBS S8, S8, S11
	FDIVS S8, S8, S13
	STURS S6, [X15, #0]
	STURS S8, [X15, #4] 	
	ADDI  X15, X15, #8   // sets up X15 for the next iteration. 
	ADDI  X16, X16, #1
	SUBS  XZR, X0, X16  // checker 
	B.GT   Normal
	Br LR








