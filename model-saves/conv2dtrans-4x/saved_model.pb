??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
?
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8??
~
Adam/decoded/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/decoded/bias/v
w
'Adam/decoded/bias/v/Read/ReadVariableOpReadVariableOpAdam/decoded/bias/v*
_output_shapes
:*
dtype0
?
Adam/decoded/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/decoded/kernel/v
?
)Adam/decoded/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decoded/kernel/v*&
_output_shapes
:*
dtype0
?
 Adam/conv2d_transpose_243/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_243/bias/v
?
4Adam/conv2d_transpose_243/bias/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_243/bias/v*
_output_shapes
:*
dtype0
?
"Adam/conv2d_transpose_243/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/conv2d_transpose_243/kernel/v
?
6Adam/conv2d_transpose_243/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/conv2d_transpose_243/kernel/v*&
_output_shapes
: *
dtype0
?
 Adam/conv2d_transpose_242/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/conv2d_transpose_242/bias/v
?
4Adam/conv2d_transpose_242/bias/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_242/bias/v*
_output_shapes
: *
dtype0
?
"Adam/conv2d_transpose_242/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*3
shared_name$"Adam/conv2d_transpose_242/kernel/v
?
6Adam/conv2d_transpose_242/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/conv2d_transpose_242/kernel/v*&
_output_shapes
: @*
dtype0
?
 Adam/conv2d_transpose_241/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/conv2d_transpose_241/bias/v
?
4Adam/conv2d_transpose_241/bias/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_241/bias/v*
_output_shapes
:@*
dtype0
?
"Adam/conv2d_transpose_241/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *3
shared_name$"Adam/conv2d_transpose_241/kernel/v
?
6Adam/conv2d_transpose_241/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/conv2d_transpose_241/kernel/v*&
_output_shapes
:@ *
dtype0
~
Adam/encoded/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/encoded/beta/v
w
'Adam/encoded/beta/v/Read/ReadVariableOpReadVariableOpAdam/encoded/beta/v*
_output_shapes
: *
dtype0
?
Adam/encoded/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/encoded/gamma/v
y
(Adam/encoded/gamma/v/Read/ReadVariableOpReadVariableOpAdam/encoded/gamma/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_296/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_296/bias/v
}
*Adam/conv2d_296/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_296/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_296/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *)
shared_nameAdam/conv2d_296/kernel/v
?
,Adam/conv2d_296/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_296/kernel/v*&
_output_shapes
:@ *
dtype0
?
#Adam/batch_normalization_355/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_355/beta/v
?
7Adam/batch_normalization_355/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_355/beta/v*
_output_shapes
:@*
dtype0
?
$Adam/batch_normalization_355/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$Adam/batch_normalization_355/gamma/v
?
8Adam/batch_normalization_355/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_355/gamma/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_295/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_295/bias/v
}
*Adam/conv2d_295/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_295/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_295/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/conv2d_295/kernel/v
?
,Adam/conv2d_295/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_295/kernel/v*&
_output_shapes
: @*
dtype0
?
#Adam/batch_normalization_354/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_354/beta/v
?
7Adam/batch_normalization_354/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_354/beta/v*
_output_shapes
: *
dtype0
?
$Adam/batch_normalization_354/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adam/batch_normalization_354/gamma/v
?
8Adam/batch_normalization_354/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_354/gamma/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_294/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_294/bias/v
}
*Adam/conv2d_294/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_294/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_294/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_294/kernel/v
?
,Adam/conv2d_294/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_294/kernel/v*&
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_353/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_353/beta/v
?
7Adam/batch_normalization_353/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_353/beta/v*
_output_shapes
:*
dtype0
?
$Adam/batch_normalization_353/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_353/gamma/v
?
8Adam/batch_normalization_353/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_353/gamma/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_293/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_293/bias/v
}
*Adam/conv2d_293/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_293/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_293/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_293/kernel/v
?
,Adam/conv2d_293/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_293/kernel/v*&
_output_shapes
:*
dtype0
~
Adam/decoded/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/decoded/bias/m
w
'Adam/decoded/bias/m/Read/ReadVariableOpReadVariableOpAdam/decoded/bias/m*
_output_shapes
:*
dtype0
?
Adam/decoded/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/decoded/kernel/m
?
)Adam/decoded/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decoded/kernel/m*&
_output_shapes
:*
dtype0
?
 Adam/conv2d_transpose_243/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_243/bias/m
?
4Adam/conv2d_transpose_243/bias/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_243/bias/m*
_output_shapes
:*
dtype0
?
"Adam/conv2d_transpose_243/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/conv2d_transpose_243/kernel/m
?
6Adam/conv2d_transpose_243/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/conv2d_transpose_243/kernel/m*&
_output_shapes
: *
dtype0
?
 Adam/conv2d_transpose_242/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/conv2d_transpose_242/bias/m
?
4Adam/conv2d_transpose_242/bias/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_242/bias/m*
_output_shapes
: *
dtype0
?
"Adam/conv2d_transpose_242/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*3
shared_name$"Adam/conv2d_transpose_242/kernel/m
?
6Adam/conv2d_transpose_242/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/conv2d_transpose_242/kernel/m*&
_output_shapes
: @*
dtype0
?
 Adam/conv2d_transpose_241/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/conv2d_transpose_241/bias/m
?
4Adam/conv2d_transpose_241/bias/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_241/bias/m*
_output_shapes
:@*
dtype0
?
"Adam/conv2d_transpose_241/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *3
shared_name$"Adam/conv2d_transpose_241/kernel/m
?
6Adam/conv2d_transpose_241/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/conv2d_transpose_241/kernel/m*&
_output_shapes
:@ *
dtype0
~
Adam/encoded/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/encoded/beta/m
w
'Adam/encoded/beta/m/Read/ReadVariableOpReadVariableOpAdam/encoded/beta/m*
_output_shapes
: *
dtype0
?
Adam/encoded/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/encoded/gamma/m
y
(Adam/encoded/gamma/m/Read/ReadVariableOpReadVariableOpAdam/encoded/gamma/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_296/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_296/bias/m
}
*Adam/conv2d_296/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_296/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_296/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *)
shared_nameAdam/conv2d_296/kernel/m
?
,Adam/conv2d_296/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_296/kernel/m*&
_output_shapes
:@ *
dtype0
?
#Adam/batch_normalization_355/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_355/beta/m
?
7Adam/batch_normalization_355/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_355/beta/m*
_output_shapes
:@*
dtype0
?
$Adam/batch_normalization_355/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$Adam/batch_normalization_355/gamma/m
?
8Adam/batch_normalization_355/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_355/gamma/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_295/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_295/bias/m
}
*Adam/conv2d_295/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_295/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_295/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/conv2d_295/kernel/m
?
,Adam/conv2d_295/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_295/kernel/m*&
_output_shapes
: @*
dtype0
?
#Adam/batch_normalization_354/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_354/beta/m
?
7Adam/batch_normalization_354/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_354/beta/m*
_output_shapes
: *
dtype0
?
$Adam/batch_normalization_354/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adam/batch_normalization_354/gamma/m
?
8Adam/batch_normalization_354/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_354/gamma/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_294/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_294/bias/m
}
*Adam/conv2d_294/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_294/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_294/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_294/kernel/m
?
,Adam/conv2d_294/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_294/kernel/m*&
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_353/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_353/beta/m
?
7Adam/batch_normalization_353/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_353/beta/m*
_output_shapes
:*
dtype0
?
$Adam/batch_normalization_353/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_353/gamma/m
?
8Adam/batch_normalization_353/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_353/gamma/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_293/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_293/bias/m
}
*Adam/conv2d_293/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_293/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_293/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_293/kernel/m
?
,Adam/conv2d_293/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_293/kernel/m*&
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
p
decoded/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedecoded/bias
i
 decoded/bias/Read/ReadVariableOpReadVariableOpdecoded/bias*
_output_shapes
:*
dtype0
?
decoded/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedecoded/kernel
y
"decoded/kernel/Read/ReadVariableOpReadVariableOpdecoded/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_243/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_243/bias
?
-conv2d_transpose_243/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_243/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_243/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameconv2d_transpose_243/kernel
?
/conv2d_transpose_243/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_243/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_242/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_transpose_242/bias
?
-conv2d_transpose_242/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_242/bias*
_output_shapes
: *
dtype0
?
conv2d_transpose_242/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*,
shared_nameconv2d_transpose_242/kernel
?
/conv2d_transpose_242/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_242/kernel*&
_output_shapes
: @*
dtype0
?
conv2d_transpose_241/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameconv2d_transpose_241/bias
?
-conv2d_transpose_241/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_241/bias*
_output_shapes
:@*
dtype0
?
conv2d_transpose_241/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *,
shared_nameconv2d_transpose_241/kernel
?
/conv2d_transpose_241/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_241/kernel*&
_output_shapes
:@ *
dtype0
?
encoded/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameencoded/moving_variance

+encoded/moving_variance/Read/ReadVariableOpReadVariableOpencoded/moving_variance*
_output_shapes
: *
dtype0
~
encoded/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameencoded/moving_mean
w
'encoded/moving_mean/Read/ReadVariableOpReadVariableOpencoded/moving_mean*
_output_shapes
: *
dtype0
p
encoded/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameencoded/beta
i
 encoded/beta/Read/ReadVariableOpReadVariableOpencoded/beta*
_output_shapes
: *
dtype0
r
encoded/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameencoded/gamma
k
!encoded/gamma/Read/ReadVariableOpReadVariableOpencoded/gamma*
_output_shapes
: *
dtype0
v
conv2d_296/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_296/bias
o
#conv2d_296/bias/Read/ReadVariableOpReadVariableOpconv2d_296/bias*
_output_shapes
: *
dtype0
?
conv2d_296/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *"
shared_nameconv2d_296/kernel

%conv2d_296/kernel/Read/ReadVariableOpReadVariableOpconv2d_296/kernel*&
_output_shapes
:@ *
dtype0
?
'batch_normalization_355/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'batch_normalization_355/moving_variance
?
;batch_normalization_355/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_355/moving_variance*
_output_shapes
:@*
dtype0
?
#batch_normalization_355/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization_355/moving_mean
?
7batch_normalization_355/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_355/moving_mean*
_output_shapes
:@*
dtype0
?
batch_normalization_355/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_355/beta
?
0batch_normalization_355/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_355/beta*
_output_shapes
:@*
dtype0
?
batch_normalization_355/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatch_normalization_355/gamma
?
1batch_normalization_355/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_355/gamma*
_output_shapes
:@*
dtype0
v
conv2d_295/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_295/bias
o
#conv2d_295/bias/Read/ReadVariableOpReadVariableOpconv2d_295/bias*
_output_shapes
:@*
dtype0
?
conv2d_295/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*"
shared_nameconv2d_295/kernel

%conv2d_295/kernel/Read/ReadVariableOpReadVariableOpconv2d_295/kernel*&
_output_shapes
: @*
dtype0
?
'batch_normalization_354/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'batch_normalization_354/moving_variance
?
;batch_normalization_354/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_354/moving_variance*
_output_shapes
: *
dtype0
?
#batch_normalization_354/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization_354/moving_mean
?
7batch_normalization_354/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_354/moving_mean*
_output_shapes
: *
dtype0
?
batch_normalization_354/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_354/beta
?
0batch_normalization_354/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_354/beta*
_output_shapes
: *
dtype0
?
batch_normalization_354/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_354/gamma
?
1batch_normalization_354/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_354/gamma*
_output_shapes
: *
dtype0
v
conv2d_294/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_294/bias
o
#conv2d_294/bias/Read/ReadVariableOpReadVariableOpconv2d_294/bias*
_output_shapes
: *
dtype0
?
conv2d_294/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_294/kernel

%conv2d_294/kernel/Read/ReadVariableOpReadVariableOpconv2d_294/kernel*&
_output_shapes
: *
dtype0
?
'batch_normalization_353/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_353/moving_variance
?
;batch_normalization_353/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_353/moving_variance*
_output_shapes
:*
dtype0
?
#batch_normalization_353/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_353/moving_mean
?
7batch_normalization_353/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_353/moving_mean*
_output_shapes
:*
dtype0
?
batch_normalization_353/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_353/beta
?
0batch_normalization_353/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_353/beta*
_output_shapes
:*
dtype0
?
batch_normalization_353/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_353/gamma
?
1batch_normalization_353/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_353/gamma*
_output_shapes
:*
dtype0
v
conv2d_293/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_293/bias
o
#conv2d_293/bias/Read/ReadVariableOpReadVariableOpconv2d_293/bias*
_output_shapes
:*
dtype0
?
conv2d_293/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_293/kernel

%conv2d_293/kernel/Read/ReadVariableOpReadVariableOpconv2d_293/kernel*&
_output_shapes
:*
dtype0
?
 serving_default_conv2d_293_inputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?	
StatefulPartitionedCallStatefulPartitionedCall serving_default_conv2d_293_inputconv2d_293/kernelconv2d_293/biasbatch_normalization_353/gammabatch_normalization_353/beta#batch_normalization_353/moving_mean'batch_normalization_353/moving_varianceconv2d_294/kernelconv2d_294/biasbatch_normalization_354/gammabatch_normalization_354/beta#batch_normalization_354/moving_mean'batch_normalization_354/moving_varianceconv2d_295/kernelconv2d_295/biasbatch_normalization_355/gammabatch_normalization_355/beta#batch_normalization_355/moving_mean'batch_normalization_355/moving_varianceconv2d_296/kernelconv2d_296/biasencoded/gammaencoded/betaencoded/moving_meanencoded/moving_varianceconv2d_transpose_241/kernelconv2d_transpose_241/biasconv2d_transpose_242/kernelconv2d_transpose_242/biasconv2d_transpose_243/kernelconv2d_transpose_243/biasdecoded/kerneldecoded/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2 *0J 8? *-
f(R&
$__inference_signature_wrapper_314692

NoOpNoOp
ʼ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
?
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias
 $_jit_compiled_convolution_op*
?
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses* 
?
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
1axis
	2gamma
3beta
4moving_mean
5moving_variance*
?
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses
<_random_generator* 
?
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

Ckernel
Dbias
 E_jit_compiled_convolution_op*
?
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses* 
?
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
Raxis
	Sgamma
Tbeta
Umoving_mean
Vmoving_variance*
?
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

]kernel
^bias
 __jit_compiled_convolution_op*
?
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses* 
?
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses
laxis
	mgamma
nbeta
omoving_mean
pmoving_variance*
?
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses
w_random_generator* 
?
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

~kernel
bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
!?_jit_compiled_convolution_op*
?
"0
#1
22
33
44
55
C6
D7
S8
T9
U10
V11
]12
^13
m14
n15
o16
p17
~18
19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31*
?
"0
#1
22
33
C4
D5
S6
T7
]8
^9
m10
n11
~12
13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
* 
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate"m?#m?2m?3m?Cm?Dm?Sm?Tm?]m?^m?mm?nm?~m?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?"v?#v?2v?3v?Cv?Dv?Sv?Tv?]v?^v?mv?nv?~v?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?*

?serving_default* 

"0
#1*

"0
#1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
a[
VARIABLE_VALUEconv2d_293/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_293/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
 
20
31
42
53*

20
31*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_353/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_353/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_353/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE'batch_normalization_353/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

C0
D1*

C0
D1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
a[
VARIABLE_VALUEconv2d_294/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_294/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
 
S0
T1
U2
V3*

S0
T1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_354/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_354/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_354/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE'batch_normalization_354/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

]0
^1*

]0
^1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
a[
VARIABLE_VALUEconv2d_295/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_295/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
 
m0
n1
o2
p3*

m0
n1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_355/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_355/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_355/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE'batch_normalization_355/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

~0
1*

~0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
a[
VARIABLE_VALUEconv2d_296/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_296/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
$
?0
?1
?2
?3*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
\V
VARIABLE_VALUEencoded/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEencoded/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEencoded/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEencoded/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
ke
VARIABLE_VALUEconv2d_transpose_241/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEconv2d_transpose_241/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
ke
VARIABLE_VALUEconv2d_transpose_242/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEconv2d_transpose_242/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
lf
VARIABLE_VALUEconv2d_transpose_243/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEconv2d_transpose_243/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEdecoded/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdecoded/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
>
40
51
U2
V3
o4
p5
?6
?7*
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17*

?0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

40
51*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

U0
V1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

o0
p1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_293/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_293/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_353/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_353/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_294/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_294/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_354/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_354/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_295/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_295/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_355/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_355/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_296/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_296/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/encoded/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/encoded/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/conv2d_transpose_241/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/conv2d_transpose_241/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/conv2d_transpose_242/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/conv2d_transpose_242/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/conv2d_transpose_243/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/conv2d_transpose_243/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/decoded/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/decoded/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_293/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_293/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_353/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_353/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_294/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_294/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_354/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_354/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_295/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_295/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/batch_normalization_355/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_355/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_296/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_296/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/encoded/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/encoded/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/conv2d_transpose_241/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/conv2d_transpose_241/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/conv2d_transpose_242/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/conv2d_transpose_242/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/conv2d_transpose_243/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/conv2d_transpose_243/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/decoded/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/decoded/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?"
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_293/kernel/Read/ReadVariableOp#conv2d_293/bias/Read/ReadVariableOp1batch_normalization_353/gamma/Read/ReadVariableOp0batch_normalization_353/beta/Read/ReadVariableOp7batch_normalization_353/moving_mean/Read/ReadVariableOp;batch_normalization_353/moving_variance/Read/ReadVariableOp%conv2d_294/kernel/Read/ReadVariableOp#conv2d_294/bias/Read/ReadVariableOp1batch_normalization_354/gamma/Read/ReadVariableOp0batch_normalization_354/beta/Read/ReadVariableOp7batch_normalization_354/moving_mean/Read/ReadVariableOp;batch_normalization_354/moving_variance/Read/ReadVariableOp%conv2d_295/kernel/Read/ReadVariableOp#conv2d_295/bias/Read/ReadVariableOp1batch_normalization_355/gamma/Read/ReadVariableOp0batch_normalization_355/beta/Read/ReadVariableOp7batch_normalization_355/moving_mean/Read/ReadVariableOp;batch_normalization_355/moving_variance/Read/ReadVariableOp%conv2d_296/kernel/Read/ReadVariableOp#conv2d_296/bias/Read/ReadVariableOp!encoded/gamma/Read/ReadVariableOp encoded/beta/Read/ReadVariableOp'encoded/moving_mean/Read/ReadVariableOp+encoded/moving_variance/Read/ReadVariableOp/conv2d_transpose_241/kernel/Read/ReadVariableOp-conv2d_transpose_241/bias/Read/ReadVariableOp/conv2d_transpose_242/kernel/Read/ReadVariableOp-conv2d_transpose_242/bias/Read/ReadVariableOp/conv2d_transpose_243/kernel/Read/ReadVariableOp-conv2d_transpose_243/bias/Read/ReadVariableOp"decoded/kernel/Read/ReadVariableOp decoded/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,Adam/conv2d_293/kernel/m/Read/ReadVariableOp*Adam/conv2d_293/bias/m/Read/ReadVariableOp8Adam/batch_normalization_353/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_353/beta/m/Read/ReadVariableOp,Adam/conv2d_294/kernel/m/Read/ReadVariableOp*Adam/conv2d_294/bias/m/Read/ReadVariableOp8Adam/batch_normalization_354/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_354/beta/m/Read/ReadVariableOp,Adam/conv2d_295/kernel/m/Read/ReadVariableOp*Adam/conv2d_295/bias/m/Read/ReadVariableOp8Adam/batch_normalization_355/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_355/beta/m/Read/ReadVariableOp,Adam/conv2d_296/kernel/m/Read/ReadVariableOp*Adam/conv2d_296/bias/m/Read/ReadVariableOp(Adam/encoded/gamma/m/Read/ReadVariableOp'Adam/encoded/beta/m/Read/ReadVariableOp6Adam/conv2d_transpose_241/kernel/m/Read/ReadVariableOp4Adam/conv2d_transpose_241/bias/m/Read/ReadVariableOp6Adam/conv2d_transpose_242/kernel/m/Read/ReadVariableOp4Adam/conv2d_transpose_242/bias/m/Read/ReadVariableOp6Adam/conv2d_transpose_243/kernel/m/Read/ReadVariableOp4Adam/conv2d_transpose_243/bias/m/Read/ReadVariableOp)Adam/decoded/kernel/m/Read/ReadVariableOp'Adam/decoded/bias/m/Read/ReadVariableOp,Adam/conv2d_293/kernel/v/Read/ReadVariableOp*Adam/conv2d_293/bias/v/Read/ReadVariableOp8Adam/batch_normalization_353/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_353/beta/v/Read/ReadVariableOp,Adam/conv2d_294/kernel/v/Read/ReadVariableOp*Adam/conv2d_294/bias/v/Read/ReadVariableOp8Adam/batch_normalization_354/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_354/beta/v/Read/ReadVariableOp,Adam/conv2d_295/kernel/v/Read/ReadVariableOp*Adam/conv2d_295/bias/v/Read/ReadVariableOp8Adam/batch_normalization_355/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_355/beta/v/Read/ReadVariableOp,Adam/conv2d_296/kernel/v/Read/ReadVariableOp*Adam/conv2d_296/bias/v/Read/ReadVariableOp(Adam/encoded/gamma/v/Read/ReadVariableOp'Adam/encoded/beta/v/Read/ReadVariableOp6Adam/conv2d_transpose_241/kernel/v/Read/ReadVariableOp4Adam/conv2d_transpose_241/bias/v/Read/ReadVariableOp6Adam/conv2d_transpose_242/kernel/v/Read/ReadVariableOp4Adam/conv2d_transpose_242/bias/v/Read/ReadVariableOp6Adam/conv2d_transpose_243/kernel/v/Read/ReadVariableOp4Adam/conv2d_transpose_243/bias/v/Read/ReadVariableOp)Adam/decoded/kernel/v/Read/ReadVariableOp'Adam/decoded/bias/v/Read/ReadVariableOpConst*d
Tin]
[2Y	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *(
f#R!
__inference__traced_save_316066
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_293/kernelconv2d_293/biasbatch_normalization_353/gammabatch_normalization_353/beta#batch_normalization_353/moving_mean'batch_normalization_353/moving_varianceconv2d_294/kernelconv2d_294/biasbatch_normalization_354/gammabatch_normalization_354/beta#batch_normalization_354/moving_mean'batch_normalization_354/moving_varianceconv2d_295/kernelconv2d_295/biasbatch_normalization_355/gammabatch_normalization_355/beta#batch_normalization_355/moving_mean'batch_normalization_355/moving_varianceconv2d_296/kernelconv2d_296/biasencoded/gammaencoded/betaencoded/moving_meanencoded/moving_varianceconv2d_transpose_241/kernelconv2d_transpose_241/biasconv2d_transpose_242/kernelconv2d_transpose_242/biasconv2d_transpose_243/kernelconv2d_transpose_243/biasdecoded/kerneldecoded/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_293/kernel/mAdam/conv2d_293/bias/m$Adam/batch_normalization_353/gamma/m#Adam/batch_normalization_353/beta/mAdam/conv2d_294/kernel/mAdam/conv2d_294/bias/m$Adam/batch_normalization_354/gamma/m#Adam/batch_normalization_354/beta/mAdam/conv2d_295/kernel/mAdam/conv2d_295/bias/m$Adam/batch_normalization_355/gamma/m#Adam/batch_normalization_355/beta/mAdam/conv2d_296/kernel/mAdam/conv2d_296/bias/mAdam/encoded/gamma/mAdam/encoded/beta/m"Adam/conv2d_transpose_241/kernel/m Adam/conv2d_transpose_241/bias/m"Adam/conv2d_transpose_242/kernel/m Adam/conv2d_transpose_242/bias/m"Adam/conv2d_transpose_243/kernel/m Adam/conv2d_transpose_243/bias/mAdam/decoded/kernel/mAdam/decoded/bias/mAdam/conv2d_293/kernel/vAdam/conv2d_293/bias/v$Adam/batch_normalization_353/gamma/v#Adam/batch_normalization_353/beta/vAdam/conv2d_294/kernel/vAdam/conv2d_294/bias/v$Adam/batch_normalization_354/gamma/v#Adam/batch_normalization_354/beta/vAdam/conv2d_295/kernel/vAdam/conv2d_295/bias/v$Adam/batch_normalization_355/gamma/v#Adam/batch_normalization_355/beta/vAdam/conv2d_296/kernel/vAdam/conv2d_296/bias/vAdam/encoded/gamma/vAdam/encoded/beta/v"Adam/conv2d_transpose_241/kernel/v Adam/conv2d_transpose_241/bias/v"Adam/conv2d_transpose_242/kernel/v Adam/conv2d_transpose_242/bias/v"Adam/conv2d_transpose_243/kernel/v Adam/conv2d_transpose_243/bias/vAdam/decoded/kernel/vAdam/decoded/bias/v*c
Tin\
Z2X*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__traced_restore_316337??
?!
?
C__inference_decoded_layer_call_and_return_conditional_losses_313788

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_encoded_layer_call_and_return_conditional_losses_315606

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_241_layer_call_fn_315615

inputs!
unknown:@ 
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_241_layer_call_and_return_conditional_losses_313653?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
F__inference_conv2d_294_layer_call_and_return_conditional_losses_315325

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????22 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?

e
F__inference_dropout_95_layer_call_and_return_conditional_losses_314070

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_353_layer_call_fn_315243

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_353_layer_call_and_return_conditional_losses_313408?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?!
?
C__inference_decoded_layer_call_and_return_conditional_losses_315782

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?!
?
P__inference_conv2d_transpose_242_layer_call_and_return_conditional_losses_313698

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
f
J__inference_activation_346_layer_call_and_return_conditional_losses_315426

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
K
/__inference_activation_347_layer_call_fn_315539

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_347_layer_call_and_return_conditional_losses_313933h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_355_layer_call_fn_315452

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_355_layer_call_and_return_conditional_losses_313536?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

e
F__inference_dropout_94_layer_call_and_return_conditional_losses_315306

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????ddC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????dd*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????ddw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????ddq
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????dda
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????dd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????dd:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?!
?
P__inference_conv2d_transpose_243_layer_call_and_return_conditional_losses_315739

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_353_layer_call_and_return_conditional_losses_315261

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_355_layer_call_and_return_conditional_losses_315488

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_354_layer_call_fn_315348

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_354_layer_call_and_return_conditional_losses_313441?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_353_layer_call_fn_315230

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_353_layer_call_and_return_conditional_losses_313377?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_encoded_layer_call_and_return_conditional_losses_315588

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?(
__inference__traced_save_316066
file_prefix0
,savev2_conv2d_293_kernel_read_readvariableop.
*savev2_conv2d_293_bias_read_readvariableop<
8savev2_batch_normalization_353_gamma_read_readvariableop;
7savev2_batch_normalization_353_beta_read_readvariableopB
>savev2_batch_normalization_353_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_353_moving_variance_read_readvariableop0
,savev2_conv2d_294_kernel_read_readvariableop.
*savev2_conv2d_294_bias_read_readvariableop<
8savev2_batch_normalization_354_gamma_read_readvariableop;
7savev2_batch_normalization_354_beta_read_readvariableopB
>savev2_batch_normalization_354_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_354_moving_variance_read_readvariableop0
,savev2_conv2d_295_kernel_read_readvariableop.
*savev2_conv2d_295_bias_read_readvariableop<
8savev2_batch_normalization_355_gamma_read_readvariableop;
7savev2_batch_normalization_355_beta_read_readvariableopB
>savev2_batch_normalization_355_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_355_moving_variance_read_readvariableop0
,savev2_conv2d_296_kernel_read_readvariableop.
*savev2_conv2d_296_bias_read_readvariableop,
(savev2_encoded_gamma_read_readvariableop+
'savev2_encoded_beta_read_readvariableop2
.savev2_encoded_moving_mean_read_readvariableop6
2savev2_encoded_moving_variance_read_readvariableop:
6savev2_conv2d_transpose_241_kernel_read_readvariableop8
4savev2_conv2d_transpose_241_bias_read_readvariableop:
6savev2_conv2d_transpose_242_kernel_read_readvariableop8
4savev2_conv2d_transpose_242_bias_read_readvariableop:
6savev2_conv2d_transpose_243_kernel_read_readvariableop8
4savev2_conv2d_transpose_243_bias_read_readvariableop-
)savev2_decoded_kernel_read_readvariableop+
'savev2_decoded_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_adam_conv2d_293_kernel_m_read_readvariableop5
1savev2_adam_conv2d_293_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_353_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_353_beta_m_read_readvariableop7
3savev2_adam_conv2d_294_kernel_m_read_readvariableop5
1savev2_adam_conv2d_294_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_354_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_354_beta_m_read_readvariableop7
3savev2_adam_conv2d_295_kernel_m_read_readvariableop5
1savev2_adam_conv2d_295_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_355_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_355_beta_m_read_readvariableop7
3savev2_adam_conv2d_296_kernel_m_read_readvariableop5
1savev2_adam_conv2d_296_bias_m_read_readvariableop3
/savev2_adam_encoded_gamma_m_read_readvariableop2
.savev2_adam_encoded_beta_m_read_readvariableopA
=savev2_adam_conv2d_transpose_241_kernel_m_read_readvariableop?
;savev2_adam_conv2d_transpose_241_bias_m_read_readvariableopA
=savev2_adam_conv2d_transpose_242_kernel_m_read_readvariableop?
;savev2_adam_conv2d_transpose_242_bias_m_read_readvariableopA
=savev2_adam_conv2d_transpose_243_kernel_m_read_readvariableop?
;savev2_adam_conv2d_transpose_243_bias_m_read_readvariableop4
0savev2_adam_decoded_kernel_m_read_readvariableop2
.savev2_adam_decoded_bias_m_read_readvariableop7
3savev2_adam_conv2d_293_kernel_v_read_readvariableop5
1savev2_adam_conv2d_293_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_353_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_353_beta_v_read_readvariableop7
3savev2_adam_conv2d_294_kernel_v_read_readvariableop5
1savev2_adam_conv2d_294_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_354_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_354_beta_v_read_readvariableop7
3savev2_adam_conv2d_295_kernel_v_read_readvariableop5
1savev2_adam_conv2d_295_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_355_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_355_beta_v_read_readvariableop7
3savev2_adam_conv2d_296_kernel_v_read_readvariableop5
1savev2_adam_conv2d_296_bias_v_read_readvariableop3
/savev2_adam_encoded_gamma_v_read_readvariableop2
.savev2_adam_encoded_beta_v_read_readvariableopA
=savev2_adam_conv2d_transpose_241_kernel_v_read_readvariableop?
;savev2_adam_conv2d_transpose_241_bias_v_read_readvariableopA
=savev2_adam_conv2d_transpose_242_kernel_v_read_readvariableop?
;savev2_adam_conv2d_transpose_242_bias_v_read_readvariableopA
=savev2_adam_conv2d_transpose_243_kernel_v_read_readvariableop?
;savev2_adam_conv2d_transpose_243_bias_v_read_readvariableop4
0savev2_adam_decoded_kernel_v_read_readvariableop2
.savev2_adam_decoded_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?1
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*?0
value?0B?0XB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*?
value?B?XB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?&
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_293_kernel_read_readvariableop*savev2_conv2d_293_bias_read_readvariableop8savev2_batch_normalization_353_gamma_read_readvariableop7savev2_batch_normalization_353_beta_read_readvariableop>savev2_batch_normalization_353_moving_mean_read_readvariableopBsavev2_batch_normalization_353_moving_variance_read_readvariableop,savev2_conv2d_294_kernel_read_readvariableop*savev2_conv2d_294_bias_read_readvariableop8savev2_batch_normalization_354_gamma_read_readvariableop7savev2_batch_normalization_354_beta_read_readvariableop>savev2_batch_normalization_354_moving_mean_read_readvariableopBsavev2_batch_normalization_354_moving_variance_read_readvariableop,savev2_conv2d_295_kernel_read_readvariableop*savev2_conv2d_295_bias_read_readvariableop8savev2_batch_normalization_355_gamma_read_readvariableop7savev2_batch_normalization_355_beta_read_readvariableop>savev2_batch_normalization_355_moving_mean_read_readvariableopBsavev2_batch_normalization_355_moving_variance_read_readvariableop,savev2_conv2d_296_kernel_read_readvariableop*savev2_conv2d_296_bias_read_readvariableop(savev2_encoded_gamma_read_readvariableop'savev2_encoded_beta_read_readvariableop.savev2_encoded_moving_mean_read_readvariableop2savev2_encoded_moving_variance_read_readvariableop6savev2_conv2d_transpose_241_kernel_read_readvariableop4savev2_conv2d_transpose_241_bias_read_readvariableop6savev2_conv2d_transpose_242_kernel_read_readvariableop4savev2_conv2d_transpose_242_bias_read_readvariableop6savev2_conv2d_transpose_243_kernel_read_readvariableop4savev2_conv2d_transpose_243_bias_read_readvariableop)savev2_decoded_kernel_read_readvariableop'savev2_decoded_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_adam_conv2d_293_kernel_m_read_readvariableop1savev2_adam_conv2d_293_bias_m_read_readvariableop?savev2_adam_batch_normalization_353_gamma_m_read_readvariableop>savev2_adam_batch_normalization_353_beta_m_read_readvariableop3savev2_adam_conv2d_294_kernel_m_read_readvariableop1savev2_adam_conv2d_294_bias_m_read_readvariableop?savev2_adam_batch_normalization_354_gamma_m_read_readvariableop>savev2_adam_batch_normalization_354_beta_m_read_readvariableop3savev2_adam_conv2d_295_kernel_m_read_readvariableop1savev2_adam_conv2d_295_bias_m_read_readvariableop?savev2_adam_batch_normalization_355_gamma_m_read_readvariableop>savev2_adam_batch_normalization_355_beta_m_read_readvariableop3savev2_adam_conv2d_296_kernel_m_read_readvariableop1savev2_adam_conv2d_296_bias_m_read_readvariableop/savev2_adam_encoded_gamma_m_read_readvariableop.savev2_adam_encoded_beta_m_read_readvariableop=savev2_adam_conv2d_transpose_241_kernel_m_read_readvariableop;savev2_adam_conv2d_transpose_241_bias_m_read_readvariableop=savev2_adam_conv2d_transpose_242_kernel_m_read_readvariableop;savev2_adam_conv2d_transpose_242_bias_m_read_readvariableop=savev2_adam_conv2d_transpose_243_kernel_m_read_readvariableop;savev2_adam_conv2d_transpose_243_bias_m_read_readvariableop0savev2_adam_decoded_kernel_m_read_readvariableop.savev2_adam_decoded_bias_m_read_readvariableop3savev2_adam_conv2d_293_kernel_v_read_readvariableop1savev2_adam_conv2d_293_bias_v_read_readvariableop?savev2_adam_batch_normalization_353_gamma_v_read_readvariableop>savev2_adam_batch_normalization_353_beta_v_read_readvariableop3savev2_adam_conv2d_294_kernel_v_read_readvariableop1savev2_adam_conv2d_294_bias_v_read_readvariableop?savev2_adam_batch_normalization_354_gamma_v_read_readvariableop>savev2_adam_batch_normalization_354_beta_v_read_readvariableop3savev2_adam_conv2d_295_kernel_v_read_readvariableop1savev2_adam_conv2d_295_bias_v_read_readvariableop?savev2_adam_batch_normalization_355_gamma_v_read_readvariableop>savev2_adam_batch_normalization_355_beta_v_read_readvariableop3savev2_adam_conv2d_296_kernel_v_read_readvariableop1savev2_adam_conv2d_296_bias_v_read_readvariableop/savev2_adam_encoded_gamma_v_read_readvariableop.savev2_adam_encoded_beta_v_read_readvariableop=savev2_adam_conv2d_transpose_241_kernel_v_read_readvariableop;savev2_adam_conv2d_transpose_241_bias_v_read_readvariableop=savev2_adam_conv2d_transpose_242_kernel_v_read_readvariableop;savev2_adam_conv2d_transpose_242_bias_v_read_readvariableop=savev2_adam_conv2d_transpose_243_kernel_v_read_readvariableop;savev2_adam_conv2d_transpose_243_bias_v_read_readvariableop0savev2_adam_decoded_kernel_v_read_readvariableop.savev2_adam_decoded_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *f
dtypes\
Z2X	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::: : : : : : : @:@:@:@:@:@:@ : : : : : :@ :@: @: : :::: : : : : : : ::::: : : : : @:@:@:@:@ : : : :@ :@: @: : :::::::: : : : : @:@:@:@:@ : : : :@ :@: @: : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@ : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:@ : 

_output_shapes
:@:,(
&
_output_shapes
: @: 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
::  

_output_shapes
::!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :,((
&
_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
::,,(
&
_output_shapes
: : -

_output_shapes
: : .

_output_shapes
: : /

_output_shapes
: :,0(
&
_output_shapes
: @: 1

_output_shapes
:@: 2

_output_shapes
:@: 3

_output_shapes
:@:,4(
&
_output_shapes
:@ : 5

_output_shapes
: : 6

_output_shapes
: : 7

_output_shapes
: :,8(
&
_output_shapes
:@ : 9

_output_shapes
:@:,:(
&
_output_shapes
: @: ;

_output_shapes
: :,<(
&
_output_shapes
: : =

_output_shapes
::,>(
&
_output_shapes
:: ?

_output_shapes
::,@(
&
_output_shapes
:: A

_output_shapes
:: B

_output_shapes
:: C

_output_shapes
::,D(
&
_output_shapes
: : E

_output_shapes
: : F

_output_shapes
: : G

_output_shapes
: :,H(
&
_output_shapes
: @: I

_output_shapes
:@: J

_output_shapes
:@: K

_output_shapes
:@:,L(
&
_output_shapes
:@ : M

_output_shapes
: : N

_output_shapes
: : O

_output_shapes
: :,P(
&
_output_shapes
:@ : Q

_output_shapes
:@:,R(
&
_output_shapes
: @: S

_output_shapes
: :,T(
&
_output_shapes
: : U

_output_shapes
::,V(
&
_output_shapes
:: W

_output_shapes
::X

_output_shapes
: 
?
?
S__inference_batch_normalization_353_layer_call_and_return_conditional_losses_315279

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?a
?
I__inference_sequential_36_layer_call_and_return_conditional_losses_314615
conv2d_293_input+
conv2d_293_314530:
conv2d_293_314532:,
batch_normalization_353_314536:,
batch_normalization_353_314538:,
batch_normalization_353_314540:,
batch_normalization_353_314542:+
conv2d_294_314546: 
conv2d_294_314548: ,
batch_normalization_354_314552: ,
batch_normalization_354_314554: ,
batch_normalization_354_314556: ,
batch_normalization_354_314558: +
conv2d_295_314561: @
conv2d_295_314563:@,
batch_normalization_355_314567:@,
batch_normalization_355_314569:@,
batch_normalization_355_314571:@,
batch_normalization_355_314573:@+
conv2d_296_314577:@ 
conv2d_296_314579: 
encoded_314584: 
encoded_314586: 
encoded_314588: 
encoded_314590: 5
conv2d_transpose_241_314594:@ )
conv2d_transpose_241_314596:@5
conv2d_transpose_242_314599: @)
conv2d_transpose_242_314601: 5
conv2d_transpose_243_314604: )
conv2d_transpose_243_314606:(
decoded_314609:
decoded_314611:
identity??/batch_normalization_353/StatefulPartitionedCall?/batch_normalization_354/StatefulPartitionedCall?/batch_normalization_355/StatefulPartitionedCall?"conv2d_293/StatefulPartitionedCall?"conv2d_294/StatefulPartitionedCall?"conv2d_295/StatefulPartitionedCall?"conv2d_296/StatefulPartitionedCall?,conv2d_transpose_241/StatefulPartitionedCall?,conv2d_transpose_242/StatefulPartitionedCall?,conv2d_transpose_243/StatefulPartitionedCall?decoded/StatefulPartitionedCall?"dropout_94/StatefulPartitionedCall?"dropout_95/StatefulPartitionedCall?encoded/StatefulPartitionedCall?
"conv2d_293/StatefulPartitionedCallStatefulPartitionedCallconv2d_293_inputconv2d_293_314530conv2d_293_314532*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_293_layer_call_and_return_conditional_losses_313812?
activation_344/PartitionedCallPartitionedCall+conv2d_293/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_344_layer_call_and_return_conditional_losses_313823?
/batch_normalization_353/StatefulPartitionedCallStatefulPartitionedCall'activation_344/PartitionedCall:output:0batch_normalization_353_314536batch_normalization_353_314538batch_normalization_353_314540batch_normalization_353_314542*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_353_layer_call_and_return_conditional_losses_313408?
"dropout_94/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_353/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dropout_94_layer_call_and_return_conditional_losses_314125?
"conv2d_294/StatefulPartitionedCallStatefulPartitionedCall+dropout_94/StatefulPartitionedCall:output:0conv2d_294_314546conv2d_294_314548*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_294_layer_call_and_return_conditional_losses_313851?
activation_345/PartitionedCallPartitionedCall+conv2d_294/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_345_layer_call_and_return_conditional_losses_313862?
/batch_normalization_354/StatefulPartitionedCallStatefulPartitionedCall'activation_345/PartitionedCall:output:0batch_normalization_354_314552batch_normalization_354_314554batch_normalization_354_314556batch_normalization_354_314558*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_354_layer_call_and_return_conditional_losses_313472?
"conv2d_295/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_354/StatefulPartitionedCall:output:0conv2d_295_314561conv2d_295_314563*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_295_layer_call_and_return_conditional_losses_313883?
activation_346/PartitionedCallPartitionedCall+conv2d_295/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_346_layer_call_and_return_conditional_losses_313894?
/batch_normalization_355/StatefulPartitionedCallStatefulPartitionedCall'activation_346/PartitionedCall:output:0batch_normalization_355_314567batch_normalization_355_314569batch_normalization_355_314571batch_normalization_355_314573*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_355_layer_call_and_return_conditional_losses_313536?
"dropout_95/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_355/StatefulPartitionedCall:output:0#^dropout_94/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dropout_95_layer_call_and_return_conditional_losses_314070?
"conv2d_296/StatefulPartitionedCallStatefulPartitionedCall+dropout_95/StatefulPartitionedCall:output:0conv2d_296_314577conv2d_296_314579*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_296_layer_call_and_return_conditional_losses_313922?
activation_347/PartitionedCallPartitionedCall+conv2d_296/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_347_layer_call_and_return_conditional_losses_313933?
encoded/CastCast'activation_347/PartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
encoded/StatefulPartitionedCallStatefulPartitionedCallencoded/Cast:y:0encoded_314584encoded_314586encoded_314588encoded_314590*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_encoded_layer_call_and_return_conditional_losses_313600?
conv2d_transpose_241/CastCast(encoded/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
,conv2d_transpose_241/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_241/Cast:y:0conv2d_transpose_241_314594conv2d_transpose_241_314596*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_241_layer_call_and_return_conditional_losses_313653?
,conv2d_transpose_242/StatefulPartitionedCallStatefulPartitionedCall5conv2d_transpose_241/StatefulPartitionedCall:output:0conv2d_transpose_242_314599conv2d_transpose_242_314601*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_242_layer_call_and_return_conditional_losses_313698?
,conv2d_transpose_243/StatefulPartitionedCallStatefulPartitionedCall5conv2d_transpose_242/StatefulPartitionedCall:output:0conv2d_transpose_243_314604conv2d_transpose_243_314606*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_243_layer_call_and_return_conditional_losses_313743?
decoded/StatefulPartitionedCallStatefulPartitionedCall5conv2d_transpose_243/StatefulPartitionedCall:output:0decoded_314609decoded_314611*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_decoded_layer_call_and_return_conditional_losses_313788?
IdentityIdentity(decoded/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp0^batch_normalization_353/StatefulPartitionedCall0^batch_normalization_354/StatefulPartitionedCall0^batch_normalization_355/StatefulPartitionedCall#^conv2d_293/StatefulPartitionedCall#^conv2d_294/StatefulPartitionedCall#^conv2d_295/StatefulPartitionedCall#^conv2d_296/StatefulPartitionedCall-^conv2d_transpose_241/StatefulPartitionedCall-^conv2d_transpose_242/StatefulPartitionedCall-^conv2d_transpose_243/StatefulPartitionedCall ^decoded/StatefulPartitionedCall#^dropout_94/StatefulPartitionedCall#^dropout_95/StatefulPartitionedCall ^encoded/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_353/StatefulPartitionedCall/batch_normalization_353/StatefulPartitionedCall2b
/batch_normalization_354/StatefulPartitionedCall/batch_normalization_354/StatefulPartitionedCall2b
/batch_normalization_355/StatefulPartitionedCall/batch_normalization_355/StatefulPartitionedCall2H
"conv2d_293/StatefulPartitionedCall"conv2d_293/StatefulPartitionedCall2H
"conv2d_294/StatefulPartitionedCall"conv2d_294/StatefulPartitionedCall2H
"conv2d_295/StatefulPartitionedCall"conv2d_295/StatefulPartitionedCall2H
"conv2d_296/StatefulPartitionedCall"conv2d_296/StatefulPartitionedCall2\
,conv2d_transpose_241/StatefulPartitionedCall,conv2d_transpose_241/StatefulPartitionedCall2\
,conv2d_transpose_242/StatefulPartitionedCall,conv2d_transpose_242/StatefulPartitionedCall2\
,conv2d_transpose_243/StatefulPartitionedCall,conv2d_transpose_243/StatefulPartitionedCall2B
decoded/StatefulPartitionedCalldecoded/StatefulPartitionedCall2H
"dropout_94/StatefulPartitionedCall"dropout_94/StatefulPartitionedCall2H
"dropout_95/StatefulPartitionedCall"dropout_95/StatefulPartitionedCall2B
encoded/StatefulPartitionedCallencoded/StatefulPartitionedCall:c _
1
_output_shapes
:???????????
*
_user_specified_nameconv2d_293_input
?
f
J__inference_activation_347_layer_call_and_return_conditional_losses_313933

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:????????? b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
.__inference_sequential_36_layer_call_fn_314761

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@ 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23:@ 

unknown_24:@$

unknown_25: @

unknown_26: $

unknown_27: 

unknown_28:$

unknown_29:

unknown_30:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_sequential_36_layer_call_and_return_conditional_losses_313967y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_36_layer_call_fn_314830

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@ 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23:@ 

unknown_24:@$

unknown_25: @

unknown_26: $

unknown_27: 

unknown_28:$

unknown_29:

unknown_30:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*:
_read_only_resource_inputs
	
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_sequential_36_layer_call_and_return_conditional_losses_314303y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_95_layer_call_fn_315493

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dropout_95_layer_call_and_return_conditional_losses_313910h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
F__inference_conv2d_295_layer_call_and_return_conditional_losses_313883

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????22 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????22 
 
_user_specified_nameinputs
?
?
C__inference_encoded_layer_call_and_return_conditional_losses_313569

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_354_layer_call_fn_315361

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_354_layer_call_and_return_conditional_losses_313472?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
d
+__inference_dropout_95_layer_call_fn_315498

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dropout_95_layer_call_and_return_conditional_losses_314070w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
F__inference_dropout_95_layer_call_and_return_conditional_losses_315503

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_encoded_layer_call_fn_315557

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_encoded_layer_call_and_return_conditional_losses_313569?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
f
J__inference_activation_347_layer_call_and_return_conditional_losses_315544

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:????????? b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
K
/__inference_activation_345_layer_call_fn_315330

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_345_layer_call_and_return_conditional_losses_313862h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????22 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22 :W S
/
_output_shapes
:?????????22 
 
_user_specified_nameinputs
?
?
(__inference_decoded_layer_call_fn_315748

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_decoded_layer_call_and_return_conditional_losses_313788?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
K
/__inference_activation_346_layer_call_fn_315421

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_346_layer_call_and_return_conditional_losses_313894h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
+__inference_dropout_94_layer_call_fn_315289

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dropout_94_layer_call_and_return_conditional_losses_314125w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????dd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????dd22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?

?
F__inference_conv2d_296_layer_call_and_return_conditional_losses_315534

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?!
?
P__inference_conv2d_transpose_242_layer_call_and_return_conditional_losses_315696

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

?
F__inference_conv2d_293_layer_call_and_return_conditional_losses_313812

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ddg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????ddw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_355_layer_call_and_return_conditional_losses_313536

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
f
J__inference_activation_346_layer_call_and_return_conditional_losses_313894

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_355_layer_call_and_return_conditional_losses_315470

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
f
J__inference_activation_344_layer_call_and_return_conditional_losses_313823

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????ddb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????dd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????dd:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?

?
F__inference_conv2d_293_layer_call_and_return_conditional_losses_315207

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ddg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????ddw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
F__inference_conv2d_295_layer_call_and_return_conditional_losses_315416

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????22 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????22 
 
_user_specified_nameinputs
?

?
F__inference_conv2d_294_layer_call_and_return_conditional_losses_313851

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????22 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?

e
F__inference_dropout_94_layer_call_and_return_conditional_losses_314125

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????ddC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????dd*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????ddw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????ddq
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????dda
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????dd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????dd:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
̈́
?
I__inference_sequential_36_layer_call_and_return_conditional_losses_315188

inputsC
)conv2d_293_conv2d_readvariableop_resource:8
*conv2d_293_biasadd_readvariableop_resource:=
/batch_normalization_353_readvariableop_resource:?
1batch_normalization_353_readvariableop_1_resource:N
@batch_normalization_353_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_353_fusedbatchnormv3_readvariableop_1_resource:C
)conv2d_294_conv2d_readvariableop_resource: 8
*conv2d_294_biasadd_readvariableop_resource: =
/batch_normalization_354_readvariableop_resource: ?
1batch_normalization_354_readvariableop_1_resource: N
@batch_normalization_354_fusedbatchnormv3_readvariableop_resource: P
Bbatch_normalization_354_fusedbatchnormv3_readvariableop_1_resource: C
)conv2d_295_conv2d_readvariableop_resource: @8
*conv2d_295_biasadd_readvariableop_resource:@=
/batch_normalization_355_readvariableop_resource:@?
1batch_normalization_355_readvariableop_1_resource:@N
@batch_normalization_355_fusedbatchnormv3_readvariableop_resource:@P
Bbatch_normalization_355_fusedbatchnormv3_readvariableop_1_resource:@C
)conv2d_296_conv2d_readvariableop_resource:@ 8
*conv2d_296_biasadd_readvariableop_resource: -
encoded_readvariableop_resource: /
!encoded_readvariableop_1_resource: >
0encoded_fusedbatchnormv3_readvariableop_resource: @
2encoded_fusedbatchnormv3_readvariableop_1_resource: W
=conv2d_transpose_241_conv2d_transpose_readvariableop_resource:@ B
4conv2d_transpose_241_biasadd_readvariableop_resource:@W
=conv2d_transpose_242_conv2d_transpose_readvariableop_resource: @B
4conv2d_transpose_242_biasadd_readvariableop_resource: W
=conv2d_transpose_243_conv2d_transpose_readvariableop_resource: B
4conv2d_transpose_243_biasadd_readvariableop_resource:J
0decoded_conv2d_transpose_readvariableop_resource:5
'decoded_biasadd_readvariableop_resource:
identity??&batch_normalization_353/AssignNewValue?(batch_normalization_353/AssignNewValue_1?7batch_normalization_353/FusedBatchNormV3/ReadVariableOp?9batch_normalization_353/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_353/ReadVariableOp?(batch_normalization_353/ReadVariableOp_1?&batch_normalization_354/AssignNewValue?(batch_normalization_354/AssignNewValue_1?7batch_normalization_354/FusedBatchNormV3/ReadVariableOp?9batch_normalization_354/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_354/ReadVariableOp?(batch_normalization_354/ReadVariableOp_1?&batch_normalization_355/AssignNewValue?(batch_normalization_355/AssignNewValue_1?7batch_normalization_355/FusedBatchNormV3/ReadVariableOp?9batch_normalization_355/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_355/ReadVariableOp?(batch_normalization_355/ReadVariableOp_1?!conv2d_293/BiasAdd/ReadVariableOp? conv2d_293/Conv2D/ReadVariableOp?!conv2d_294/BiasAdd/ReadVariableOp? conv2d_294/Conv2D/ReadVariableOp?!conv2d_295/BiasAdd/ReadVariableOp? conv2d_295/Conv2D/ReadVariableOp?!conv2d_296/BiasAdd/ReadVariableOp? conv2d_296/Conv2D/ReadVariableOp?+conv2d_transpose_241/BiasAdd/ReadVariableOp?4conv2d_transpose_241/conv2d_transpose/ReadVariableOp?+conv2d_transpose_242/BiasAdd/ReadVariableOp?4conv2d_transpose_242/conv2d_transpose/ReadVariableOp?+conv2d_transpose_243/BiasAdd/ReadVariableOp?4conv2d_transpose_243/conv2d_transpose/ReadVariableOp?decoded/BiasAdd/ReadVariableOp?'decoded/conv2d_transpose/ReadVariableOp?encoded/AssignNewValue?encoded/AssignNewValue_1?'encoded/FusedBatchNormV3/ReadVariableOp?)encoded/FusedBatchNormV3/ReadVariableOp_1?encoded/ReadVariableOp?encoded/ReadVariableOp_1?
 conv2d_293/Conv2D/ReadVariableOpReadVariableOp)conv2d_293_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_293/Conv2DConv2Dinputs(conv2d_293/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd*
paddingSAME*
strides
?
!conv2d_293/BiasAdd/ReadVariableOpReadVariableOp*conv2d_293_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_293/BiasAddBiasAddconv2d_293/Conv2D:output:0)conv2d_293/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ddr
activation_344/ReluReluconv2d_293/BiasAdd:output:0*
T0*/
_output_shapes
:?????????dd?
&batch_normalization_353/ReadVariableOpReadVariableOp/batch_normalization_353_readvariableop_resource*
_output_shapes
:*
dtype0?
(batch_normalization_353/ReadVariableOp_1ReadVariableOp1batch_normalization_353_readvariableop_1_resource*
_output_shapes
:*
dtype0?
7batch_normalization_353/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_353_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
9batch_normalization_353/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_353_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
(batch_normalization_353/FusedBatchNormV3FusedBatchNormV3!activation_344/Relu:activations:0.batch_normalization_353/ReadVariableOp:value:00batch_normalization_353/ReadVariableOp_1:value:0?batch_normalization_353/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_353/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????dd:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
&batch_normalization_353/AssignNewValueAssignVariableOp@batch_normalization_353_fusedbatchnormv3_readvariableop_resource5batch_normalization_353/FusedBatchNormV3:batch_mean:08^batch_normalization_353/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
(batch_normalization_353/AssignNewValue_1AssignVariableOpBbatch_normalization_353_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_353/FusedBatchNormV3:batch_variance:0:^batch_normalization_353/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(]
dropout_94/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_94/dropout/MulMul,batch_normalization_353/FusedBatchNormV3:y:0!dropout_94/dropout/Const:output:0*
T0*/
_output_shapes
:?????????ddt
dropout_94/dropout/ShapeShape,batch_normalization_353/FusedBatchNormV3:y:0*
T0*
_output_shapes
:?
/dropout_94/dropout/random_uniform/RandomUniformRandomUniform!dropout_94/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????dd*
dtype0f
!dropout_94/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_94/dropout/GreaterEqualGreaterEqual8dropout_94/dropout/random_uniform/RandomUniform:output:0*dropout_94/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????dd?
dropout_94/dropout/CastCast#dropout_94/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????dd?
dropout_94/dropout/Mul_1Muldropout_94/dropout/Mul:z:0dropout_94/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????dd?
 conv2d_294/Conv2D/ReadVariableOpReadVariableOp)conv2d_294_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_294/Conv2DConv2Ddropout_94/dropout/Mul_1:z:0(conv2d_294/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 *
paddingSAME*
strides
?
!conv2d_294/BiasAdd/ReadVariableOpReadVariableOp*conv2d_294_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_294/BiasAddBiasAddconv2d_294/Conv2D:output:0)conv2d_294/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 r
activation_345/ReluReluconv2d_294/BiasAdd:output:0*
T0*/
_output_shapes
:?????????22 ?
&batch_normalization_354/ReadVariableOpReadVariableOp/batch_normalization_354_readvariableop_resource*
_output_shapes
: *
dtype0?
(batch_normalization_354/ReadVariableOp_1ReadVariableOp1batch_normalization_354_readvariableop_1_resource*
_output_shapes
: *
dtype0?
7batch_normalization_354/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_354_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
9batch_normalization_354/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_354_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
(batch_normalization_354/FusedBatchNormV3FusedBatchNormV3!activation_345/Relu:activations:0.batch_normalization_354/ReadVariableOp:value:00batch_normalization_354/ReadVariableOp_1:value:0?batch_normalization_354/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_354/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????22 : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
&batch_normalization_354/AssignNewValueAssignVariableOp@batch_normalization_354_fusedbatchnormv3_readvariableop_resource5batch_normalization_354/FusedBatchNormV3:batch_mean:08^batch_normalization_354/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
(batch_normalization_354/AssignNewValue_1AssignVariableOpBbatch_normalization_354_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_354/FusedBatchNormV3:batch_variance:0:^batch_normalization_354/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
 conv2d_295/Conv2D/ReadVariableOpReadVariableOp)conv2d_295_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_295/Conv2DConv2D,batch_normalization_354/FusedBatchNormV3:y:0(conv2d_295/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
!conv2d_295/BiasAdd/ReadVariableOpReadVariableOp*conv2d_295_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_295/BiasAddBiasAddconv2d_295/Conv2D:output:0)conv2d_295/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@r
activation_346/ReluReluconv2d_295/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
&batch_normalization_355/ReadVariableOpReadVariableOp/batch_normalization_355_readvariableop_resource*
_output_shapes
:@*
dtype0?
(batch_normalization_355/ReadVariableOp_1ReadVariableOp1batch_normalization_355_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
7batch_normalization_355/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_355_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
9batch_normalization_355/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_355_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
(batch_normalization_355/FusedBatchNormV3FusedBatchNormV3!activation_346/Relu:activations:0.batch_normalization_355/ReadVariableOp:value:00batch_normalization_355/ReadVariableOp_1:value:0?batch_normalization_355/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_355/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
&batch_normalization_355/AssignNewValueAssignVariableOp@batch_normalization_355_fusedbatchnormv3_readvariableop_resource5batch_normalization_355/FusedBatchNormV3:batch_mean:08^batch_normalization_355/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
(batch_normalization_355/AssignNewValue_1AssignVariableOpBbatch_normalization_355_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_355/FusedBatchNormV3:batch_variance:0:^batch_normalization_355/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(]
dropout_95/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_95/dropout/MulMul,batch_normalization_355/FusedBatchNormV3:y:0!dropout_95/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@t
dropout_95/dropout/ShapeShape,batch_normalization_355/FusedBatchNormV3:y:0*
T0*
_output_shapes
:?
/dropout_95/dropout/random_uniform/RandomUniformRandomUniform!dropout_95/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0f
!dropout_95/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_95/dropout/GreaterEqualGreaterEqual8dropout_95/dropout/random_uniform/RandomUniform:output:0*dropout_95/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@?
dropout_95/dropout/CastCast#dropout_95/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@?
dropout_95/dropout/Mul_1Muldropout_95/dropout/Mul:z:0dropout_95/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@?
 conv2d_296/Conv2D/ReadVariableOpReadVariableOp)conv2d_296_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
conv2d_296/Conv2DConv2Ddropout_95/dropout/Mul_1:z:0(conv2d_296/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
!conv2d_296/BiasAdd/ReadVariableOpReadVariableOp*conv2d_296_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_296/BiasAddBiasAddconv2d_296/Conv2D:output:0)conv2d_296/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? r
activation_347/ReluReluconv2d_296/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
encoded/CastCast!activation_347/Relu:activations:0*

DstT0*

SrcT0*/
_output_shapes
:????????? r
encoded/ReadVariableOpReadVariableOpencoded_readvariableop_resource*
_output_shapes
: *
dtype0v
encoded/ReadVariableOp_1ReadVariableOp!encoded_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'encoded/FusedBatchNormV3/ReadVariableOpReadVariableOp0encoded_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
)encoded/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp2encoded_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
encoded/FusedBatchNormV3FusedBatchNormV3encoded/Cast:y:0encoded/ReadVariableOp:value:0 encoded/ReadVariableOp_1:value:0/encoded/FusedBatchNormV3/ReadVariableOp:value:01encoded/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
encoded/AssignNewValueAssignVariableOp0encoded_fusedbatchnormv3_readvariableop_resource%encoded/FusedBatchNormV3:batch_mean:0(^encoded/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
encoded/AssignNewValue_1AssignVariableOp2encoded_fusedbatchnormv3_readvariableop_1_resource)encoded/FusedBatchNormV3:batch_variance:0*^encoded/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
conv2d_transpose_241/CastCastencoded/FusedBatchNormV3:y:0*

DstT0*

SrcT0*/
_output_shapes
:????????? g
conv2d_transpose_241/ShapeShapeconv2d_transpose_241/Cast:y:0*
T0*
_output_shapes
:r
(conv2d_transpose_241/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_241/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_241/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_241/strided_sliceStridedSlice#conv2d_transpose_241/Shape:output:01conv2d_transpose_241/strided_slice/stack:output:03conv2d_transpose_241/strided_slice/stack_1:output:03conv2d_transpose_241/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_241/stack/1Const*
_output_shapes
: *
dtype0*
value	B :^
conv2d_transpose_241/stack/2Const*
_output_shapes
: *
dtype0*
value	B :^
conv2d_transpose_241/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose_241/stackPack+conv2d_transpose_241/strided_slice:output:0%conv2d_transpose_241/stack/1:output:0%conv2d_transpose_241/stack/2:output:0%conv2d_transpose_241/stack/3:output:0*
N*
T0*
_output_shapes
:t
*conv2d_transpose_241/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,conv2d_transpose_241/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,conv2d_transpose_241/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$conv2d_transpose_241/strided_slice_1StridedSlice#conv2d_transpose_241/stack:output:03conv2d_transpose_241/strided_slice_1/stack:output:05conv2d_transpose_241/strided_slice_1/stack_1:output:05conv2d_transpose_241/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
4conv2d_transpose_241/conv2d_transpose/ReadVariableOpReadVariableOp=conv2d_transpose_241_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
%conv2d_transpose_241/conv2d_transposeConv2DBackpropInput#conv2d_transpose_241/stack:output:0<conv2d_transpose_241/conv2d_transpose/ReadVariableOp:value:0conv2d_transpose_241/Cast:y:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
+conv2d_transpose_241/BiasAdd/ReadVariableOpReadVariableOp4conv2d_transpose_241_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_transpose_241/BiasAddBiasAdd.conv2d_transpose_241/conv2d_transpose:output:03conv2d_transpose_241/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
conv2d_transpose_241/ReluRelu%conv2d_transpose_241/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@q
conv2d_transpose_242/ShapeShape'conv2d_transpose_241/Relu:activations:0*
T0*
_output_shapes
:r
(conv2d_transpose_242/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_242/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_242/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_242/strided_sliceStridedSlice#conv2d_transpose_242/Shape:output:01conv2d_transpose_242/strided_slice/stack:output:03conv2d_transpose_242/strided_slice/stack_1:output:03conv2d_transpose_242/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_242/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2^
conv2d_transpose_242/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2^
conv2d_transpose_242/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_242/stackPack+conv2d_transpose_242/strided_slice:output:0%conv2d_transpose_242/stack/1:output:0%conv2d_transpose_242/stack/2:output:0%conv2d_transpose_242/stack/3:output:0*
N*
T0*
_output_shapes
:t
*conv2d_transpose_242/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,conv2d_transpose_242/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,conv2d_transpose_242/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$conv2d_transpose_242/strided_slice_1StridedSlice#conv2d_transpose_242/stack:output:03conv2d_transpose_242/strided_slice_1/stack:output:05conv2d_transpose_242/strided_slice_1/stack_1:output:05conv2d_transpose_242/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
4conv2d_transpose_242/conv2d_transpose/ReadVariableOpReadVariableOp=conv2d_transpose_242_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
%conv2d_transpose_242/conv2d_transposeConv2DBackpropInput#conv2d_transpose_242/stack:output:0<conv2d_transpose_242/conv2d_transpose/ReadVariableOp:value:0'conv2d_transpose_241/Relu:activations:0*
T0*/
_output_shapes
:?????????22 *
paddingSAME*
strides
?
+conv2d_transpose_242/BiasAdd/ReadVariableOpReadVariableOp4conv2d_transpose_242_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_242/BiasAddBiasAdd.conv2d_transpose_242/conv2d_transpose:output:03conv2d_transpose_242/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 ?
conv2d_transpose_242/ReluRelu%conv2d_transpose_242/BiasAdd:output:0*
T0*/
_output_shapes
:?????????22 q
conv2d_transpose_243/ShapeShape'conv2d_transpose_242/Relu:activations:0*
T0*
_output_shapes
:r
(conv2d_transpose_243/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_243/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_243/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_243/strided_sliceStridedSlice#conv2d_transpose_243/Shape:output:01conv2d_transpose_243/strided_slice/stack:output:03conv2d_transpose_243/strided_slice/stack_1:output:03conv2d_transpose_243/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_243/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d^
conv2d_transpose_243/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d^
conv2d_transpose_243/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_243/stackPack+conv2d_transpose_243/strided_slice:output:0%conv2d_transpose_243/stack/1:output:0%conv2d_transpose_243/stack/2:output:0%conv2d_transpose_243/stack/3:output:0*
N*
T0*
_output_shapes
:t
*conv2d_transpose_243/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,conv2d_transpose_243/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,conv2d_transpose_243/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$conv2d_transpose_243/strided_slice_1StridedSlice#conv2d_transpose_243/stack:output:03conv2d_transpose_243/strided_slice_1/stack:output:05conv2d_transpose_243/strided_slice_1/stack_1:output:05conv2d_transpose_243/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
4conv2d_transpose_243/conv2d_transpose/ReadVariableOpReadVariableOp=conv2d_transpose_243_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
%conv2d_transpose_243/conv2d_transposeConv2DBackpropInput#conv2d_transpose_243/stack:output:0<conv2d_transpose_243/conv2d_transpose/ReadVariableOp:value:0'conv2d_transpose_242/Relu:activations:0*
T0*/
_output_shapes
:?????????dd*
paddingSAME*
strides
?
+conv2d_transpose_243/BiasAdd/ReadVariableOpReadVariableOp4conv2d_transpose_243_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_243/BiasAddBiasAdd.conv2d_transpose_243/conv2d_transpose:output:03conv2d_transpose_243/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd?
conv2d_transpose_243/ReluRelu%conv2d_transpose_243/BiasAdd:output:0*
T0*/
_output_shapes
:?????????ddd
decoded/ShapeShape'conv2d_transpose_243/Relu:activations:0*
T0*
_output_shapes
:e
decoded/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
decoded/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
decoded/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
decoded/strided_sliceStridedSlicedecoded/Shape:output:0$decoded/strided_slice/stack:output:0&decoded/strided_slice/stack_1:output:0&decoded/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
decoded/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?R
decoded/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?Q
decoded/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
decoded/stackPackdecoded/strided_slice:output:0decoded/stack/1:output:0decoded/stack/2:output:0decoded/stack/3:output:0*
N*
T0*
_output_shapes
:g
decoded/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
decoded/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
decoded/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
decoded/strided_slice_1StridedSlicedecoded/stack:output:0&decoded/strided_slice_1/stack:output:0(decoded/strided_slice_1/stack_1:output:0(decoded/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
'decoded/conv2d_transpose/ReadVariableOpReadVariableOp0decoded_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
decoded/conv2d_transposeConv2DBackpropInputdecoded/stack:output:0/decoded/conv2d_transpose/ReadVariableOp:value:0'conv2d_transpose_243/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
decoded/BiasAdd/ReadVariableOpReadVariableOp'decoded_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
decoded/BiasAddBiasAdd!decoded/conv2d_transpose:output:0&decoded/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????p
decoded/SigmoidSigmoiddecoded/BiasAdd:output:0*
T0*1
_output_shapes
:???????????l
IdentityIdentitydecoded/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp'^batch_normalization_353/AssignNewValue)^batch_normalization_353/AssignNewValue_18^batch_normalization_353/FusedBatchNormV3/ReadVariableOp:^batch_normalization_353/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_353/ReadVariableOp)^batch_normalization_353/ReadVariableOp_1'^batch_normalization_354/AssignNewValue)^batch_normalization_354/AssignNewValue_18^batch_normalization_354/FusedBatchNormV3/ReadVariableOp:^batch_normalization_354/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_354/ReadVariableOp)^batch_normalization_354/ReadVariableOp_1'^batch_normalization_355/AssignNewValue)^batch_normalization_355/AssignNewValue_18^batch_normalization_355/FusedBatchNormV3/ReadVariableOp:^batch_normalization_355/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_355/ReadVariableOp)^batch_normalization_355/ReadVariableOp_1"^conv2d_293/BiasAdd/ReadVariableOp!^conv2d_293/Conv2D/ReadVariableOp"^conv2d_294/BiasAdd/ReadVariableOp!^conv2d_294/Conv2D/ReadVariableOp"^conv2d_295/BiasAdd/ReadVariableOp!^conv2d_295/Conv2D/ReadVariableOp"^conv2d_296/BiasAdd/ReadVariableOp!^conv2d_296/Conv2D/ReadVariableOp,^conv2d_transpose_241/BiasAdd/ReadVariableOp5^conv2d_transpose_241/conv2d_transpose/ReadVariableOp,^conv2d_transpose_242/BiasAdd/ReadVariableOp5^conv2d_transpose_242/conv2d_transpose/ReadVariableOp,^conv2d_transpose_243/BiasAdd/ReadVariableOp5^conv2d_transpose_243/conv2d_transpose/ReadVariableOp^decoded/BiasAdd/ReadVariableOp(^decoded/conv2d_transpose/ReadVariableOp^encoded/AssignNewValue^encoded/AssignNewValue_1(^encoded/FusedBatchNormV3/ReadVariableOp*^encoded/FusedBatchNormV3/ReadVariableOp_1^encoded/ReadVariableOp^encoded/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_353/AssignNewValue&batch_normalization_353/AssignNewValue2T
(batch_normalization_353/AssignNewValue_1(batch_normalization_353/AssignNewValue_12r
7batch_normalization_353/FusedBatchNormV3/ReadVariableOp7batch_normalization_353/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_353/FusedBatchNormV3/ReadVariableOp_19batch_normalization_353/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_353/ReadVariableOp&batch_normalization_353/ReadVariableOp2T
(batch_normalization_353/ReadVariableOp_1(batch_normalization_353/ReadVariableOp_12P
&batch_normalization_354/AssignNewValue&batch_normalization_354/AssignNewValue2T
(batch_normalization_354/AssignNewValue_1(batch_normalization_354/AssignNewValue_12r
7batch_normalization_354/FusedBatchNormV3/ReadVariableOp7batch_normalization_354/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_354/FusedBatchNormV3/ReadVariableOp_19batch_normalization_354/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_354/ReadVariableOp&batch_normalization_354/ReadVariableOp2T
(batch_normalization_354/ReadVariableOp_1(batch_normalization_354/ReadVariableOp_12P
&batch_normalization_355/AssignNewValue&batch_normalization_355/AssignNewValue2T
(batch_normalization_355/AssignNewValue_1(batch_normalization_355/AssignNewValue_12r
7batch_normalization_355/FusedBatchNormV3/ReadVariableOp7batch_normalization_355/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_355/FusedBatchNormV3/ReadVariableOp_19batch_normalization_355/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_355/ReadVariableOp&batch_normalization_355/ReadVariableOp2T
(batch_normalization_355/ReadVariableOp_1(batch_normalization_355/ReadVariableOp_12F
!conv2d_293/BiasAdd/ReadVariableOp!conv2d_293/BiasAdd/ReadVariableOp2D
 conv2d_293/Conv2D/ReadVariableOp conv2d_293/Conv2D/ReadVariableOp2F
!conv2d_294/BiasAdd/ReadVariableOp!conv2d_294/BiasAdd/ReadVariableOp2D
 conv2d_294/Conv2D/ReadVariableOp conv2d_294/Conv2D/ReadVariableOp2F
!conv2d_295/BiasAdd/ReadVariableOp!conv2d_295/BiasAdd/ReadVariableOp2D
 conv2d_295/Conv2D/ReadVariableOp conv2d_295/Conv2D/ReadVariableOp2F
!conv2d_296/BiasAdd/ReadVariableOp!conv2d_296/BiasAdd/ReadVariableOp2D
 conv2d_296/Conv2D/ReadVariableOp conv2d_296/Conv2D/ReadVariableOp2Z
+conv2d_transpose_241/BiasAdd/ReadVariableOp+conv2d_transpose_241/BiasAdd/ReadVariableOp2l
4conv2d_transpose_241/conv2d_transpose/ReadVariableOp4conv2d_transpose_241/conv2d_transpose/ReadVariableOp2Z
+conv2d_transpose_242/BiasAdd/ReadVariableOp+conv2d_transpose_242/BiasAdd/ReadVariableOp2l
4conv2d_transpose_242/conv2d_transpose/ReadVariableOp4conv2d_transpose_242/conv2d_transpose/ReadVariableOp2Z
+conv2d_transpose_243/BiasAdd/ReadVariableOp+conv2d_transpose_243/BiasAdd/ReadVariableOp2l
4conv2d_transpose_243/conv2d_transpose/ReadVariableOp4conv2d_transpose_243/conv2d_transpose/ReadVariableOp2@
decoded/BiasAdd/ReadVariableOpdecoded/BiasAdd/ReadVariableOp2R
'decoded/conv2d_transpose/ReadVariableOp'decoded/conv2d_transpose/ReadVariableOp20
encoded/AssignNewValueencoded/AssignNewValue24
encoded/AssignNewValue_1encoded/AssignNewValue_12R
'encoded/FusedBatchNormV3/ReadVariableOp'encoded/FusedBatchNormV3/ReadVariableOp2V
)encoded/FusedBatchNormV3/ReadVariableOp_1)encoded/FusedBatchNormV3/ReadVariableOp_120
encoded/ReadVariableOpencoded/ReadVariableOp24
encoded/ReadVariableOp_1encoded/ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?a
?
I__inference_sequential_36_layer_call_and_return_conditional_losses_314303

inputs+
conv2d_293_314218:
conv2d_293_314220:,
batch_normalization_353_314224:,
batch_normalization_353_314226:,
batch_normalization_353_314228:,
batch_normalization_353_314230:+
conv2d_294_314234: 
conv2d_294_314236: ,
batch_normalization_354_314240: ,
batch_normalization_354_314242: ,
batch_normalization_354_314244: ,
batch_normalization_354_314246: +
conv2d_295_314249: @
conv2d_295_314251:@,
batch_normalization_355_314255:@,
batch_normalization_355_314257:@,
batch_normalization_355_314259:@,
batch_normalization_355_314261:@+
conv2d_296_314265:@ 
conv2d_296_314267: 
encoded_314272: 
encoded_314274: 
encoded_314276: 
encoded_314278: 5
conv2d_transpose_241_314282:@ )
conv2d_transpose_241_314284:@5
conv2d_transpose_242_314287: @)
conv2d_transpose_242_314289: 5
conv2d_transpose_243_314292: )
conv2d_transpose_243_314294:(
decoded_314297:
decoded_314299:
identity??/batch_normalization_353/StatefulPartitionedCall?/batch_normalization_354/StatefulPartitionedCall?/batch_normalization_355/StatefulPartitionedCall?"conv2d_293/StatefulPartitionedCall?"conv2d_294/StatefulPartitionedCall?"conv2d_295/StatefulPartitionedCall?"conv2d_296/StatefulPartitionedCall?,conv2d_transpose_241/StatefulPartitionedCall?,conv2d_transpose_242/StatefulPartitionedCall?,conv2d_transpose_243/StatefulPartitionedCall?decoded/StatefulPartitionedCall?"dropout_94/StatefulPartitionedCall?"dropout_95/StatefulPartitionedCall?encoded/StatefulPartitionedCall?
"conv2d_293/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_293_314218conv2d_293_314220*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_293_layer_call_and_return_conditional_losses_313812?
activation_344/PartitionedCallPartitionedCall+conv2d_293/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_344_layer_call_and_return_conditional_losses_313823?
/batch_normalization_353/StatefulPartitionedCallStatefulPartitionedCall'activation_344/PartitionedCall:output:0batch_normalization_353_314224batch_normalization_353_314226batch_normalization_353_314228batch_normalization_353_314230*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_353_layer_call_and_return_conditional_losses_313408?
"dropout_94/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_353/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dropout_94_layer_call_and_return_conditional_losses_314125?
"conv2d_294/StatefulPartitionedCallStatefulPartitionedCall+dropout_94/StatefulPartitionedCall:output:0conv2d_294_314234conv2d_294_314236*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_294_layer_call_and_return_conditional_losses_313851?
activation_345/PartitionedCallPartitionedCall+conv2d_294/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_345_layer_call_and_return_conditional_losses_313862?
/batch_normalization_354/StatefulPartitionedCallStatefulPartitionedCall'activation_345/PartitionedCall:output:0batch_normalization_354_314240batch_normalization_354_314242batch_normalization_354_314244batch_normalization_354_314246*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_354_layer_call_and_return_conditional_losses_313472?
"conv2d_295/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_354/StatefulPartitionedCall:output:0conv2d_295_314249conv2d_295_314251*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_295_layer_call_and_return_conditional_losses_313883?
activation_346/PartitionedCallPartitionedCall+conv2d_295/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_346_layer_call_and_return_conditional_losses_313894?
/batch_normalization_355/StatefulPartitionedCallStatefulPartitionedCall'activation_346/PartitionedCall:output:0batch_normalization_355_314255batch_normalization_355_314257batch_normalization_355_314259batch_normalization_355_314261*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_355_layer_call_and_return_conditional_losses_313536?
"dropout_95/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_355/StatefulPartitionedCall:output:0#^dropout_94/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dropout_95_layer_call_and_return_conditional_losses_314070?
"conv2d_296/StatefulPartitionedCallStatefulPartitionedCall+dropout_95/StatefulPartitionedCall:output:0conv2d_296_314265conv2d_296_314267*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_296_layer_call_and_return_conditional_losses_313922?
activation_347/PartitionedCallPartitionedCall+conv2d_296/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_347_layer_call_and_return_conditional_losses_313933?
encoded/CastCast'activation_347/PartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
encoded/StatefulPartitionedCallStatefulPartitionedCallencoded/Cast:y:0encoded_314272encoded_314274encoded_314276encoded_314278*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_encoded_layer_call_and_return_conditional_losses_313600?
conv2d_transpose_241/CastCast(encoded/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
,conv2d_transpose_241/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_241/Cast:y:0conv2d_transpose_241_314282conv2d_transpose_241_314284*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_241_layer_call_and_return_conditional_losses_313653?
,conv2d_transpose_242/StatefulPartitionedCallStatefulPartitionedCall5conv2d_transpose_241/StatefulPartitionedCall:output:0conv2d_transpose_242_314287conv2d_transpose_242_314289*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_242_layer_call_and_return_conditional_losses_313698?
,conv2d_transpose_243/StatefulPartitionedCallStatefulPartitionedCall5conv2d_transpose_242/StatefulPartitionedCall:output:0conv2d_transpose_243_314292conv2d_transpose_243_314294*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_243_layer_call_and_return_conditional_losses_313743?
decoded/StatefulPartitionedCallStatefulPartitionedCall5conv2d_transpose_243/StatefulPartitionedCall:output:0decoded_314297decoded_314299*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_decoded_layer_call_and_return_conditional_losses_313788?
IdentityIdentity(decoded/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp0^batch_normalization_353/StatefulPartitionedCall0^batch_normalization_354/StatefulPartitionedCall0^batch_normalization_355/StatefulPartitionedCall#^conv2d_293/StatefulPartitionedCall#^conv2d_294/StatefulPartitionedCall#^conv2d_295/StatefulPartitionedCall#^conv2d_296/StatefulPartitionedCall-^conv2d_transpose_241/StatefulPartitionedCall-^conv2d_transpose_242/StatefulPartitionedCall-^conv2d_transpose_243/StatefulPartitionedCall ^decoded/StatefulPartitionedCall#^dropout_94/StatefulPartitionedCall#^dropout_95/StatefulPartitionedCall ^encoded/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_353/StatefulPartitionedCall/batch_normalization_353/StatefulPartitionedCall2b
/batch_normalization_354/StatefulPartitionedCall/batch_normalization_354/StatefulPartitionedCall2b
/batch_normalization_355/StatefulPartitionedCall/batch_normalization_355/StatefulPartitionedCall2H
"conv2d_293/StatefulPartitionedCall"conv2d_293/StatefulPartitionedCall2H
"conv2d_294/StatefulPartitionedCall"conv2d_294/StatefulPartitionedCall2H
"conv2d_295/StatefulPartitionedCall"conv2d_295/StatefulPartitionedCall2H
"conv2d_296/StatefulPartitionedCall"conv2d_296/StatefulPartitionedCall2\
,conv2d_transpose_241/StatefulPartitionedCall,conv2d_transpose_241/StatefulPartitionedCall2\
,conv2d_transpose_242/StatefulPartitionedCall,conv2d_transpose_242/StatefulPartitionedCall2\
,conv2d_transpose_243/StatefulPartitionedCall,conv2d_transpose_243/StatefulPartitionedCall2B
decoded/StatefulPartitionedCalldecoded/StatefulPartitionedCall2H
"dropout_94/StatefulPartitionedCall"dropout_94/StatefulPartitionedCall2H
"dropout_95/StatefulPartitionedCall"dropout_95/StatefulPartitionedCall2B
encoded/StatefulPartitionedCallencoded/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
F__inference_conv2d_296_layer_call_and_return_conditional_losses_313922

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
I__inference_sequential_36_layer_call_and_return_conditional_losses_315002

inputsC
)conv2d_293_conv2d_readvariableop_resource:8
*conv2d_293_biasadd_readvariableop_resource:=
/batch_normalization_353_readvariableop_resource:?
1batch_normalization_353_readvariableop_1_resource:N
@batch_normalization_353_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_353_fusedbatchnormv3_readvariableop_1_resource:C
)conv2d_294_conv2d_readvariableop_resource: 8
*conv2d_294_biasadd_readvariableop_resource: =
/batch_normalization_354_readvariableop_resource: ?
1batch_normalization_354_readvariableop_1_resource: N
@batch_normalization_354_fusedbatchnormv3_readvariableop_resource: P
Bbatch_normalization_354_fusedbatchnormv3_readvariableop_1_resource: C
)conv2d_295_conv2d_readvariableop_resource: @8
*conv2d_295_biasadd_readvariableop_resource:@=
/batch_normalization_355_readvariableop_resource:@?
1batch_normalization_355_readvariableop_1_resource:@N
@batch_normalization_355_fusedbatchnormv3_readvariableop_resource:@P
Bbatch_normalization_355_fusedbatchnormv3_readvariableop_1_resource:@C
)conv2d_296_conv2d_readvariableop_resource:@ 8
*conv2d_296_biasadd_readvariableop_resource: -
encoded_readvariableop_resource: /
!encoded_readvariableop_1_resource: >
0encoded_fusedbatchnormv3_readvariableop_resource: @
2encoded_fusedbatchnormv3_readvariableop_1_resource: W
=conv2d_transpose_241_conv2d_transpose_readvariableop_resource:@ B
4conv2d_transpose_241_biasadd_readvariableop_resource:@W
=conv2d_transpose_242_conv2d_transpose_readvariableop_resource: @B
4conv2d_transpose_242_biasadd_readvariableop_resource: W
=conv2d_transpose_243_conv2d_transpose_readvariableop_resource: B
4conv2d_transpose_243_biasadd_readvariableop_resource:J
0decoded_conv2d_transpose_readvariableop_resource:5
'decoded_biasadd_readvariableop_resource:
identity??7batch_normalization_353/FusedBatchNormV3/ReadVariableOp?9batch_normalization_353/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_353/ReadVariableOp?(batch_normalization_353/ReadVariableOp_1?7batch_normalization_354/FusedBatchNormV3/ReadVariableOp?9batch_normalization_354/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_354/ReadVariableOp?(batch_normalization_354/ReadVariableOp_1?7batch_normalization_355/FusedBatchNormV3/ReadVariableOp?9batch_normalization_355/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_355/ReadVariableOp?(batch_normalization_355/ReadVariableOp_1?!conv2d_293/BiasAdd/ReadVariableOp? conv2d_293/Conv2D/ReadVariableOp?!conv2d_294/BiasAdd/ReadVariableOp? conv2d_294/Conv2D/ReadVariableOp?!conv2d_295/BiasAdd/ReadVariableOp? conv2d_295/Conv2D/ReadVariableOp?!conv2d_296/BiasAdd/ReadVariableOp? conv2d_296/Conv2D/ReadVariableOp?+conv2d_transpose_241/BiasAdd/ReadVariableOp?4conv2d_transpose_241/conv2d_transpose/ReadVariableOp?+conv2d_transpose_242/BiasAdd/ReadVariableOp?4conv2d_transpose_242/conv2d_transpose/ReadVariableOp?+conv2d_transpose_243/BiasAdd/ReadVariableOp?4conv2d_transpose_243/conv2d_transpose/ReadVariableOp?decoded/BiasAdd/ReadVariableOp?'decoded/conv2d_transpose/ReadVariableOp?'encoded/FusedBatchNormV3/ReadVariableOp?)encoded/FusedBatchNormV3/ReadVariableOp_1?encoded/ReadVariableOp?encoded/ReadVariableOp_1?
 conv2d_293/Conv2D/ReadVariableOpReadVariableOp)conv2d_293_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_293/Conv2DConv2Dinputs(conv2d_293/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd*
paddingSAME*
strides
?
!conv2d_293/BiasAdd/ReadVariableOpReadVariableOp*conv2d_293_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_293/BiasAddBiasAddconv2d_293/Conv2D:output:0)conv2d_293/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ddr
activation_344/ReluReluconv2d_293/BiasAdd:output:0*
T0*/
_output_shapes
:?????????dd?
&batch_normalization_353/ReadVariableOpReadVariableOp/batch_normalization_353_readvariableop_resource*
_output_shapes
:*
dtype0?
(batch_normalization_353/ReadVariableOp_1ReadVariableOp1batch_normalization_353_readvariableop_1_resource*
_output_shapes
:*
dtype0?
7batch_normalization_353/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_353_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
9batch_normalization_353/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_353_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
(batch_normalization_353/FusedBatchNormV3FusedBatchNormV3!activation_344/Relu:activations:0.batch_normalization_353/ReadVariableOp:value:00batch_normalization_353/ReadVariableOp_1:value:0?batch_normalization_353/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_353/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????dd:::::*
epsilon%o?:*
is_training( ?
dropout_94/IdentityIdentity,batch_normalization_353/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????dd?
 conv2d_294/Conv2D/ReadVariableOpReadVariableOp)conv2d_294_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_294/Conv2DConv2Ddropout_94/Identity:output:0(conv2d_294/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 *
paddingSAME*
strides
?
!conv2d_294/BiasAdd/ReadVariableOpReadVariableOp*conv2d_294_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_294/BiasAddBiasAddconv2d_294/Conv2D:output:0)conv2d_294/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 r
activation_345/ReluReluconv2d_294/BiasAdd:output:0*
T0*/
_output_shapes
:?????????22 ?
&batch_normalization_354/ReadVariableOpReadVariableOp/batch_normalization_354_readvariableop_resource*
_output_shapes
: *
dtype0?
(batch_normalization_354/ReadVariableOp_1ReadVariableOp1batch_normalization_354_readvariableop_1_resource*
_output_shapes
: *
dtype0?
7batch_normalization_354/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_354_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
9batch_normalization_354/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_354_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
(batch_normalization_354/FusedBatchNormV3FusedBatchNormV3!activation_345/Relu:activations:0.batch_normalization_354/ReadVariableOp:value:00batch_normalization_354/ReadVariableOp_1:value:0?batch_normalization_354/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_354/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????22 : : : : :*
epsilon%o?:*
is_training( ?
 conv2d_295/Conv2D/ReadVariableOpReadVariableOp)conv2d_295_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_295/Conv2DConv2D,batch_normalization_354/FusedBatchNormV3:y:0(conv2d_295/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
!conv2d_295/BiasAdd/ReadVariableOpReadVariableOp*conv2d_295_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_295/BiasAddBiasAddconv2d_295/Conv2D:output:0)conv2d_295/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@r
activation_346/ReluReluconv2d_295/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
&batch_normalization_355/ReadVariableOpReadVariableOp/batch_normalization_355_readvariableop_resource*
_output_shapes
:@*
dtype0?
(batch_normalization_355/ReadVariableOp_1ReadVariableOp1batch_normalization_355_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
7batch_normalization_355/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_355_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
9batch_normalization_355/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_355_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
(batch_normalization_355/FusedBatchNormV3FusedBatchNormV3!activation_346/Relu:activations:0.batch_normalization_355/ReadVariableOp:value:00batch_normalization_355/ReadVariableOp_1:value:0?batch_normalization_355/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_355/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
dropout_95/IdentityIdentity,batch_normalization_355/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@?
 conv2d_296/Conv2D/ReadVariableOpReadVariableOp)conv2d_296_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
conv2d_296/Conv2DConv2Ddropout_95/Identity:output:0(conv2d_296/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
!conv2d_296/BiasAdd/ReadVariableOpReadVariableOp*conv2d_296_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_296/BiasAddBiasAddconv2d_296/Conv2D:output:0)conv2d_296/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? r
activation_347/ReluReluconv2d_296/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
encoded/CastCast!activation_347/Relu:activations:0*

DstT0*

SrcT0*/
_output_shapes
:????????? r
encoded/ReadVariableOpReadVariableOpencoded_readvariableop_resource*
_output_shapes
: *
dtype0v
encoded/ReadVariableOp_1ReadVariableOp!encoded_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'encoded/FusedBatchNormV3/ReadVariableOpReadVariableOp0encoded_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
)encoded/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp2encoded_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
encoded/FusedBatchNormV3FusedBatchNormV3encoded/Cast:y:0encoded/ReadVariableOp:value:0 encoded/ReadVariableOp_1:value:0/encoded/FusedBatchNormV3/ReadVariableOp:value:01encoded/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
conv2d_transpose_241/CastCastencoded/FusedBatchNormV3:y:0*

DstT0*

SrcT0*/
_output_shapes
:????????? g
conv2d_transpose_241/ShapeShapeconv2d_transpose_241/Cast:y:0*
T0*
_output_shapes
:r
(conv2d_transpose_241/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_241/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_241/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_241/strided_sliceStridedSlice#conv2d_transpose_241/Shape:output:01conv2d_transpose_241/strided_slice/stack:output:03conv2d_transpose_241/strided_slice/stack_1:output:03conv2d_transpose_241/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_241/stack/1Const*
_output_shapes
: *
dtype0*
value	B :^
conv2d_transpose_241/stack/2Const*
_output_shapes
: *
dtype0*
value	B :^
conv2d_transpose_241/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose_241/stackPack+conv2d_transpose_241/strided_slice:output:0%conv2d_transpose_241/stack/1:output:0%conv2d_transpose_241/stack/2:output:0%conv2d_transpose_241/stack/3:output:0*
N*
T0*
_output_shapes
:t
*conv2d_transpose_241/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,conv2d_transpose_241/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,conv2d_transpose_241/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$conv2d_transpose_241/strided_slice_1StridedSlice#conv2d_transpose_241/stack:output:03conv2d_transpose_241/strided_slice_1/stack:output:05conv2d_transpose_241/strided_slice_1/stack_1:output:05conv2d_transpose_241/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
4conv2d_transpose_241/conv2d_transpose/ReadVariableOpReadVariableOp=conv2d_transpose_241_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
%conv2d_transpose_241/conv2d_transposeConv2DBackpropInput#conv2d_transpose_241/stack:output:0<conv2d_transpose_241/conv2d_transpose/ReadVariableOp:value:0conv2d_transpose_241/Cast:y:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
+conv2d_transpose_241/BiasAdd/ReadVariableOpReadVariableOp4conv2d_transpose_241_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_transpose_241/BiasAddBiasAdd.conv2d_transpose_241/conv2d_transpose:output:03conv2d_transpose_241/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
conv2d_transpose_241/ReluRelu%conv2d_transpose_241/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@q
conv2d_transpose_242/ShapeShape'conv2d_transpose_241/Relu:activations:0*
T0*
_output_shapes
:r
(conv2d_transpose_242/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_242/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_242/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_242/strided_sliceStridedSlice#conv2d_transpose_242/Shape:output:01conv2d_transpose_242/strided_slice/stack:output:03conv2d_transpose_242/strided_slice/stack_1:output:03conv2d_transpose_242/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_242/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2^
conv2d_transpose_242/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2^
conv2d_transpose_242/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_242/stackPack+conv2d_transpose_242/strided_slice:output:0%conv2d_transpose_242/stack/1:output:0%conv2d_transpose_242/stack/2:output:0%conv2d_transpose_242/stack/3:output:0*
N*
T0*
_output_shapes
:t
*conv2d_transpose_242/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,conv2d_transpose_242/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,conv2d_transpose_242/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$conv2d_transpose_242/strided_slice_1StridedSlice#conv2d_transpose_242/stack:output:03conv2d_transpose_242/strided_slice_1/stack:output:05conv2d_transpose_242/strided_slice_1/stack_1:output:05conv2d_transpose_242/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
4conv2d_transpose_242/conv2d_transpose/ReadVariableOpReadVariableOp=conv2d_transpose_242_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
%conv2d_transpose_242/conv2d_transposeConv2DBackpropInput#conv2d_transpose_242/stack:output:0<conv2d_transpose_242/conv2d_transpose/ReadVariableOp:value:0'conv2d_transpose_241/Relu:activations:0*
T0*/
_output_shapes
:?????????22 *
paddingSAME*
strides
?
+conv2d_transpose_242/BiasAdd/ReadVariableOpReadVariableOp4conv2d_transpose_242_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_242/BiasAddBiasAdd.conv2d_transpose_242/conv2d_transpose:output:03conv2d_transpose_242/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 ?
conv2d_transpose_242/ReluRelu%conv2d_transpose_242/BiasAdd:output:0*
T0*/
_output_shapes
:?????????22 q
conv2d_transpose_243/ShapeShape'conv2d_transpose_242/Relu:activations:0*
T0*
_output_shapes
:r
(conv2d_transpose_243/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_243/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_243/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_243/strided_sliceStridedSlice#conv2d_transpose_243/Shape:output:01conv2d_transpose_243/strided_slice/stack:output:03conv2d_transpose_243/strided_slice/stack_1:output:03conv2d_transpose_243/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_243/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d^
conv2d_transpose_243/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d^
conv2d_transpose_243/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_243/stackPack+conv2d_transpose_243/strided_slice:output:0%conv2d_transpose_243/stack/1:output:0%conv2d_transpose_243/stack/2:output:0%conv2d_transpose_243/stack/3:output:0*
N*
T0*
_output_shapes
:t
*conv2d_transpose_243/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,conv2d_transpose_243/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,conv2d_transpose_243/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$conv2d_transpose_243/strided_slice_1StridedSlice#conv2d_transpose_243/stack:output:03conv2d_transpose_243/strided_slice_1/stack:output:05conv2d_transpose_243/strided_slice_1/stack_1:output:05conv2d_transpose_243/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
4conv2d_transpose_243/conv2d_transpose/ReadVariableOpReadVariableOp=conv2d_transpose_243_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
%conv2d_transpose_243/conv2d_transposeConv2DBackpropInput#conv2d_transpose_243/stack:output:0<conv2d_transpose_243/conv2d_transpose/ReadVariableOp:value:0'conv2d_transpose_242/Relu:activations:0*
T0*/
_output_shapes
:?????????dd*
paddingSAME*
strides
?
+conv2d_transpose_243/BiasAdd/ReadVariableOpReadVariableOp4conv2d_transpose_243_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_243/BiasAddBiasAdd.conv2d_transpose_243/conv2d_transpose:output:03conv2d_transpose_243/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd?
conv2d_transpose_243/ReluRelu%conv2d_transpose_243/BiasAdd:output:0*
T0*/
_output_shapes
:?????????ddd
decoded/ShapeShape'conv2d_transpose_243/Relu:activations:0*
T0*
_output_shapes
:e
decoded/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
decoded/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
decoded/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
decoded/strided_sliceStridedSlicedecoded/Shape:output:0$decoded/strided_slice/stack:output:0&decoded/strided_slice/stack_1:output:0&decoded/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
decoded/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?R
decoded/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?Q
decoded/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
decoded/stackPackdecoded/strided_slice:output:0decoded/stack/1:output:0decoded/stack/2:output:0decoded/stack/3:output:0*
N*
T0*
_output_shapes
:g
decoded/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
decoded/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
decoded/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
decoded/strided_slice_1StridedSlicedecoded/stack:output:0&decoded/strided_slice_1/stack:output:0(decoded/strided_slice_1/stack_1:output:0(decoded/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
'decoded/conv2d_transpose/ReadVariableOpReadVariableOp0decoded_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
decoded/conv2d_transposeConv2DBackpropInputdecoded/stack:output:0/decoded/conv2d_transpose/ReadVariableOp:value:0'conv2d_transpose_243/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
decoded/BiasAdd/ReadVariableOpReadVariableOp'decoded_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
decoded/BiasAddBiasAdd!decoded/conv2d_transpose:output:0&decoded/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????p
decoded/SigmoidSigmoiddecoded/BiasAdd:output:0*
T0*1
_output_shapes
:???????????l
IdentityIdentitydecoded/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp8^batch_normalization_353/FusedBatchNormV3/ReadVariableOp:^batch_normalization_353/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_353/ReadVariableOp)^batch_normalization_353/ReadVariableOp_18^batch_normalization_354/FusedBatchNormV3/ReadVariableOp:^batch_normalization_354/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_354/ReadVariableOp)^batch_normalization_354/ReadVariableOp_18^batch_normalization_355/FusedBatchNormV3/ReadVariableOp:^batch_normalization_355/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_355/ReadVariableOp)^batch_normalization_355/ReadVariableOp_1"^conv2d_293/BiasAdd/ReadVariableOp!^conv2d_293/Conv2D/ReadVariableOp"^conv2d_294/BiasAdd/ReadVariableOp!^conv2d_294/Conv2D/ReadVariableOp"^conv2d_295/BiasAdd/ReadVariableOp!^conv2d_295/Conv2D/ReadVariableOp"^conv2d_296/BiasAdd/ReadVariableOp!^conv2d_296/Conv2D/ReadVariableOp,^conv2d_transpose_241/BiasAdd/ReadVariableOp5^conv2d_transpose_241/conv2d_transpose/ReadVariableOp,^conv2d_transpose_242/BiasAdd/ReadVariableOp5^conv2d_transpose_242/conv2d_transpose/ReadVariableOp,^conv2d_transpose_243/BiasAdd/ReadVariableOp5^conv2d_transpose_243/conv2d_transpose/ReadVariableOp^decoded/BiasAdd/ReadVariableOp(^decoded/conv2d_transpose/ReadVariableOp(^encoded/FusedBatchNormV3/ReadVariableOp*^encoded/FusedBatchNormV3/ReadVariableOp_1^encoded/ReadVariableOp^encoded/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2r
7batch_normalization_353/FusedBatchNormV3/ReadVariableOp7batch_normalization_353/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_353/FusedBatchNormV3/ReadVariableOp_19batch_normalization_353/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_353/ReadVariableOp&batch_normalization_353/ReadVariableOp2T
(batch_normalization_353/ReadVariableOp_1(batch_normalization_353/ReadVariableOp_12r
7batch_normalization_354/FusedBatchNormV3/ReadVariableOp7batch_normalization_354/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_354/FusedBatchNormV3/ReadVariableOp_19batch_normalization_354/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_354/ReadVariableOp&batch_normalization_354/ReadVariableOp2T
(batch_normalization_354/ReadVariableOp_1(batch_normalization_354/ReadVariableOp_12r
7batch_normalization_355/FusedBatchNormV3/ReadVariableOp7batch_normalization_355/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_355/FusedBatchNormV3/ReadVariableOp_19batch_normalization_355/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_355/ReadVariableOp&batch_normalization_355/ReadVariableOp2T
(batch_normalization_355/ReadVariableOp_1(batch_normalization_355/ReadVariableOp_12F
!conv2d_293/BiasAdd/ReadVariableOp!conv2d_293/BiasAdd/ReadVariableOp2D
 conv2d_293/Conv2D/ReadVariableOp conv2d_293/Conv2D/ReadVariableOp2F
!conv2d_294/BiasAdd/ReadVariableOp!conv2d_294/BiasAdd/ReadVariableOp2D
 conv2d_294/Conv2D/ReadVariableOp conv2d_294/Conv2D/ReadVariableOp2F
!conv2d_295/BiasAdd/ReadVariableOp!conv2d_295/BiasAdd/ReadVariableOp2D
 conv2d_295/Conv2D/ReadVariableOp conv2d_295/Conv2D/ReadVariableOp2F
!conv2d_296/BiasAdd/ReadVariableOp!conv2d_296/BiasAdd/ReadVariableOp2D
 conv2d_296/Conv2D/ReadVariableOp conv2d_296/Conv2D/ReadVariableOp2Z
+conv2d_transpose_241/BiasAdd/ReadVariableOp+conv2d_transpose_241/BiasAdd/ReadVariableOp2l
4conv2d_transpose_241/conv2d_transpose/ReadVariableOp4conv2d_transpose_241/conv2d_transpose/ReadVariableOp2Z
+conv2d_transpose_242/BiasAdd/ReadVariableOp+conv2d_transpose_242/BiasAdd/ReadVariableOp2l
4conv2d_transpose_242/conv2d_transpose/ReadVariableOp4conv2d_transpose_242/conv2d_transpose/ReadVariableOp2Z
+conv2d_transpose_243/BiasAdd/ReadVariableOp+conv2d_transpose_243/BiasAdd/ReadVariableOp2l
4conv2d_transpose_243/conv2d_transpose/ReadVariableOp4conv2d_transpose_243/conv2d_transpose/ReadVariableOp2@
decoded/BiasAdd/ReadVariableOpdecoded/BiasAdd/ReadVariableOp2R
'decoded/conv2d_transpose/ReadVariableOp'decoded/conv2d_transpose/ReadVariableOp2R
'encoded/FusedBatchNormV3/ReadVariableOp'encoded/FusedBatchNormV3/ReadVariableOp2V
)encoded/FusedBatchNormV3/ReadVariableOp_1)encoded/FusedBatchNormV3/ReadVariableOp_120
encoded/ReadVariableOpencoded/ReadVariableOp24
encoded/ReadVariableOp_1encoded/ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?^
?
I__inference_sequential_36_layer_call_and_return_conditional_losses_313967

inputs+
conv2d_293_313813:
conv2d_293_313815:,
batch_normalization_353_313825:,
batch_normalization_353_313827:,
batch_normalization_353_313829:,
batch_normalization_353_313831:+
conv2d_294_313852: 
conv2d_294_313854: ,
batch_normalization_354_313864: ,
batch_normalization_354_313866: ,
batch_normalization_354_313868: ,
batch_normalization_354_313870: +
conv2d_295_313884: @
conv2d_295_313886:@,
batch_normalization_355_313896:@,
batch_normalization_355_313898:@,
batch_normalization_355_313900:@,
batch_normalization_355_313902:@+
conv2d_296_313923:@ 
conv2d_296_313925: 
encoded_313936: 
encoded_313938: 
encoded_313940: 
encoded_313942: 5
conv2d_transpose_241_313946:@ )
conv2d_transpose_241_313948:@5
conv2d_transpose_242_313951: @)
conv2d_transpose_242_313953: 5
conv2d_transpose_243_313956: )
conv2d_transpose_243_313958:(
decoded_313961:
decoded_313963:
identity??/batch_normalization_353/StatefulPartitionedCall?/batch_normalization_354/StatefulPartitionedCall?/batch_normalization_355/StatefulPartitionedCall?"conv2d_293/StatefulPartitionedCall?"conv2d_294/StatefulPartitionedCall?"conv2d_295/StatefulPartitionedCall?"conv2d_296/StatefulPartitionedCall?,conv2d_transpose_241/StatefulPartitionedCall?,conv2d_transpose_242/StatefulPartitionedCall?,conv2d_transpose_243/StatefulPartitionedCall?decoded/StatefulPartitionedCall?encoded/StatefulPartitionedCall?
"conv2d_293/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_293_313813conv2d_293_313815*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_293_layer_call_and_return_conditional_losses_313812?
activation_344/PartitionedCallPartitionedCall+conv2d_293/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_344_layer_call_and_return_conditional_losses_313823?
/batch_normalization_353/StatefulPartitionedCallStatefulPartitionedCall'activation_344/PartitionedCall:output:0batch_normalization_353_313825batch_normalization_353_313827batch_normalization_353_313829batch_normalization_353_313831*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_353_layer_call_and_return_conditional_losses_313377?
dropout_94/PartitionedCallPartitionedCall8batch_normalization_353/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dropout_94_layer_call_and_return_conditional_losses_313839?
"conv2d_294/StatefulPartitionedCallStatefulPartitionedCall#dropout_94/PartitionedCall:output:0conv2d_294_313852conv2d_294_313854*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_294_layer_call_and_return_conditional_losses_313851?
activation_345/PartitionedCallPartitionedCall+conv2d_294/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_345_layer_call_and_return_conditional_losses_313862?
/batch_normalization_354/StatefulPartitionedCallStatefulPartitionedCall'activation_345/PartitionedCall:output:0batch_normalization_354_313864batch_normalization_354_313866batch_normalization_354_313868batch_normalization_354_313870*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_354_layer_call_and_return_conditional_losses_313441?
"conv2d_295/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_354/StatefulPartitionedCall:output:0conv2d_295_313884conv2d_295_313886*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_295_layer_call_and_return_conditional_losses_313883?
activation_346/PartitionedCallPartitionedCall+conv2d_295/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_346_layer_call_and_return_conditional_losses_313894?
/batch_normalization_355/StatefulPartitionedCallStatefulPartitionedCall'activation_346/PartitionedCall:output:0batch_normalization_355_313896batch_normalization_355_313898batch_normalization_355_313900batch_normalization_355_313902*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_355_layer_call_and_return_conditional_losses_313505?
dropout_95/PartitionedCallPartitionedCall8batch_normalization_355/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dropout_95_layer_call_and_return_conditional_losses_313910?
"conv2d_296/StatefulPartitionedCallStatefulPartitionedCall#dropout_95/PartitionedCall:output:0conv2d_296_313923conv2d_296_313925*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_296_layer_call_and_return_conditional_losses_313922?
activation_347/PartitionedCallPartitionedCall+conv2d_296/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_347_layer_call_and_return_conditional_losses_313933?
encoded/CastCast'activation_347/PartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
encoded/StatefulPartitionedCallStatefulPartitionedCallencoded/Cast:y:0encoded_313936encoded_313938encoded_313940encoded_313942*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_encoded_layer_call_and_return_conditional_losses_313569?
conv2d_transpose_241/CastCast(encoded/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
,conv2d_transpose_241/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_241/Cast:y:0conv2d_transpose_241_313946conv2d_transpose_241_313948*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_241_layer_call_and_return_conditional_losses_313653?
,conv2d_transpose_242/StatefulPartitionedCallStatefulPartitionedCall5conv2d_transpose_241/StatefulPartitionedCall:output:0conv2d_transpose_242_313951conv2d_transpose_242_313953*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_242_layer_call_and_return_conditional_losses_313698?
,conv2d_transpose_243/StatefulPartitionedCallStatefulPartitionedCall5conv2d_transpose_242/StatefulPartitionedCall:output:0conv2d_transpose_243_313956conv2d_transpose_243_313958*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_243_layer_call_and_return_conditional_losses_313743?
decoded/StatefulPartitionedCallStatefulPartitionedCall5conv2d_transpose_243/StatefulPartitionedCall:output:0decoded_313961decoded_313963*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_decoded_layer_call_and_return_conditional_losses_313788?
IdentityIdentity(decoded/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp0^batch_normalization_353/StatefulPartitionedCall0^batch_normalization_354/StatefulPartitionedCall0^batch_normalization_355/StatefulPartitionedCall#^conv2d_293/StatefulPartitionedCall#^conv2d_294/StatefulPartitionedCall#^conv2d_295/StatefulPartitionedCall#^conv2d_296/StatefulPartitionedCall-^conv2d_transpose_241/StatefulPartitionedCall-^conv2d_transpose_242/StatefulPartitionedCall-^conv2d_transpose_243/StatefulPartitionedCall ^decoded/StatefulPartitionedCall ^encoded/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_353/StatefulPartitionedCall/batch_normalization_353/StatefulPartitionedCall2b
/batch_normalization_354/StatefulPartitionedCall/batch_normalization_354/StatefulPartitionedCall2b
/batch_normalization_355/StatefulPartitionedCall/batch_normalization_355/StatefulPartitionedCall2H
"conv2d_293/StatefulPartitionedCall"conv2d_293/StatefulPartitionedCall2H
"conv2d_294/StatefulPartitionedCall"conv2d_294/StatefulPartitionedCall2H
"conv2d_295/StatefulPartitionedCall"conv2d_295/StatefulPartitionedCall2H
"conv2d_296/StatefulPartitionedCall"conv2d_296/StatefulPartitionedCall2\
,conv2d_transpose_241/StatefulPartitionedCall,conv2d_transpose_241/StatefulPartitionedCall2\
,conv2d_transpose_242/StatefulPartitionedCall,conv2d_transpose_242/StatefulPartitionedCall2\
,conv2d_transpose_243/StatefulPartitionedCall,conv2d_transpose_243/StatefulPartitionedCall2B
decoded/StatefulPartitionedCalldecoded/StatefulPartitionedCall2B
encoded/StatefulPartitionedCallencoded/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
f
J__inference_activation_345_layer_call_and_return_conditional_losses_315335

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????22 b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????22 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22 :W S
/
_output_shapes
:?????????22 
 
_user_specified_nameinputs
?

e
F__inference_dropout_95_layer_call_and_return_conditional_losses_315515

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_353_layer_call_and_return_conditional_losses_313377

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_354_layer_call_and_return_conditional_losses_315379

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
.__inference_sequential_36_layer_call_fn_314439
conv2d_293_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@ 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23:@ 

unknown_24:@$

unknown_25: @

unknown_26: $

unknown_27: 

unknown_28:$

unknown_29:

unknown_30:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_293_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*:
_read_only_resource_inputs
	
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_sequential_36_layer_call_and_return_conditional_losses_314303y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
1
_output_shapes
:???????????
*
_user_specified_nameconv2d_293_input
?
f
J__inference_activation_344_layer_call_and_return_conditional_losses_315217

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????ddb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????dd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????dd:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_314692
conv2d_293_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@ 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23:@ 

unknown_24:@$

unknown_25: @

unknown_26: $

unknown_27: 

unknown_28:$

unknown_29:

unknown_30:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_293_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2 *0J 8? **
f%R#
!__inference__wrapped_model_313355y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
1
_output_shapes
:???????????
*
_user_specified_nameconv2d_293_input
?
?
+__inference_conv2d_293_layer_call_fn_315197

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_293_layer_call_and_return_conditional_losses_313812w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????dd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_94_layer_call_fn_315284

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dropout_94_layer_call_and_return_conditional_losses_313839h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????dd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????dd:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_242_layer_call_fn_315662

inputs!
unknown: @
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_242_layer_call_and_return_conditional_losses_313698?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
d
F__inference_dropout_94_layer_call_and_return_conditional_losses_313839

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????ddc

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????dd"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????dd:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
?
C__inference_encoded_layer_call_and_return_conditional_losses_313600

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
+__inference_conv2d_294_layer_call_fn_315315

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_294_layer_call_and_return_conditional_losses_313851w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????22 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
??
?#
!__inference__wrapped_model_313355
conv2d_293_inputQ
7sequential_36_conv2d_293_conv2d_readvariableop_resource:F
8sequential_36_conv2d_293_biasadd_readvariableop_resource:K
=sequential_36_batch_normalization_353_readvariableop_resource:M
?sequential_36_batch_normalization_353_readvariableop_1_resource:\
Nsequential_36_batch_normalization_353_fusedbatchnormv3_readvariableop_resource:^
Psequential_36_batch_normalization_353_fusedbatchnormv3_readvariableop_1_resource:Q
7sequential_36_conv2d_294_conv2d_readvariableop_resource: F
8sequential_36_conv2d_294_biasadd_readvariableop_resource: K
=sequential_36_batch_normalization_354_readvariableop_resource: M
?sequential_36_batch_normalization_354_readvariableop_1_resource: \
Nsequential_36_batch_normalization_354_fusedbatchnormv3_readvariableop_resource: ^
Psequential_36_batch_normalization_354_fusedbatchnormv3_readvariableop_1_resource: Q
7sequential_36_conv2d_295_conv2d_readvariableop_resource: @F
8sequential_36_conv2d_295_biasadd_readvariableop_resource:@K
=sequential_36_batch_normalization_355_readvariableop_resource:@M
?sequential_36_batch_normalization_355_readvariableop_1_resource:@\
Nsequential_36_batch_normalization_355_fusedbatchnormv3_readvariableop_resource:@^
Psequential_36_batch_normalization_355_fusedbatchnormv3_readvariableop_1_resource:@Q
7sequential_36_conv2d_296_conv2d_readvariableop_resource:@ F
8sequential_36_conv2d_296_biasadd_readvariableop_resource: ;
-sequential_36_encoded_readvariableop_resource: =
/sequential_36_encoded_readvariableop_1_resource: L
>sequential_36_encoded_fusedbatchnormv3_readvariableop_resource: N
@sequential_36_encoded_fusedbatchnormv3_readvariableop_1_resource: e
Ksequential_36_conv2d_transpose_241_conv2d_transpose_readvariableop_resource:@ P
Bsequential_36_conv2d_transpose_241_biasadd_readvariableop_resource:@e
Ksequential_36_conv2d_transpose_242_conv2d_transpose_readvariableop_resource: @P
Bsequential_36_conv2d_transpose_242_biasadd_readvariableop_resource: e
Ksequential_36_conv2d_transpose_243_conv2d_transpose_readvariableop_resource: P
Bsequential_36_conv2d_transpose_243_biasadd_readvariableop_resource:X
>sequential_36_decoded_conv2d_transpose_readvariableop_resource:C
5sequential_36_decoded_biasadd_readvariableop_resource:
identity??Esequential_36/batch_normalization_353/FusedBatchNormV3/ReadVariableOp?Gsequential_36/batch_normalization_353/FusedBatchNormV3/ReadVariableOp_1?4sequential_36/batch_normalization_353/ReadVariableOp?6sequential_36/batch_normalization_353/ReadVariableOp_1?Esequential_36/batch_normalization_354/FusedBatchNormV3/ReadVariableOp?Gsequential_36/batch_normalization_354/FusedBatchNormV3/ReadVariableOp_1?4sequential_36/batch_normalization_354/ReadVariableOp?6sequential_36/batch_normalization_354/ReadVariableOp_1?Esequential_36/batch_normalization_355/FusedBatchNormV3/ReadVariableOp?Gsequential_36/batch_normalization_355/FusedBatchNormV3/ReadVariableOp_1?4sequential_36/batch_normalization_355/ReadVariableOp?6sequential_36/batch_normalization_355/ReadVariableOp_1?/sequential_36/conv2d_293/BiasAdd/ReadVariableOp?.sequential_36/conv2d_293/Conv2D/ReadVariableOp?/sequential_36/conv2d_294/BiasAdd/ReadVariableOp?.sequential_36/conv2d_294/Conv2D/ReadVariableOp?/sequential_36/conv2d_295/BiasAdd/ReadVariableOp?.sequential_36/conv2d_295/Conv2D/ReadVariableOp?/sequential_36/conv2d_296/BiasAdd/ReadVariableOp?.sequential_36/conv2d_296/Conv2D/ReadVariableOp?9sequential_36/conv2d_transpose_241/BiasAdd/ReadVariableOp?Bsequential_36/conv2d_transpose_241/conv2d_transpose/ReadVariableOp?9sequential_36/conv2d_transpose_242/BiasAdd/ReadVariableOp?Bsequential_36/conv2d_transpose_242/conv2d_transpose/ReadVariableOp?9sequential_36/conv2d_transpose_243/BiasAdd/ReadVariableOp?Bsequential_36/conv2d_transpose_243/conv2d_transpose/ReadVariableOp?,sequential_36/decoded/BiasAdd/ReadVariableOp?5sequential_36/decoded/conv2d_transpose/ReadVariableOp?5sequential_36/encoded/FusedBatchNormV3/ReadVariableOp?7sequential_36/encoded/FusedBatchNormV3/ReadVariableOp_1?$sequential_36/encoded/ReadVariableOp?&sequential_36/encoded/ReadVariableOp_1?
.sequential_36/conv2d_293/Conv2D/ReadVariableOpReadVariableOp7sequential_36_conv2d_293_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_36/conv2d_293/Conv2DConv2Dconv2d_293_input6sequential_36/conv2d_293/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd*
paddingSAME*
strides
?
/sequential_36/conv2d_293/BiasAdd/ReadVariableOpReadVariableOp8sequential_36_conv2d_293_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
 sequential_36/conv2d_293/BiasAddBiasAdd(sequential_36/conv2d_293/Conv2D:output:07sequential_36/conv2d_293/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd?
!sequential_36/activation_344/ReluRelu)sequential_36/conv2d_293/BiasAdd:output:0*
T0*/
_output_shapes
:?????????dd?
4sequential_36/batch_normalization_353/ReadVariableOpReadVariableOp=sequential_36_batch_normalization_353_readvariableop_resource*
_output_shapes
:*
dtype0?
6sequential_36/batch_normalization_353/ReadVariableOp_1ReadVariableOp?sequential_36_batch_normalization_353_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Esequential_36/batch_normalization_353/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_36_batch_normalization_353_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Gsequential_36/batch_normalization_353/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_36_batch_normalization_353_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6sequential_36/batch_normalization_353/FusedBatchNormV3FusedBatchNormV3/sequential_36/activation_344/Relu:activations:0<sequential_36/batch_normalization_353/ReadVariableOp:value:0>sequential_36/batch_normalization_353/ReadVariableOp_1:value:0Msequential_36/batch_normalization_353/FusedBatchNormV3/ReadVariableOp:value:0Osequential_36/batch_normalization_353/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????dd:::::*
epsilon%o?:*
is_training( ?
!sequential_36/dropout_94/IdentityIdentity:sequential_36/batch_normalization_353/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????dd?
.sequential_36/conv2d_294/Conv2D/ReadVariableOpReadVariableOp7sequential_36_conv2d_294_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
sequential_36/conv2d_294/Conv2DConv2D*sequential_36/dropout_94/Identity:output:06sequential_36/conv2d_294/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 *
paddingSAME*
strides
?
/sequential_36/conv2d_294/BiasAdd/ReadVariableOpReadVariableOp8sequential_36_conv2d_294_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
 sequential_36/conv2d_294/BiasAddBiasAdd(sequential_36/conv2d_294/Conv2D:output:07sequential_36/conv2d_294/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 ?
!sequential_36/activation_345/ReluRelu)sequential_36/conv2d_294/BiasAdd:output:0*
T0*/
_output_shapes
:?????????22 ?
4sequential_36/batch_normalization_354/ReadVariableOpReadVariableOp=sequential_36_batch_normalization_354_readvariableop_resource*
_output_shapes
: *
dtype0?
6sequential_36/batch_normalization_354/ReadVariableOp_1ReadVariableOp?sequential_36_batch_normalization_354_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Esequential_36/batch_normalization_354/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_36_batch_normalization_354_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Gsequential_36/batch_normalization_354/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_36_batch_normalization_354_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6sequential_36/batch_normalization_354/FusedBatchNormV3FusedBatchNormV3/sequential_36/activation_345/Relu:activations:0<sequential_36/batch_normalization_354/ReadVariableOp:value:0>sequential_36/batch_normalization_354/ReadVariableOp_1:value:0Msequential_36/batch_normalization_354/FusedBatchNormV3/ReadVariableOp:value:0Osequential_36/batch_normalization_354/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????22 : : : : :*
epsilon%o?:*
is_training( ?
.sequential_36/conv2d_295/Conv2D/ReadVariableOpReadVariableOp7sequential_36_conv2d_295_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
sequential_36/conv2d_295/Conv2DConv2D:sequential_36/batch_normalization_354/FusedBatchNormV3:y:06sequential_36/conv2d_295/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
/sequential_36/conv2d_295/BiasAdd/ReadVariableOpReadVariableOp8sequential_36_conv2d_295_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
 sequential_36/conv2d_295/BiasAddBiasAdd(sequential_36/conv2d_295/Conv2D:output:07sequential_36/conv2d_295/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
!sequential_36/activation_346/ReluRelu)sequential_36/conv2d_295/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
4sequential_36/batch_normalization_355/ReadVariableOpReadVariableOp=sequential_36_batch_normalization_355_readvariableop_resource*
_output_shapes
:@*
dtype0?
6sequential_36/batch_normalization_355/ReadVariableOp_1ReadVariableOp?sequential_36_batch_normalization_355_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Esequential_36/batch_normalization_355/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_36_batch_normalization_355_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Gsequential_36/batch_normalization_355/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_36_batch_normalization_355_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6sequential_36/batch_normalization_355/FusedBatchNormV3FusedBatchNormV3/sequential_36/activation_346/Relu:activations:0<sequential_36/batch_normalization_355/ReadVariableOp:value:0>sequential_36/batch_normalization_355/ReadVariableOp_1:value:0Msequential_36/batch_normalization_355/FusedBatchNormV3/ReadVariableOp:value:0Osequential_36/batch_normalization_355/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
!sequential_36/dropout_95/IdentityIdentity:sequential_36/batch_normalization_355/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@?
.sequential_36/conv2d_296/Conv2D/ReadVariableOpReadVariableOp7sequential_36_conv2d_296_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
sequential_36/conv2d_296/Conv2DConv2D*sequential_36/dropout_95/Identity:output:06sequential_36/conv2d_296/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
/sequential_36/conv2d_296/BiasAdd/ReadVariableOpReadVariableOp8sequential_36_conv2d_296_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
 sequential_36/conv2d_296/BiasAddBiasAdd(sequential_36/conv2d_296/Conv2D:output:07sequential_36/conv2d_296/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
!sequential_36/activation_347/ReluRelu)sequential_36/conv2d_296/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
sequential_36/encoded/CastCast/sequential_36/activation_347/Relu:activations:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
$sequential_36/encoded/ReadVariableOpReadVariableOp-sequential_36_encoded_readvariableop_resource*
_output_shapes
: *
dtype0?
&sequential_36/encoded/ReadVariableOp_1ReadVariableOp/sequential_36_encoded_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5sequential_36/encoded/FusedBatchNormV3/ReadVariableOpReadVariableOp>sequential_36_encoded_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
7sequential_36/encoded/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@sequential_36_encoded_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
&sequential_36/encoded/FusedBatchNormV3FusedBatchNormV3sequential_36/encoded/Cast:y:0,sequential_36/encoded/ReadVariableOp:value:0.sequential_36/encoded/ReadVariableOp_1:value:0=sequential_36/encoded/FusedBatchNormV3/ReadVariableOp:value:0?sequential_36/encoded/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
'sequential_36/conv2d_transpose_241/CastCast*sequential_36/encoded/FusedBatchNormV3:y:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
(sequential_36/conv2d_transpose_241/ShapeShape+sequential_36/conv2d_transpose_241/Cast:y:0*
T0*
_output_shapes
:?
6sequential_36/conv2d_transpose_241/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8sequential_36/conv2d_transpose_241/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_36/conv2d_transpose_241/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0sequential_36/conv2d_transpose_241/strided_sliceStridedSlice1sequential_36/conv2d_transpose_241/Shape:output:0?sequential_36/conv2d_transpose_241/strided_slice/stack:output:0Asequential_36/conv2d_transpose_241/strided_slice/stack_1:output:0Asequential_36/conv2d_transpose_241/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*sequential_36/conv2d_transpose_241/stack/1Const*
_output_shapes
: *
dtype0*
value	B :l
*sequential_36/conv2d_transpose_241/stack/2Const*
_output_shapes
: *
dtype0*
value	B :l
*sequential_36/conv2d_transpose_241/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
(sequential_36/conv2d_transpose_241/stackPack9sequential_36/conv2d_transpose_241/strided_slice:output:03sequential_36/conv2d_transpose_241/stack/1:output:03sequential_36/conv2d_transpose_241/stack/2:output:03sequential_36/conv2d_transpose_241/stack/3:output:0*
N*
T0*
_output_shapes
:?
8sequential_36/conv2d_transpose_241/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:sequential_36/conv2d_transpose_241/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:sequential_36/conv2d_transpose_241/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2sequential_36/conv2d_transpose_241/strided_slice_1StridedSlice1sequential_36/conv2d_transpose_241/stack:output:0Asequential_36/conv2d_transpose_241/strided_slice_1/stack:output:0Csequential_36/conv2d_transpose_241/strided_slice_1/stack_1:output:0Csequential_36/conv2d_transpose_241/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Bsequential_36/conv2d_transpose_241/conv2d_transpose/ReadVariableOpReadVariableOpKsequential_36_conv2d_transpose_241_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
3sequential_36/conv2d_transpose_241/conv2d_transposeConv2DBackpropInput1sequential_36/conv2d_transpose_241/stack:output:0Jsequential_36/conv2d_transpose_241/conv2d_transpose/ReadVariableOp:value:0+sequential_36/conv2d_transpose_241/Cast:y:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
9sequential_36/conv2d_transpose_241/BiasAdd/ReadVariableOpReadVariableOpBsequential_36_conv2d_transpose_241_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
*sequential_36/conv2d_transpose_241/BiasAddBiasAdd<sequential_36/conv2d_transpose_241/conv2d_transpose:output:0Asequential_36/conv2d_transpose_241/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
'sequential_36/conv2d_transpose_241/ReluRelu3sequential_36/conv2d_transpose_241/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
(sequential_36/conv2d_transpose_242/ShapeShape5sequential_36/conv2d_transpose_241/Relu:activations:0*
T0*
_output_shapes
:?
6sequential_36/conv2d_transpose_242/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8sequential_36/conv2d_transpose_242/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_36/conv2d_transpose_242/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0sequential_36/conv2d_transpose_242/strided_sliceStridedSlice1sequential_36/conv2d_transpose_242/Shape:output:0?sequential_36/conv2d_transpose_242/strided_slice/stack:output:0Asequential_36/conv2d_transpose_242/strided_slice/stack_1:output:0Asequential_36/conv2d_transpose_242/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*sequential_36/conv2d_transpose_242/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2l
*sequential_36/conv2d_transpose_242/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2l
*sequential_36/conv2d_transpose_242/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_36/conv2d_transpose_242/stackPack9sequential_36/conv2d_transpose_242/strided_slice:output:03sequential_36/conv2d_transpose_242/stack/1:output:03sequential_36/conv2d_transpose_242/stack/2:output:03sequential_36/conv2d_transpose_242/stack/3:output:0*
N*
T0*
_output_shapes
:?
8sequential_36/conv2d_transpose_242/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:sequential_36/conv2d_transpose_242/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:sequential_36/conv2d_transpose_242/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2sequential_36/conv2d_transpose_242/strided_slice_1StridedSlice1sequential_36/conv2d_transpose_242/stack:output:0Asequential_36/conv2d_transpose_242/strided_slice_1/stack:output:0Csequential_36/conv2d_transpose_242/strided_slice_1/stack_1:output:0Csequential_36/conv2d_transpose_242/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Bsequential_36/conv2d_transpose_242/conv2d_transpose/ReadVariableOpReadVariableOpKsequential_36_conv2d_transpose_242_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
3sequential_36/conv2d_transpose_242/conv2d_transposeConv2DBackpropInput1sequential_36/conv2d_transpose_242/stack:output:0Jsequential_36/conv2d_transpose_242/conv2d_transpose/ReadVariableOp:value:05sequential_36/conv2d_transpose_241/Relu:activations:0*
T0*/
_output_shapes
:?????????22 *
paddingSAME*
strides
?
9sequential_36/conv2d_transpose_242/BiasAdd/ReadVariableOpReadVariableOpBsequential_36_conv2d_transpose_242_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
*sequential_36/conv2d_transpose_242/BiasAddBiasAdd<sequential_36/conv2d_transpose_242/conv2d_transpose:output:0Asequential_36/conv2d_transpose_242/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????22 ?
'sequential_36/conv2d_transpose_242/ReluRelu3sequential_36/conv2d_transpose_242/BiasAdd:output:0*
T0*/
_output_shapes
:?????????22 ?
(sequential_36/conv2d_transpose_243/ShapeShape5sequential_36/conv2d_transpose_242/Relu:activations:0*
T0*
_output_shapes
:?
6sequential_36/conv2d_transpose_243/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8sequential_36/conv2d_transpose_243/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_36/conv2d_transpose_243/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0sequential_36/conv2d_transpose_243/strided_sliceStridedSlice1sequential_36/conv2d_transpose_243/Shape:output:0?sequential_36/conv2d_transpose_243/strided_slice/stack:output:0Asequential_36/conv2d_transpose_243/strided_slice/stack_1:output:0Asequential_36/conv2d_transpose_243/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*sequential_36/conv2d_transpose_243/stack/1Const*
_output_shapes
: *
dtype0*
value	B :dl
*sequential_36/conv2d_transpose_243/stack/2Const*
_output_shapes
: *
dtype0*
value	B :dl
*sequential_36/conv2d_transpose_243/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
(sequential_36/conv2d_transpose_243/stackPack9sequential_36/conv2d_transpose_243/strided_slice:output:03sequential_36/conv2d_transpose_243/stack/1:output:03sequential_36/conv2d_transpose_243/stack/2:output:03sequential_36/conv2d_transpose_243/stack/3:output:0*
N*
T0*
_output_shapes
:?
8sequential_36/conv2d_transpose_243/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:sequential_36/conv2d_transpose_243/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:sequential_36/conv2d_transpose_243/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2sequential_36/conv2d_transpose_243/strided_slice_1StridedSlice1sequential_36/conv2d_transpose_243/stack:output:0Asequential_36/conv2d_transpose_243/strided_slice_1/stack:output:0Csequential_36/conv2d_transpose_243/strided_slice_1/stack_1:output:0Csequential_36/conv2d_transpose_243/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Bsequential_36/conv2d_transpose_243/conv2d_transpose/ReadVariableOpReadVariableOpKsequential_36_conv2d_transpose_243_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
3sequential_36/conv2d_transpose_243/conv2d_transposeConv2DBackpropInput1sequential_36/conv2d_transpose_243/stack:output:0Jsequential_36/conv2d_transpose_243/conv2d_transpose/ReadVariableOp:value:05sequential_36/conv2d_transpose_242/Relu:activations:0*
T0*/
_output_shapes
:?????????dd*
paddingSAME*
strides
?
9sequential_36/conv2d_transpose_243/BiasAdd/ReadVariableOpReadVariableOpBsequential_36_conv2d_transpose_243_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
*sequential_36/conv2d_transpose_243/BiasAddBiasAdd<sequential_36/conv2d_transpose_243/conv2d_transpose:output:0Asequential_36/conv2d_transpose_243/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????dd?
'sequential_36/conv2d_transpose_243/ReluRelu3sequential_36/conv2d_transpose_243/BiasAdd:output:0*
T0*/
_output_shapes
:?????????dd?
sequential_36/decoded/ShapeShape5sequential_36/conv2d_transpose_243/Relu:activations:0*
T0*
_output_shapes
:s
)sequential_36/decoded/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_36/decoded/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_36/decoded/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#sequential_36/decoded/strided_sliceStridedSlice$sequential_36/decoded/Shape:output:02sequential_36/decoded/strided_slice/stack:output:04sequential_36/decoded/strided_slice/stack_1:output:04sequential_36/decoded/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
sequential_36/decoded/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?`
sequential_36/decoded/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?_
sequential_36/decoded/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
sequential_36/decoded/stackPack,sequential_36/decoded/strided_slice:output:0&sequential_36/decoded/stack/1:output:0&sequential_36/decoded/stack/2:output:0&sequential_36/decoded/stack/3:output:0*
N*
T0*
_output_shapes
:u
+sequential_36/decoded/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_36/decoded/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_36/decoded/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%sequential_36/decoded/strided_slice_1StridedSlice$sequential_36/decoded/stack:output:04sequential_36/decoded/strided_slice_1/stack:output:06sequential_36/decoded/strided_slice_1/stack_1:output:06sequential_36/decoded/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
5sequential_36/decoded/conv2d_transpose/ReadVariableOpReadVariableOp>sequential_36_decoded_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
&sequential_36/decoded/conv2d_transposeConv2DBackpropInput$sequential_36/decoded/stack:output:0=sequential_36/decoded/conv2d_transpose/ReadVariableOp:value:05sequential_36/conv2d_transpose_243/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
,sequential_36/decoded/BiasAdd/ReadVariableOpReadVariableOp5sequential_36_decoded_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_36/decoded/BiasAddBiasAdd/sequential_36/decoded/conv2d_transpose:output:04sequential_36/decoded/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
sequential_36/decoded/SigmoidSigmoid&sequential_36/decoded/BiasAdd:output:0*
T0*1
_output_shapes
:???????????z
IdentityIdentity!sequential_36/decoded/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOpF^sequential_36/batch_normalization_353/FusedBatchNormV3/ReadVariableOpH^sequential_36/batch_normalization_353/FusedBatchNormV3/ReadVariableOp_15^sequential_36/batch_normalization_353/ReadVariableOp7^sequential_36/batch_normalization_353/ReadVariableOp_1F^sequential_36/batch_normalization_354/FusedBatchNormV3/ReadVariableOpH^sequential_36/batch_normalization_354/FusedBatchNormV3/ReadVariableOp_15^sequential_36/batch_normalization_354/ReadVariableOp7^sequential_36/batch_normalization_354/ReadVariableOp_1F^sequential_36/batch_normalization_355/FusedBatchNormV3/ReadVariableOpH^sequential_36/batch_normalization_355/FusedBatchNormV3/ReadVariableOp_15^sequential_36/batch_normalization_355/ReadVariableOp7^sequential_36/batch_normalization_355/ReadVariableOp_10^sequential_36/conv2d_293/BiasAdd/ReadVariableOp/^sequential_36/conv2d_293/Conv2D/ReadVariableOp0^sequential_36/conv2d_294/BiasAdd/ReadVariableOp/^sequential_36/conv2d_294/Conv2D/ReadVariableOp0^sequential_36/conv2d_295/BiasAdd/ReadVariableOp/^sequential_36/conv2d_295/Conv2D/ReadVariableOp0^sequential_36/conv2d_296/BiasAdd/ReadVariableOp/^sequential_36/conv2d_296/Conv2D/ReadVariableOp:^sequential_36/conv2d_transpose_241/BiasAdd/ReadVariableOpC^sequential_36/conv2d_transpose_241/conv2d_transpose/ReadVariableOp:^sequential_36/conv2d_transpose_242/BiasAdd/ReadVariableOpC^sequential_36/conv2d_transpose_242/conv2d_transpose/ReadVariableOp:^sequential_36/conv2d_transpose_243/BiasAdd/ReadVariableOpC^sequential_36/conv2d_transpose_243/conv2d_transpose/ReadVariableOp-^sequential_36/decoded/BiasAdd/ReadVariableOp6^sequential_36/decoded/conv2d_transpose/ReadVariableOp6^sequential_36/encoded/FusedBatchNormV3/ReadVariableOp8^sequential_36/encoded/FusedBatchNormV3/ReadVariableOp_1%^sequential_36/encoded/ReadVariableOp'^sequential_36/encoded/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2?
Esequential_36/batch_normalization_353/FusedBatchNormV3/ReadVariableOpEsequential_36/batch_normalization_353/FusedBatchNormV3/ReadVariableOp2?
Gsequential_36/batch_normalization_353/FusedBatchNormV3/ReadVariableOp_1Gsequential_36/batch_normalization_353/FusedBatchNormV3/ReadVariableOp_12l
4sequential_36/batch_normalization_353/ReadVariableOp4sequential_36/batch_normalization_353/ReadVariableOp2p
6sequential_36/batch_normalization_353/ReadVariableOp_16sequential_36/batch_normalization_353/ReadVariableOp_12?
Esequential_36/batch_normalization_354/FusedBatchNormV3/ReadVariableOpEsequential_36/batch_normalization_354/FusedBatchNormV3/ReadVariableOp2?
Gsequential_36/batch_normalization_354/FusedBatchNormV3/ReadVariableOp_1Gsequential_36/batch_normalization_354/FusedBatchNormV3/ReadVariableOp_12l
4sequential_36/batch_normalization_354/ReadVariableOp4sequential_36/batch_normalization_354/ReadVariableOp2p
6sequential_36/batch_normalization_354/ReadVariableOp_16sequential_36/batch_normalization_354/ReadVariableOp_12?
Esequential_36/batch_normalization_355/FusedBatchNormV3/ReadVariableOpEsequential_36/batch_normalization_355/FusedBatchNormV3/ReadVariableOp2?
Gsequential_36/batch_normalization_355/FusedBatchNormV3/ReadVariableOp_1Gsequential_36/batch_normalization_355/FusedBatchNormV3/ReadVariableOp_12l
4sequential_36/batch_normalization_355/ReadVariableOp4sequential_36/batch_normalization_355/ReadVariableOp2p
6sequential_36/batch_normalization_355/ReadVariableOp_16sequential_36/batch_normalization_355/ReadVariableOp_12b
/sequential_36/conv2d_293/BiasAdd/ReadVariableOp/sequential_36/conv2d_293/BiasAdd/ReadVariableOp2`
.sequential_36/conv2d_293/Conv2D/ReadVariableOp.sequential_36/conv2d_293/Conv2D/ReadVariableOp2b
/sequential_36/conv2d_294/BiasAdd/ReadVariableOp/sequential_36/conv2d_294/BiasAdd/ReadVariableOp2`
.sequential_36/conv2d_294/Conv2D/ReadVariableOp.sequential_36/conv2d_294/Conv2D/ReadVariableOp2b
/sequential_36/conv2d_295/BiasAdd/ReadVariableOp/sequential_36/conv2d_295/BiasAdd/ReadVariableOp2`
.sequential_36/conv2d_295/Conv2D/ReadVariableOp.sequential_36/conv2d_295/Conv2D/ReadVariableOp2b
/sequential_36/conv2d_296/BiasAdd/ReadVariableOp/sequential_36/conv2d_296/BiasAdd/ReadVariableOp2`
.sequential_36/conv2d_296/Conv2D/ReadVariableOp.sequential_36/conv2d_296/Conv2D/ReadVariableOp2v
9sequential_36/conv2d_transpose_241/BiasAdd/ReadVariableOp9sequential_36/conv2d_transpose_241/BiasAdd/ReadVariableOp2?
Bsequential_36/conv2d_transpose_241/conv2d_transpose/ReadVariableOpBsequential_36/conv2d_transpose_241/conv2d_transpose/ReadVariableOp2v
9sequential_36/conv2d_transpose_242/BiasAdd/ReadVariableOp9sequential_36/conv2d_transpose_242/BiasAdd/ReadVariableOp2?
Bsequential_36/conv2d_transpose_242/conv2d_transpose/ReadVariableOpBsequential_36/conv2d_transpose_242/conv2d_transpose/ReadVariableOp2v
9sequential_36/conv2d_transpose_243/BiasAdd/ReadVariableOp9sequential_36/conv2d_transpose_243/BiasAdd/ReadVariableOp2?
Bsequential_36/conv2d_transpose_243/conv2d_transpose/ReadVariableOpBsequential_36/conv2d_transpose_243/conv2d_transpose/ReadVariableOp2\
,sequential_36/decoded/BiasAdd/ReadVariableOp,sequential_36/decoded/BiasAdd/ReadVariableOp2n
5sequential_36/decoded/conv2d_transpose/ReadVariableOp5sequential_36/decoded/conv2d_transpose/ReadVariableOp2n
5sequential_36/encoded/FusedBatchNormV3/ReadVariableOp5sequential_36/encoded/FusedBatchNormV3/ReadVariableOp2r
7sequential_36/encoded/FusedBatchNormV3/ReadVariableOp_17sequential_36/encoded/FusedBatchNormV3/ReadVariableOp_12L
$sequential_36/encoded/ReadVariableOp$sequential_36/encoded/ReadVariableOp2P
&sequential_36/encoded/ReadVariableOp_1&sequential_36/encoded/ReadVariableOp_1:c _
1
_output_shapes
:???????????
*
_user_specified_nameconv2d_293_input
?#
?
P__inference_conv2d_transpose_241_layer_call_and_return_conditional_losses_313653

inputsB
(conv2d_transpose_readvariableop_resource:@ -
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
+__inference_conv2d_295_layer_call_fn_315406

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_295_layer_call_and_return_conditional_losses_313883w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????22 : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????22 
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_243_layer_call_fn_315705

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_243_layer_call_and_return_conditional_losses_313743?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?!
?
P__inference_conv2d_transpose_243_layer_call_and_return_conditional_losses_313743

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_354_layer_call_and_return_conditional_losses_315397

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
d
F__inference_dropout_95_layer_call_and_return_conditional_losses_313910

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?^
?
I__inference_sequential_36_layer_call_and_return_conditional_losses_314527
conv2d_293_input+
conv2d_293_314442:
conv2d_293_314444:,
batch_normalization_353_314448:,
batch_normalization_353_314450:,
batch_normalization_353_314452:,
batch_normalization_353_314454:+
conv2d_294_314458: 
conv2d_294_314460: ,
batch_normalization_354_314464: ,
batch_normalization_354_314466: ,
batch_normalization_354_314468: ,
batch_normalization_354_314470: +
conv2d_295_314473: @
conv2d_295_314475:@,
batch_normalization_355_314479:@,
batch_normalization_355_314481:@,
batch_normalization_355_314483:@,
batch_normalization_355_314485:@+
conv2d_296_314489:@ 
conv2d_296_314491: 
encoded_314496: 
encoded_314498: 
encoded_314500: 
encoded_314502: 5
conv2d_transpose_241_314506:@ )
conv2d_transpose_241_314508:@5
conv2d_transpose_242_314511: @)
conv2d_transpose_242_314513: 5
conv2d_transpose_243_314516: )
conv2d_transpose_243_314518:(
decoded_314521:
decoded_314523:
identity??/batch_normalization_353/StatefulPartitionedCall?/batch_normalization_354/StatefulPartitionedCall?/batch_normalization_355/StatefulPartitionedCall?"conv2d_293/StatefulPartitionedCall?"conv2d_294/StatefulPartitionedCall?"conv2d_295/StatefulPartitionedCall?"conv2d_296/StatefulPartitionedCall?,conv2d_transpose_241/StatefulPartitionedCall?,conv2d_transpose_242/StatefulPartitionedCall?,conv2d_transpose_243/StatefulPartitionedCall?decoded/StatefulPartitionedCall?encoded/StatefulPartitionedCall?
"conv2d_293/StatefulPartitionedCallStatefulPartitionedCallconv2d_293_inputconv2d_293_314442conv2d_293_314444*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_293_layer_call_and_return_conditional_losses_313812?
activation_344/PartitionedCallPartitionedCall+conv2d_293/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_344_layer_call_and_return_conditional_losses_313823?
/batch_normalization_353/StatefulPartitionedCallStatefulPartitionedCall'activation_344/PartitionedCall:output:0batch_normalization_353_314448batch_normalization_353_314450batch_normalization_353_314452batch_normalization_353_314454*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_353_layer_call_and_return_conditional_losses_313377?
dropout_94/PartitionedCallPartitionedCall8batch_normalization_353/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dropout_94_layer_call_and_return_conditional_losses_313839?
"conv2d_294/StatefulPartitionedCallStatefulPartitionedCall#dropout_94/PartitionedCall:output:0conv2d_294_314458conv2d_294_314460*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_294_layer_call_and_return_conditional_losses_313851?
activation_345/PartitionedCallPartitionedCall+conv2d_294/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_345_layer_call_and_return_conditional_losses_313862?
/batch_normalization_354/StatefulPartitionedCallStatefulPartitionedCall'activation_345/PartitionedCall:output:0batch_normalization_354_314464batch_normalization_354_314466batch_normalization_354_314468batch_normalization_354_314470*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_354_layer_call_and_return_conditional_losses_313441?
"conv2d_295/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_354/StatefulPartitionedCall:output:0conv2d_295_314473conv2d_295_314475*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_295_layer_call_and_return_conditional_losses_313883?
activation_346/PartitionedCallPartitionedCall+conv2d_295/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_346_layer_call_and_return_conditional_losses_313894?
/batch_normalization_355/StatefulPartitionedCallStatefulPartitionedCall'activation_346/PartitionedCall:output:0batch_normalization_355_314479batch_normalization_355_314481batch_normalization_355_314483batch_normalization_355_314485*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_355_layer_call_and_return_conditional_losses_313505?
dropout_95/PartitionedCallPartitionedCall8batch_normalization_355/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dropout_95_layer_call_and_return_conditional_losses_313910?
"conv2d_296/StatefulPartitionedCallStatefulPartitionedCall#dropout_95/PartitionedCall:output:0conv2d_296_314489conv2d_296_314491*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_296_layer_call_and_return_conditional_losses_313922?
activation_347/PartitionedCallPartitionedCall+conv2d_296/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_347_layer_call_and_return_conditional_losses_313933?
encoded/CastCast'activation_347/PartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
encoded/StatefulPartitionedCallStatefulPartitionedCallencoded/Cast:y:0encoded_314496encoded_314498encoded_314500encoded_314502*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_encoded_layer_call_and_return_conditional_losses_313569?
conv2d_transpose_241/CastCast(encoded/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
,conv2d_transpose_241/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_241/Cast:y:0conv2d_transpose_241_314506conv2d_transpose_241_314508*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_241_layer_call_and_return_conditional_losses_313653?
,conv2d_transpose_242/StatefulPartitionedCallStatefulPartitionedCall5conv2d_transpose_241/StatefulPartitionedCall:output:0conv2d_transpose_242_314511conv2d_transpose_242_314513*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????22 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_242_layer_call_and_return_conditional_losses_313698?
,conv2d_transpose_243/StatefulPartitionedCallStatefulPartitionedCall5conv2d_transpose_242/StatefulPartitionedCall:output:0conv2d_transpose_243_314516conv2d_transpose_243_314518*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_243_layer_call_and_return_conditional_losses_313743?
decoded/StatefulPartitionedCallStatefulPartitionedCall5conv2d_transpose_243/StatefulPartitionedCall:output:0decoded_314521decoded_314523*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_decoded_layer_call_and_return_conditional_losses_313788?
IdentityIdentity(decoded/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp0^batch_normalization_353/StatefulPartitionedCall0^batch_normalization_354/StatefulPartitionedCall0^batch_normalization_355/StatefulPartitionedCall#^conv2d_293/StatefulPartitionedCall#^conv2d_294/StatefulPartitionedCall#^conv2d_295/StatefulPartitionedCall#^conv2d_296/StatefulPartitionedCall-^conv2d_transpose_241/StatefulPartitionedCall-^conv2d_transpose_242/StatefulPartitionedCall-^conv2d_transpose_243/StatefulPartitionedCall ^decoded/StatefulPartitionedCall ^encoded/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_353/StatefulPartitionedCall/batch_normalization_353/StatefulPartitionedCall2b
/batch_normalization_354/StatefulPartitionedCall/batch_normalization_354/StatefulPartitionedCall2b
/batch_normalization_355/StatefulPartitionedCall/batch_normalization_355/StatefulPartitionedCall2H
"conv2d_293/StatefulPartitionedCall"conv2d_293/StatefulPartitionedCall2H
"conv2d_294/StatefulPartitionedCall"conv2d_294/StatefulPartitionedCall2H
"conv2d_295/StatefulPartitionedCall"conv2d_295/StatefulPartitionedCall2H
"conv2d_296/StatefulPartitionedCall"conv2d_296/StatefulPartitionedCall2\
,conv2d_transpose_241/StatefulPartitionedCall,conv2d_transpose_241/StatefulPartitionedCall2\
,conv2d_transpose_242/StatefulPartitionedCall,conv2d_transpose_242/StatefulPartitionedCall2\
,conv2d_transpose_243/StatefulPartitionedCall,conv2d_transpose_243/StatefulPartitionedCall2B
decoded/StatefulPartitionedCalldecoded/StatefulPartitionedCall2B
encoded/StatefulPartitionedCallencoded/StatefulPartitionedCall:c _
1
_output_shapes
:???????????
*
_user_specified_nameconv2d_293_input
?
?
.__inference_sequential_36_layer_call_fn_314034
conv2d_293_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@ 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23:@ 

unknown_24:@$

unknown_25: @

unknown_26: $

unknown_27: 

unknown_28:$

unknown_29:

unknown_30:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_293_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_sequential_36_layer_call_and_return_conditional_losses_313967y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
1
_output_shapes
:???????????
*
_user_specified_nameconv2d_293_input
?
f
J__inference_activation_345_layer_call_and_return_conditional_losses_313862

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????22 b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????22 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????22 :W S
/
_output_shapes
:?????????22 
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_355_layer_call_fn_315439

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_355_layer_call_and_return_conditional_losses_313505?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
+__inference_conv2d_296_layer_call_fn_315524

inputs!
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2d_296_layer_call_and_return_conditional_losses_313922w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_353_layer_call_and_return_conditional_losses_313408

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_354_layer_call_and_return_conditional_losses_313472

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
d
F__inference_dropout_94_layer_call_and_return_conditional_losses_315294

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????ddc

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????dd"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????dd:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
??
?:
"__inference__traced_restore_316337
file_prefix<
"assignvariableop_conv2d_293_kernel:0
"assignvariableop_1_conv2d_293_bias:>
0assignvariableop_2_batch_normalization_353_gamma:=
/assignvariableop_3_batch_normalization_353_beta:D
6assignvariableop_4_batch_normalization_353_moving_mean:H
:assignvariableop_5_batch_normalization_353_moving_variance:>
$assignvariableop_6_conv2d_294_kernel: 0
"assignvariableop_7_conv2d_294_bias: >
0assignvariableop_8_batch_normalization_354_gamma: =
/assignvariableop_9_batch_normalization_354_beta: E
7assignvariableop_10_batch_normalization_354_moving_mean: I
;assignvariableop_11_batch_normalization_354_moving_variance: ?
%assignvariableop_12_conv2d_295_kernel: @1
#assignvariableop_13_conv2d_295_bias:@?
1assignvariableop_14_batch_normalization_355_gamma:@>
0assignvariableop_15_batch_normalization_355_beta:@E
7assignvariableop_16_batch_normalization_355_moving_mean:@I
;assignvariableop_17_batch_normalization_355_moving_variance:@?
%assignvariableop_18_conv2d_296_kernel:@ 1
#assignvariableop_19_conv2d_296_bias: /
!assignvariableop_20_encoded_gamma: .
 assignvariableop_21_encoded_beta: 5
'assignvariableop_22_encoded_moving_mean: 9
+assignvariableop_23_encoded_moving_variance: I
/assignvariableop_24_conv2d_transpose_241_kernel:@ ;
-assignvariableop_25_conv2d_transpose_241_bias:@I
/assignvariableop_26_conv2d_transpose_242_kernel: @;
-assignvariableop_27_conv2d_transpose_242_bias: I
/assignvariableop_28_conv2d_transpose_243_kernel: ;
-assignvariableop_29_conv2d_transpose_243_bias:<
"assignvariableop_30_decoded_kernel:.
 assignvariableop_31_decoded_bias:'
assignvariableop_32_adam_iter:	 )
assignvariableop_33_adam_beta_1: )
assignvariableop_34_adam_beta_2: (
assignvariableop_35_adam_decay: 0
&assignvariableop_36_adam_learning_rate: #
assignvariableop_37_total: #
assignvariableop_38_count: F
,assignvariableop_39_adam_conv2d_293_kernel_m:8
*assignvariableop_40_adam_conv2d_293_bias_m:F
8assignvariableop_41_adam_batch_normalization_353_gamma_m:E
7assignvariableop_42_adam_batch_normalization_353_beta_m:F
,assignvariableop_43_adam_conv2d_294_kernel_m: 8
*assignvariableop_44_adam_conv2d_294_bias_m: F
8assignvariableop_45_adam_batch_normalization_354_gamma_m: E
7assignvariableop_46_adam_batch_normalization_354_beta_m: F
,assignvariableop_47_adam_conv2d_295_kernel_m: @8
*assignvariableop_48_adam_conv2d_295_bias_m:@F
8assignvariableop_49_adam_batch_normalization_355_gamma_m:@E
7assignvariableop_50_adam_batch_normalization_355_beta_m:@F
,assignvariableop_51_adam_conv2d_296_kernel_m:@ 8
*assignvariableop_52_adam_conv2d_296_bias_m: 6
(assignvariableop_53_adam_encoded_gamma_m: 5
'assignvariableop_54_adam_encoded_beta_m: P
6assignvariableop_55_adam_conv2d_transpose_241_kernel_m:@ B
4assignvariableop_56_adam_conv2d_transpose_241_bias_m:@P
6assignvariableop_57_adam_conv2d_transpose_242_kernel_m: @B
4assignvariableop_58_adam_conv2d_transpose_242_bias_m: P
6assignvariableop_59_adam_conv2d_transpose_243_kernel_m: B
4assignvariableop_60_adam_conv2d_transpose_243_bias_m:C
)assignvariableop_61_adam_decoded_kernel_m:5
'assignvariableop_62_adam_decoded_bias_m:F
,assignvariableop_63_adam_conv2d_293_kernel_v:8
*assignvariableop_64_adam_conv2d_293_bias_v:F
8assignvariableop_65_adam_batch_normalization_353_gamma_v:E
7assignvariableop_66_adam_batch_normalization_353_beta_v:F
,assignvariableop_67_adam_conv2d_294_kernel_v: 8
*assignvariableop_68_adam_conv2d_294_bias_v: F
8assignvariableop_69_adam_batch_normalization_354_gamma_v: E
7assignvariableop_70_adam_batch_normalization_354_beta_v: F
,assignvariableop_71_adam_conv2d_295_kernel_v: @8
*assignvariableop_72_adam_conv2d_295_bias_v:@F
8assignvariableop_73_adam_batch_normalization_355_gamma_v:@E
7assignvariableop_74_adam_batch_normalization_355_beta_v:@F
,assignvariableop_75_adam_conv2d_296_kernel_v:@ 8
*assignvariableop_76_adam_conv2d_296_bias_v: 6
(assignvariableop_77_adam_encoded_gamma_v: 5
'assignvariableop_78_adam_encoded_beta_v: P
6assignvariableop_79_adam_conv2d_transpose_241_kernel_v:@ B
4assignvariableop_80_adam_conv2d_transpose_241_bias_v:@P
6assignvariableop_81_adam_conv2d_transpose_242_kernel_v: @B
4assignvariableop_82_adam_conv2d_transpose_242_bias_v: P
6assignvariableop_83_adam_conv2d_transpose_243_kernel_v: B
4assignvariableop_84_adam_conv2d_transpose_243_bias_v:C
)assignvariableop_85_adam_decoded_kernel_v:5
'assignvariableop_86_adam_decoded_bias_v:
identity_88??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_9?1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*?0
value?0B?0XB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*?
value?B?XB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*f
dtypes\
Z2X	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_293_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_293_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp0assignvariableop_2_batch_normalization_353_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp/assignvariableop_3_batch_normalization_353_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp6assignvariableop_4_batch_normalization_353_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp:assignvariableop_5_batch_normalization_353_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp$assignvariableop_6_conv2d_294_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv2d_294_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp0assignvariableop_8_batch_normalization_354_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp/assignvariableop_9_batch_normalization_354_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp7assignvariableop_10_batch_normalization_354_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp;assignvariableop_11_batch_normalization_354_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp%assignvariableop_12_conv2d_295_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_295_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp1assignvariableop_14_batch_normalization_355_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp0assignvariableop_15_batch_normalization_355_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp7assignvariableop_16_batch_normalization_355_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp;assignvariableop_17_batch_normalization_355_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp%assignvariableop_18_conv2d_296_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp#assignvariableop_19_conv2d_296_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp!assignvariableop_20_encoded_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp assignvariableop_21_encoded_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_encoded_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_encoded_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp/assignvariableop_24_conv2d_transpose_241_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp-assignvariableop_25_conv2d_transpose_241_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp/assignvariableop_26_conv2d_transpose_242_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp-assignvariableop_27_conv2d_transpose_242_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp/assignvariableop_28_conv2d_transpose_243_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp-assignvariableop_29_conv2d_transpose_243_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp"assignvariableop_30_decoded_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp assignvariableop_31_decoded_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_iterIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_beta_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOpassignvariableop_34_adam_beta_2Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOpassignvariableop_35_adam_decayIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_learning_rateIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOpassignvariableop_37_totalIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOpassignvariableop_38_countIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp,assignvariableop_39_adam_conv2d_293_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_conv2d_293_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp8assignvariableop_41_adam_batch_normalization_353_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp7assignvariableop_42_adam_batch_normalization_353_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp,assignvariableop_43_adam_conv2d_294_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_conv2d_294_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp8assignvariableop_45_adam_batch_normalization_354_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp7assignvariableop_46_adam_batch_normalization_354_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp,assignvariableop_47_adam_conv2d_295_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_conv2d_295_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp8assignvariableop_49_adam_batch_normalization_355_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp7assignvariableop_50_adam_batch_normalization_355_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp,assignvariableop_51_adam_conv2d_296_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adam_conv2d_296_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_encoded_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_encoded_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp6assignvariableop_55_adam_conv2d_transpose_241_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp4assignvariableop_56_adam_conv2d_transpose_241_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_conv2d_transpose_242_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp4assignvariableop_58_adam_conv2d_transpose_242_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp6assignvariableop_59_adam_conv2d_transpose_243_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp4assignvariableop_60_adam_conv2d_transpose_243_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp)assignvariableop_61_adam_decoded_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp'assignvariableop_62_adam_decoded_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp,assignvariableop_63_adam_conv2d_293_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_conv2d_293_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp8assignvariableop_65_adam_batch_normalization_353_gamma_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp7assignvariableop_66_adam_batch_normalization_353_beta_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp,assignvariableop_67_adam_conv2d_294_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp*assignvariableop_68_adam_conv2d_294_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp8assignvariableop_69_adam_batch_normalization_354_gamma_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp7assignvariableop_70_adam_batch_normalization_354_beta_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp,assignvariableop_71_adam_conv2d_295_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_conv2d_295_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp8assignvariableop_73_adam_batch_normalization_355_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp7assignvariableop_74_adam_batch_normalization_355_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp,assignvariableop_75_adam_conv2d_296_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOp*assignvariableop_76_adam_conv2d_296_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOp(assignvariableop_77_adam_encoded_gamma_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp'assignvariableop_78_adam_encoded_beta_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp6assignvariableop_79_adam_conv2d_transpose_241_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOp4assignvariableop_80_adam_conv2d_transpose_241_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOp6assignvariableop_81_adam_conv2d_transpose_242_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOp4assignvariableop_82_adam_conv2d_transpose_242_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOp6assignvariableop_83_adam_conv2d_transpose_243_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOp4assignvariableop_84_adam_conv2d_transpose_243_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOp)assignvariableop_85_adam_decoded_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOp'assignvariableop_86_adam_decoded_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_87Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_88IdentityIdentity_87:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_88Identity_88:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
K
/__inference_activation_344_layer_call_fn_315212

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????dd* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_activation_344_layer_call_and_return_conditional_losses_313823h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????dd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????dd:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_355_layer_call_and_return_conditional_losses_313505

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?#
?
P__inference_conv2d_transpose_241_layer_call_and_return_conditional_losses_315653

inputsB
(conv2d_transpose_readvariableop_resource:@ -
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
(__inference_encoded_layer_call_fn_315570

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_encoded_layer_call_and_return_conditional_losses_313600?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_354_layer_call_and_return_conditional_losses_313441

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs"?	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
W
conv2d_293_inputC
"serving_default_conv2d_293_input:0???????????E
decoded:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
?
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias
 $_jit_compiled_convolution_op"
_tf_keras_layer
?
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer
?
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
1axis
	2gamma
3beta
4moving_mean
5moving_variance"
_tf_keras_layer
?
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses
<_random_generator"
_tf_keras_layer
?
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

Ckernel
Dbias
 E_jit_compiled_convolution_op"
_tf_keras_layer
?
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_layer
?
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
Raxis
	Sgamma
Tbeta
Umoving_mean
Vmoving_variance"
_tf_keras_layer
?
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

]kernel
^bias
 __jit_compiled_convolution_op"
_tf_keras_layer
?
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses"
_tf_keras_layer
?
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses
laxis
	mgamma
nbeta
omoving_mean
pmoving_variance"
_tf_keras_layer
?
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses
w_random_generator"
_tf_keras_layer
?
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

~kernel
bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
"0
#1
22
33
44
55
C6
D7
S8
T9
U10
V11
]12
^13
m14
n15
o16
p17
~18
19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31"
trackable_list_wrapper
?
"0
#1
22
33
C4
D5
S6
T7
]8
^9
m10
n11
~12
13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
.__inference_sequential_36_layer_call_fn_314034
.__inference_sequential_36_layer_call_fn_314761
.__inference_sequential_36_layer_call_fn_314830
.__inference_sequential_36_layer_call_fn_314439?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
I__inference_sequential_36_layer_call_and_return_conditional_losses_315002
I__inference_sequential_36_layer_call_and_return_conditional_losses_315188
I__inference_sequential_36_layer_call_and_return_conditional_losses_314527
I__inference_sequential_36_layer_call_and_return_conditional_losses_314615?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?B?
!__inference__wrapped_model_313355conv2d_293_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate"m?#m?2m?3m?Cm?Dm?Sm?Tm?]m?^m?mm?nm?~m?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?"v?#v?2v?3v?Cv?Dv?Sv?Tv?]v?^v?mv?nv?~v?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
-
?serving_default"
signature_map
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_conv2d_293_layer_call_fn_315197?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_conv2d_293_layer_call_and_return_conditional_losses_315207?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
+:)2conv2d_293/kernel
:2conv2d_293/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_activation_344_layer_call_fn_315212?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
J__inference_activation_344_layer_call_and_return_conditional_losses_315217?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
<
20
31
42
53"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
8__inference_batch_normalization_353_layer_call_fn_315230
8__inference_batch_normalization_353_layer_call_fn_315243?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
S__inference_batch_normalization_353_layer_call_and_return_conditional_losses_315261
S__inference_batch_normalization_353_layer_call_and_return_conditional_losses_315279?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
+:)2batch_normalization_353/gamma
*:(2batch_normalization_353/beta
3:1 (2#batch_normalization_353/moving_mean
7:5 (2'batch_normalization_353/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
+__inference_dropout_94_layer_call_fn_315284
+__inference_dropout_94_layer_call_fn_315289?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
F__inference_dropout_94_layer_call_and_return_conditional_losses_315294
F__inference_dropout_94_layer_call_and_return_conditional_losses_315306?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_conv2d_294_layer_call_fn_315315?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_conv2d_294_layer_call_and_return_conditional_losses_315325?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
+:) 2conv2d_294/kernel
: 2conv2d_294/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_activation_345_layer_call_fn_315330?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
J__inference_activation_345_layer_call_and_return_conditional_losses_315335?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
<
S0
T1
U2
V3"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
8__inference_batch_normalization_354_layer_call_fn_315348
8__inference_batch_normalization_354_layer_call_fn_315361?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
S__inference_batch_normalization_354_layer_call_and_return_conditional_losses_315379
S__inference_batch_normalization_354_layer_call_and_return_conditional_losses_315397?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
+:) 2batch_normalization_354/gamma
*:( 2batch_normalization_354/beta
3:1  (2#batch_normalization_354/moving_mean
7:5  (2'batch_normalization_354/moving_variance
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_conv2d_295_layer_call_fn_315406?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_conv2d_295_layer_call_and_return_conditional_losses_315416?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
+:) @2conv2d_295/kernel
:@2conv2d_295/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_activation_346_layer_call_fn_315421?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
J__inference_activation_346_layer_call_and_return_conditional_losses_315426?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
<
m0
n1
o2
p3"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
8__inference_batch_normalization_355_layer_call_fn_315439
8__inference_batch_normalization_355_layer_call_fn_315452?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
S__inference_batch_normalization_355_layer_call_and_return_conditional_losses_315470
S__inference_batch_normalization_355_layer_call_and_return_conditional_losses_315488?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
+:)@2batch_normalization_355/gamma
*:(@2batch_normalization_355/beta
3:1@ (2#batch_normalization_355/moving_mean
7:5@ (2'batch_normalization_355/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
+__inference_dropout_95_layer_call_fn_315493
+__inference_dropout_95_layer_call_fn_315498?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
F__inference_dropout_95_layer_call_and_return_conditional_losses_315503
F__inference_dropout_95_layer_call_and_return_conditional_losses_315515?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_conv2d_296_layer_call_fn_315524?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_conv2d_296_layer_call_and_return_conditional_losses_315534?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
+:)@ 2conv2d_296/kernel
: 2conv2d_296/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_activation_347_layer_call_fn_315539?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
J__inference_activation_347_layer_call_and_return_conditional_losses_315544?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
(__inference_encoded_layer_call_fn_315557
(__inference_encoded_layer_call_fn_315570?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
C__inference_encoded_layer_call_and_return_conditional_losses_315588
C__inference_encoded_layer_call_and_return_conditional_losses_315606?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
: 2encoded/gamma
: 2encoded/beta
#:!  (2encoded/moving_mean
':%  (2encoded/moving_variance
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
5__inference_conv2d_transpose_241_layer_call_fn_315615?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
P__inference_conv2d_transpose_241_layer_call_and_return_conditional_losses_315653?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
5:3@ 2conv2d_transpose_241/kernel
':%@2conv2d_transpose_241/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
5__inference_conv2d_transpose_242_layer_call_fn_315662?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
P__inference_conv2d_transpose_242_layer_call_and_return_conditional_losses_315696?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
5:3 @2conv2d_transpose_242/kernel
':% 2conv2d_transpose_242/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
5__inference_conv2d_transpose_243_layer_call_fn_315705?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
P__inference_conv2d_transpose_243_layer_call_and_return_conditional_losses_315739?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
5:3 2conv2d_transpose_243/kernel
':%2conv2d_transpose_243/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_decoded_layer_call_fn_315748?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_decoded_layer_call_and_return_conditional_losses_315782?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
(:&2decoded/kernel
:2decoded/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
Z
40
51
U2
V3
o4
p5
?6
?7"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
.__inference_sequential_36_layer_call_fn_314034conv2d_293_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
.__inference_sequential_36_layer_call_fn_314761inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
.__inference_sequential_36_layer_call_fn_314830inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
.__inference_sequential_36_layer_call_fn_314439conv2d_293_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
I__inference_sequential_36_layer_call_and_return_conditional_losses_315002inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
I__inference_sequential_36_layer_call_and_return_conditional_losses_315188inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
I__inference_sequential_36_layer_call_and_return_conditional_losses_314527conv2d_293_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
I__inference_sequential_36_layer_call_and_return_conditional_losses_314615conv2d_293_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
$__inference_signature_wrapper_314692conv2d_293_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_conv2d_293_layer_call_fn_315197inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_conv2d_293_layer_call_and_return_conditional_losses_315207inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
/__inference_activation_344_layer_call_fn_315212inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_activation_344_layer_call_and_return_conditional_losses_315217inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
8__inference_batch_normalization_353_layer_call_fn_315230inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
8__inference_batch_normalization_353_layer_call_fn_315243inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
S__inference_batch_normalization_353_layer_call_and_return_conditional_losses_315261inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
S__inference_batch_normalization_353_layer_call_and_return_conditional_losses_315279inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_dropout_94_layer_call_fn_315284inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
+__inference_dropout_94_layer_call_fn_315289inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_dropout_94_layer_call_and_return_conditional_losses_315294inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_dropout_94_layer_call_and_return_conditional_losses_315306inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_conv2d_294_layer_call_fn_315315inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_conv2d_294_layer_call_and_return_conditional_losses_315325inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
/__inference_activation_345_layer_call_fn_315330inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_activation_345_layer_call_and_return_conditional_losses_315335inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
8__inference_batch_normalization_354_layer_call_fn_315348inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
8__inference_batch_normalization_354_layer_call_fn_315361inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
S__inference_batch_normalization_354_layer_call_and_return_conditional_losses_315379inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
S__inference_batch_normalization_354_layer_call_and_return_conditional_losses_315397inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_conv2d_295_layer_call_fn_315406inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_conv2d_295_layer_call_and_return_conditional_losses_315416inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
/__inference_activation_346_layer_call_fn_315421inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_activation_346_layer_call_and_return_conditional_losses_315426inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
8__inference_batch_normalization_355_layer_call_fn_315439inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
8__inference_batch_normalization_355_layer_call_fn_315452inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
S__inference_batch_normalization_355_layer_call_and_return_conditional_losses_315470inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
S__inference_batch_normalization_355_layer_call_and_return_conditional_losses_315488inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_dropout_95_layer_call_fn_315493inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
+__inference_dropout_95_layer_call_fn_315498inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_dropout_95_layer_call_and_return_conditional_losses_315503inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_dropout_95_layer_call_and_return_conditional_losses_315515inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_conv2d_296_layer_call_fn_315524inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_conv2d_296_layer_call_and_return_conditional_losses_315534inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
/__inference_activation_347_layer_call_fn_315539inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_activation_347_layer_call_and_return_conditional_losses_315544inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_encoded_layer_call_fn_315557inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
(__inference_encoded_layer_call_fn_315570inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_encoded_layer_call_and_return_conditional_losses_315588inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_encoded_layer_call_and_return_conditional_losses_315606inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
5__inference_conv2d_transpose_241_layer_call_fn_315615inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_conv2d_transpose_241_layer_call_and_return_conditional_losses_315653inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
5__inference_conv2d_transpose_242_layer_call_fn_315662inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_conv2d_transpose_242_layer_call_and_return_conditional_losses_315696inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
5__inference_conv2d_transpose_243_layer_call_fn_315705inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_conv2d_transpose_243_layer_call_and_return_conditional_losses_315739inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_decoded_layer_call_fn_315748inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_decoded_layer_call_and_return_conditional_losses_315782inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0:.2Adam/conv2d_293/kernel/m
": 2Adam/conv2d_293/bias/m
0:.2$Adam/batch_normalization_353/gamma/m
/:-2#Adam/batch_normalization_353/beta/m
0:. 2Adam/conv2d_294/kernel/m
":  2Adam/conv2d_294/bias/m
0:. 2$Adam/batch_normalization_354/gamma/m
/:- 2#Adam/batch_normalization_354/beta/m
0:. @2Adam/conv2d_295/kernel/m
": @2Adam/conv2d_295/bias/m
0:.@2$Adam/batch_normalization_355/gamma/m
/:-@2#Adam/batch_normalization_355/beta/m
0:.@ 2Adam/conv2d_296/kernel/m
":  2Adam/conv2d_296/bias/m
 : 2Adam/encoded/gamma/m
: 2Adam/encoded/beta/m
::8@ 2"Adam/conv2d_transpose_241/kernel/m
,:*@2 Adam/conv2d_transpose_241/bias/m
::8 @2"Adam/conv2d_transpose_242/kernel/m
,:* 2 Adam/conv2d_transpose_242/bias/m
::8 2"Adam/conv2d_transpose_243/kernel/m
,:*2 Adam/conv2d_transpose_243/bias/m
-:+2Adam/decoded/kernel/m
:2Adam/decoded/bias/m
0:.2Adam/conv2d_293/kernel/v
": 2Adam/conv2d_293/bias/v
0:.2$Adam/batch_normalization_353/gamma/v
/:-2#Adam/batch_normalization_353/beta/v
0:. 2Adam/conv2d_294/kernel/v
":  2Adam/conv2d_294/bias/v
0:. 2$Adam/batch_normalization_354/gamma/v
/:- 2#Adam/batch_normalization_354/beta/v
0:. @2Adam/conv2d_295/kernel/v
": @2Adam/conv2d_295/bias/v
0:.@2$Adam/batch_normalization_355/gamma/v
/:-@2#Adam/batch_normalization_355/beta/v
0:.@ 2Adam/conv2d_296/kernel/v
":  2Adam/conv2d_296/bias/v
 : 2Adam/encoded/gamma/v
: 2Adam/encoded/beta/v
::8@ 2"Adam/conv2d_transpose_241/kernel/v
,:*@2 Adam/conv2d_transpose_241/bias/v
::8 @2"Adam/conv2d_transpose_242/kernel/v
,:* 2 Adam/conv2d_transpose_242/bias/v
::8 2"Adam/conv2d_transpose_243/kernel/v
,:*2 Adam/conv2d_transpose_243/bias/v
-:+2Adam/decoded/kernel/v
:2Adam/decoded/bias/v?
!__inference__wrapped_model_313355?,"#2345CDSTUV]^mnop~????????????C?@
9?6
4?1
conv2d_293_input???????????
? ";?8
6
decoded+?(
decoded????????????
J__inference_activation_344_layer_call_and_return_conditional_losses_315217h7?4
-?*
(?%
inputs?????????dd
? "-?*
#? 
0?????????dd
? ?
/__inference_activation_344_layer_call_fn_315212[7?4
-?*
(?%
inputs?????????dd
? " ??????????dd?
J__inference_activation_345_layer_call_and_return_conditional_losses_315335h7?4
-?*
(?%
inputs?????????22 
? "-?*
#? 
0?????????22 
? ?
/__inference_activation_345_layer_call_fn_315330[7?4
-?*
(?%
inputs?????????22 
? " ??????????22 ?
J__inference_activation_346_layer_call_and_return_conditional_losses_315426h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
/__inference_activation_346_layer_call_fn_315421[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
J__inference_activation_347_layer_call_and_return_conditional_losses_315544h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
/__inference_activation_347_layer_call_fn_315539[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
S__inference_batch_normalization_353_layer_call_and_return_conditional_losses_315261?2345M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_353_layer_call_and_return_conditional_losses_315279?2345M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_353_layer_call_fn_315230?2345M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_353_layer_call_fn_315243?2345M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
S__inference_batch_normalization_354_layer_call_and_return_conditional_losses_315379?STUVM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
S__inference_batch_normalization_354_layer_call_and_return_conditional_losses_315397?STUVM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
8__inference_batch_normalization_354_layer_call_fn_315348?STUVM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
8__inference_batch_normalization_354_layer_call_fn_315361?STUVM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
S__inference_batch_normalization_355_layer_call_and_return_conditional_losses_315470?mnopM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_355_layer_call_and_return_conditional_losses_315488?mnopM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
8__inference_batch_normalization_355_layer_call_fn_315439?mnopM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
8__inference_batch_normalization_355_layer_call_fn_315452?mnopM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
F__inference_conv2d_293_layer_call_and_return_conditional_losses_315207n"#9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????dd
? ?
+__inference_conv2d_293_layer_call_fn_315197a"#9?6
/?,
*?'
inputs???????????
? " ??????????dd?
F__inference_conv2d_294_layer_call_and_return_conditional_losses_315325lCD7?4
-?*
(?%
inputs?????????dd
? "-?*
#? 
0?????????22 
? ?
+__inference_conv2d_294_layer_call_fn_315315_CD7?4
-?*
(?%
inputs?????????dd
? " ??????????22 ?
F__inference_conv2d_295_layer_call_and_return_conditional_losses_315416l]^7?4
-?*
(?%
inputs?????????22 
? "-?*
#? 
0?????????@
? ?
+__inference_conv2d_295_layer_call_fn_315406_]^7?4
-?*
(?%
inputs?????????22 
? " ??????????@?
F__inference_conv2d_296_layer_call_and_return_conditional_losses_315534l~7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0????????? 
? ?
+__inference_conv2d_296_layer_call_fn_315524_~7?4
-?*
(?%
inputs?????????@
? " ?????????? ?
P__inference_conv2d_transpose_241_layer_call_and_return_conditional_losses_315653???I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????@
? ?
5__inference_conv2d_transpose_241_layer_call_fn_315615???I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+???????????????????????????@?
P__inference_conv2d_transpose_242_layer_call_and_return_conditional_losses_315696???I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
5__inference_conv2d_transpose_242_layer_call_fn_315662???I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
P__inference_conv2d_transpose_243_layer_call_and_return_conditional_losses_315739???I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
5__inference_conv2d_transpose_243_layer_call_fn_315705???I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
C__inference_decoded_layer_call_and_return_conditional_losses_315782???I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
(__inference_decoded_layer_call_fn_315748???I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
F__inference_dropout_94_layer_call_and_return_conditional_losses_315294l;?8
1?.
(?%
inputs?????????dd
p 
? "-?*
#? 
0?????????dd
? ?
F__inference_dropout_94_layer_call_and_return_conditional_losses_315306l;?8
1?.
(?%
inputs?????????dd
p
? "-?*
#? 
0?????????dd
? ?
+__inference_dropout_94_layer_call_fn_315284_;?8
1?.
(?%
inputs?????????dd
p 
? " ??????????dd?
+__inference_dropout_94_layer_call_fn_315289_;?8
1?.
(?%
inputs?????????dd
p
? " ??????????dd?
F__inference_dropout_95_layer_call_and_return_conditional_losses_315503l;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
F__inference_dropout_95_layer_call_and_return_conditional_losses_315515l;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
+__inference_dropout_95_layer_call_fn_315493_;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
+__inference_dropout_95_layer_call_fn_315498_;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
C__inference_encoded_layer_call_and_return_conditional_losses_315588?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
C__inference_encoded_layer_call_and_return_conditional_losses_315606?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
(__inference_encoded_layer_call_fn_315557?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
(__inference_encoded_layer_call_fn_315570?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
I__inference_sequential_36_layer_call_and_return_conditional_losses_314527?,"#2345CDSTUV]^mnop~????????????K?H
A?>
4?1
conv2d_293_input???????????
p 

 
? "/?,
%?"
0???????????
? ?
I__inference_sequential_36_layer_call_and_return_conditional_losses_314615?,"#2345CDSTUV]^mnop~????????????K?H
A?>
4?1
conv2d_293_input???????????
p

 
? "/?,
%?"
0???????????
? ?
I__inference_sequential_36_layer_call_and_return_conditional_losses_315002?,"#2345CDSTUV]^mnop~????????????A?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
I__inference_sequential_36_layer_call_and_return_conditional_losses_315188?,"#2345CDSTUV]^mnop~????????????A?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????
? ?
.__inference_sequential_36_layer_call_fn_314034?,"#2345CDSTUV]^mnop~????????????K?H
A?>
4?1
conv2d_293_input???????????
p 

 
? ""?????????????
.__inference_sequential_36_layer_call_fn_314439?,"#2345CDSTUV]^mnop~????????????K?H
A?>
4?1
conv2d_293_input???????????
p

 
? ""?????????????
.__inference_sequential_36_layer_call_fn_314761?,"#2345CDSTUV]^mnop~????????????A?>
7?4
*?'
inputs???????????
p 

 
? ""?????????????
.__inference_sequential_36_layer_call_fn_314830?,"#2345CDSTUV]^mnop~????????????A?>
7?4
*?'
inputs???????????
p

 
? ""?????????????
$__inference_signature_wrapper_314692?,"#2345CDSTUV]^mnop~????????????W?T
? 
M?J
H
conv2d_293_input4?1
conv2d_293_input???????????";?8
6
decoded+?(
decoded???????????