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
shape:*$
shared_nameAdam/decoded/bias/v
w
'Adam/decoded/bias/v/Read/ReadVariableOpReadVariableOpAdam/decoded/bias/v*
_output_shapes
:*
dtype0
?
Adam/decoded/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/decoded/kernel/v
?
)Adam/decoded/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decoded/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_14/bias/v
?
3Adam/conv2d_transpose_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_14/bias/v*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_14/kernel/v
?
5Adam/conv2d_transpose_14/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_14/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_13/bias/v
?
3Adam/conv2d_transpose_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_13/bias/v*
_output_shapes
: *
dtype0
?
!Adam/conv2d_transpose_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/conv2d_transpose_13/kernel/v
?
5Adam/conv2d_transpose_13/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_13/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_transpose_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/conv2d_transpose_12/bias/v
?
3Adam/conv2d_transpose_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_12/bias/v*
_output_shapes
:@*
dtype0
?
!Adam/conv2d_transpose_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *2
shared_name#!Adam/conv2d_transpose_12/kernel/v
?
5Adam/conv2d_transpose_12/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_12/kernel/v*&
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
Adam/conv2d_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_19/bias/v
{
)Adam/conv2d_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *(
shared_nameAdam/conv2d_19/kernel/v
?
+Adam/conv2d_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/v*&
_output_shapes
:@ *
dtype0
?
"Adam/batch_normalization_14/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_14/beta/v
?
6Adam/batch_normalization_14/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_14/beta/v*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_14/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_14/gamma/v
?
7Adam/batch_normalization_14/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_14/gamma/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_18/bias/v
{
)Adam/conv2d_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_18/kernel/v
?
+Adam/conv2d_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/v*&
_output_shapes
: @*
dtype0
?
"Adam/batch_normalization_13/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_13/beta/v
?
6Adam/batch_normalization_13/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_13/beta/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_13/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_13/gamma/v
?
7Adam/batch_normalization_13/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_13/gamma/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_17/bias/v
{
)Adam/conv2d_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_17/kernel/v
?
+Adam/conv2d_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/kernel/v*&
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_12/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_12/beta/v
?
6Adam/batch_normalization_12/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_12/beta/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_12/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_12/gamma/v
?
7Adam/batch_normalization_12/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_12/gamma/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_16/bias/v
{
)Adam/conv2d_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_16/kernel/v
?
+Adam/conv2d_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/kernel/v*&
_output_shapes
:*
dtype0
~
Adam/decoded/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/decoded/bias/m
w
'Adam/decoded/bias/m/Read/ReadVariableOpReadVariableOpAdam/decoded/bias/m*
_output_shapes
:*
dtype0
?
Adam/decoded/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/decoded/kernel/m
?
)Adam/decoded/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decoded/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_14/bias/m
?
3Adam/conv2d_transpose_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_14/bias/m*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_14/kernel/m
?
5Adam/conv2d_transpose_14/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_14/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_13/bias/m
?
3Adam/conv2d_transpose_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_13/bias/m*
_output_shapes
: *
dtype0
?
!Adam/conv2d_transpose_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/conv2d_transpose_13/kernel/m
?
5Adam/conv2d_transpose_13/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_13/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_transpose_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/conv2d_transpose_12/bias/m
?
3Adam/conv2d_transpose_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_12/bias/m*
_output_shapes
:@*
dtype0
?
!Adam/conv2d_transpose_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *2
shared_name#!Adam/conv2d_transpose_12/kernel/m
?
5Adam/conv2d_transpose_12/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_12/kernel/m*&
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
Adam/conv2d_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_19/bias/m
{
)Adam/conv2d_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *(
shared_nameAdam/conv2d_19/kernel/m
?
+Adam/conv2d_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/m*&
_output_shapes
:@ *
dtype0
?
"Adam/batch_normalization_14/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_14/beta/m
?
6Adam/batch_normalization_14/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_14/beta/m*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_14/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_14/gamma/m
?
7Adam/batch_normalization_14/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_14/gamma/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_18/bias/m
{
)Adam/conv2d_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_18/kernel/m
?
+Adam/conv2d_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/m*&
_output_shapes
: @*
dtype0
?
"Adam/batch_normalization_13/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_13/beta/m
?
6Adam/batch_normalization_13/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_13/beta/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_13/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_13/gamma/m
?
7Adam/batch_normalization_13/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_13/gamma/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_17/bias/m
{
)Adam/conv2d_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_17/kernel/m
?
+Adam/conv2d_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/kernel/m*&
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_12/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_12/beta/m
?
6Adam/batch_normalization_12/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_12/beta/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_12/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_12/gamma/m
?
7Adam/batch_normalization_12/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_12/gamma/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_16/bias/m
{
)Adam/conv2d_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_16/kernel/m
?
+Adam/conv2d_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/kernel/m*&
_output_shapes
:*
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
shape:*
shared_namedecoded/bias
i
 decoded/bias/Read/ReadVariableOpReadVariableOpdecoded/bias*
_output_shapes
:*
dtype0
?
decoded/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedecoded/kernel
y
"decoded/kernel/Read/ReadVariableOpReadVariableOpdecoded/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_14/bias
?
,conv2d_transpose_14/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_14/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_14/kernel
?
.conv2d_transpose_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_14/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_13/bias
?
,conv2d_transpose_13/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_13/bias*
_output_shapes
: *
dtype0
?
conv2d_transpose_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameconv2d_transpose_13/kernel
?
.conv2d_transpose_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_13/kernel*&
_output_shapes
: @*
dtype0
?
conv2d_transpose_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv2d_transpose_12/bias
?
,conv2d_transpose_12/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_12/bias*
_output_shapes
:@*
dtype0
?
conv2d_transpose_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *+
shared_nameconv2d_transpose_12/kernel
?
.conv2d_transpose_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_12/kernel*&
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
t
conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_19/bias
m
"conv2d_19/bias/Read/ReadVariableOpReadVariableOpconv2d_19/bias*
_output_shapes
: *
dtype0
?
conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *!
shared_nameconv2d_19/kernel
}
$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*&
_output_shapes
:@ *
dtype0
?
&batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_14/moving_variance
?
:batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_14/moving_variance*
_output_shapes
:@*
dtype0
?
"batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_14/moving_mean
?
6batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_14/moving_mean*
_output_shapes
:@*
dtype0
?
batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_14/beta
?
/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_14/beta*
_output_shapes
:@*
dtype0
?
batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_14/gamma
?
0batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_14/gamma*
_output_shapes
:@*
dtype0
t
conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_18/bias
m
"conv2d_18/bias/Read/ReadVariableOpReadVariableOpconv2d_18/bias*
_output_shapes
:@*
dtype0
?
conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_18/kernel
}
$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*&
_output_shapes
: @*
dtype0
?
&batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_13/moving_variance
?
:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes
: *
dtype0
?
"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_13/moving_mean
?
6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
_output_shapes
: *
dtype0
?
batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_13/beta
?
/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
_output_shapes
: *
dtype0
?
batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_13/gamma
?
0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes
: *
dtype0
t
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_17/bias
m
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes
: *
dtype0
?
conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_17/kernel
}
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*&
_output_shapes
: *
dtype0
?
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_12/moving_variance
?
:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes
:*
dtype0
?
"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_12/moving_mean
?
6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes
:*
dtype0
?
batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_12/beta
?
/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes
:*
dtype0
?
batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_12/gamma
?
0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes
:*
dtype0
t
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_16/bias
m
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes
:*
dtype0
?
conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_16/kernel
}
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*&
_output_shapes
:*
dtype0
?
serving_default_conv2d_16_inputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?	
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_16_inputconv2d_16/kernelconv2d_16/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_17/kernelconv2d_17/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_18/kernelconv2d_18/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_varianceconv2d_19/kernelconv2d_19/biasencoded/gammaencoded/betaencoded/moving_meanencoded/moving_varianceconv2d_transpose_12/kernelconv2d_transpose_12/biasconv2d_transpose_13/kernelconv2d_transpose_13/biasconv2d_transpose_14/kernelconv2d_transpose_14/biasdecoded/kerneldecoded/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2 *0J 8? *-
f(R&
$__inference_signature_wrapper_492228

NoOpNoOp
??
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
`Z
VARIABLE_VALUEconv2d_16/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_16/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
ke
VARIABLE_VALUEbatch_normalization_12/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_12/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_12/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_12/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
`Z
VARIABLE_VALUEconv2d_17/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_17/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
ke
VARIABLE_VALUEbatch_normalization_13/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_13/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_13/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_13/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
`Z
VARIABLE_VALUEconv2d_18/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_18/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
ke
VARIABLE_VALUEbatch_normalization_14/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_14/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_14/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_14/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
`Z
VARIABLE_VALUEconv2d_19/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_19/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
jd
VARIABLE_VALUEconv2d_transpose_12/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_12/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
jd
VARIABLE_VALUEconv2d_transpose_13/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_13/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_transpose_14/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEconv2d_transpose_14/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
?}
VARIABLE_VALUEAdam/conv2d_16/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_16/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_12/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_12/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_17/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_17/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_13/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_13/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_18/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_18/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_14/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_14/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_19/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_19/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/encoded/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/encoded/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_12/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_12/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_13/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_13/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_14/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_14/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/decoded/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/decoded/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_16/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_16/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_12/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_12/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_17/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_17/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_13/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_13/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_18/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_18/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_14/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_14/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_19/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_19/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/encoded/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/encoded/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_12/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_12/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_13/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_13/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_14/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_14/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
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
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOp0batch_normalization_13/gamma/Read/ReadVariableOp/batch_normalization_13/beta/Read/ReadVariableOp6batch_normalization_13/moving_mean/Read/ReadVariableOp:batch_normalization_13/moving_variance/Read/ReadVariableOp$conv2d_18/kernel/Read/ReadVariableOp"conv2d_18/bias/Read/ReadVariableOp0batch_normalization_14/gamma/Read/ReadVariableOp/batch_normalization_14/beta/Read/ReadVariableOp6batch_normalization_14/moving_mean/Read/ReadVariableOp:batch_normalization_14/moving_variance/Read/ReadVariableOp$conv2d_19/kernel/Read/ReadVariableOp"conv2d_19/bias/Read/ReadVariableOp!encoded/gamma/Read/ReadVariableOp encoded/beta/Read/ReadVariableOp'encoded/moving_mean/Read/ReadVariableOp+encoded/moving_variance/Read/ReadVariableOp.conv2d_transpose_12/kernel/Read/ReadVariableOp,conv2d_transpose_12/bias/Read/ReadVariableOp.conv2d_transpose_13/kernel/Read/ReadVariableOp,conv2d_transpose_13/bias/Read/ReadVariableOp.conv2d_transpose_14/kernel/Read/ReadVariableOp,conv2d_transpose_14/bias/Read/ReadVariableOp"decoded/kernel/Read/ReadVariableOp decoded/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_16/kernel/m/Read/ReadVariableOp)Adam/conv2d_16/bias/m/Read/ReadVariableOp7Adam/batch_normalization_12/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_12/beta/m/Read/ReadVariableOp+Adam/conv2d_17/kernel/m/Read/ReadVariableOp)Adam/conv2d_17/bias/m/Read/ReadVariableOp7Adam/batch_normalization_13/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_13/beta/m/Read/ReadVariableOp+Adam/conv2d_18/kernel/m/Read/ReadVariableOp)Adam/conv2d_18/bias/m/Read/ReadVariableOp7Adam/batch_normalization_14/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_14/beta/m/Read/ReadVariableOp+Adam/conv2d_19/kernel/m/Read/ReadVariableOp)Adam/conv2d_19/bias/m/Read/ReadVariableOp(Adam/encoded/gamma/m/Read/ReadVariableOp'Adam/encoded/beta/m/Read/ReadVariableOp5Adam/conv2d_transpose_12/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_12/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_13/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_13/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_14/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_14/bias/m/Read/ReadVariableOp)Adam/decoded/kernel/m/Read/ReadVariableOp'Adam/decoded/bias/m/Read/ReadVariableOp+Adam/conv2d_16/kernel/v/Read/ReadVariableOp)Adam/conv2d_16/bias/v/Read/ReadVariableOp7Adam/batch_normalization_12/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_12/beta/v/Read/ReadVariableOp+Adam/conv2d_17/kernel/v/Read/ReadVariableOp)Adam/conv2d_17/bias/v/Read/ReadVariableOp7Adam/batch_normalization_13/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_13/beta/v/Read/ReadVariableOp+Adam/conv2d_18/kernel/v/Read/ReadVariableOp)Adam/conv2d_18/bias/v/Read/ReadVariableOp7Adam/batch_normalization_14/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_14/beta/v/Read/ReadVariableOp+Adam/conv2d_19/kernel/v/Read/ReadVariableOp)Adam/conv2d_19/bias/v/Read/ReadVariableOp(Adam/encoded/gamma/v/Read/ReadVariableOp'Adam/encoded/beta/v/Read/ReadVariableOp5Adam/conv2d_transpose_12/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_12/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_13/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_13/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_14/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_14/bias/v/Read/ReadVariableOp)Adam/decoded/kernel/v/Read/ReadVariableOp'Adam/decoded/bias/v/Read/ReadVariableOpConst*d
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
__inference__traced_save_493598
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_16/kernelconv2d_16/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_17/kernelconv2d_17/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_18/kernelconv2d_18/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_varianceconv2d_19/kernelconv2d_19/biasencoded/gammaencoded/betaencoded/moving_meanencoded/moving_varianceconv2d_transpose_12/kernelconv2d_transpose_12/biasconv2d_transpose_13/kernelconv2d_transpose_13/biasconv2d_transpose_14/kernelconv2d_transpose_14/biasdecoded/kerneldecoded/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_16/kernel/mAdam/conv2d_16/bias/m#Adam/batch_normalization_12/gamma/m"Adam/batch_normalization_12/beta/mAdam/conv2d_17/kernel/mAdam/conv2d_17/bias/m#Adam/batch_normalization_13/gamma/m"Adam/batch_normalization_13/beta/mAdam/conv2d_18/kernel/mAdam/conv2d_18/bias/m#Adam/batch_normalization_14/gamma/m"Adam/batch_normalization_14/beta/mAdam/conv2d_19/kernel/mAdam/conv2d_19/bias/mAdam/encoded/gamma/mAdam/encoded/beta/m!Adam/conv2d_transpose_12/kernel/mAdam/conv2d_transpose_12/bias/m!Adam/conv2d_transpose_13/kernel/mAdam/conv2d_transpose_13/bias/m!Adam/conv2d_transpose_14/kernel/mAdam/conv2d_transpose_14/bias/mAdam/decoded/kernel/mAdam/decoded/bias/mAdam/conv2d_16/kernel/vAdam/conv2d_16/bias/v#Adam/batch_normalization_12/gamma/v"Adam/batch_normalization_12/beta/vAdam/conv2d_17/kernel/vAdam/conv2d_17/bias/v#Adam/batch_normalization_13/gamma/v"Adam/batch_normalization_13/beta/vAdam/conv2d_18/kernel/vAdam/conv2d_18/bias/v#Adam/batch_normalization_14/gamma/v"Adam/batch_normalization_14/beta/vAdam/conv2d_19/kernel/vAdam/conv2d_19/bias/vAdam/encoded/gamma/vAdam/encoded/beta/v!Adam/conv2d_transpose_12/kernel/vAdam/conv2d_transpose_12/bias/v!Adam/conv2d_transpose_13/kernel/vAdam/conv2d_transpose_13/bias/v!Adam/conv2d_transpose_14/kernel/vAdam/conv2d_transpose_14/bias/vAdam/decoded/kernel/vAdam/decoded/bias/v*c
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
"__inference__traced_restore_493869??
?
?
C__inference_encoded_layer_call_and_return_conditional_losses_493142

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
?	
?
7__inference_batch_normalization_13_layer_call_fn_492897

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
GPU2 *0J 8? *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_491012?
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
?
?
-__inference_sequential_4_layer_call_fn_492297

inputs!
unknown:
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

unknown_29:

unknown_30:
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
:???????????*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_491503y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
C__inference_encoded_layer_call_and_return_conditional_losses_491109

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
?
c
*__inference_dropout_9_layer_call_fn_493034

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
:?????????  @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_491606w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
-__inference_sequential_4_layer_call_fn_491975
conv2d_16_input!
unknown:
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

unknown_29:

unknown_30:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:???????????*:
_read_only_resource_inputs
	
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_491839y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_16_input
?
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_491446

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????  @c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????  @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_493039

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????  @c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????  @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_491045

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
e
I__inference_activation_17_layer_call_and_return_conditional_losses_492871

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????@@ b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@ :W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?

?
E__inference_conv2d_19_layer_call_and_return_conditional_losses_493070

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
:????????? *
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
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_492815

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
?	
?
7__inference_batch_normalization_12_layer_call_fn_492766

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
GPU2 *0J 8? *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_490917?
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
?

?
E__inference_conv2d_19_layer_call_and_return_conditional_losses_491458

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
:????????? *
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
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
e
I__inference_activation_18_layer_call_and_return_conditional_losses_492962

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????  @b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?

d
E__inference_dropout_8_layer_call_and_return_conditional_losses_492842

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??n
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:???????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????y
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????s
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:???????????c
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?]
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_492063
conv2d_16_input*
conv2d_16_491978:
conv2d_16_491980:+
batch_normalization_12_491984:+
batch_normalization_12_491986:+
batch_normalization_12_491988:+
batch_normalization_12_491990:*
conv2d_17_491994: 
conv2d_17_491996: +
batch_normalization_13_492000: +
batch_normalization_13_492002: +
batch_normalization_13_492004: +
batch_normalization_13_492006: *
conv2d_18_492009: @
conv2d_18_492011:@+
batch_normalization_14_492015:@+
batch_normalization_14_492017:@+
batch_normalization_14_492019:@+
batch_normalization_14_492021:@*
conv2d_19_492025:@ 
conv2d_19_492027: 
encoded_492032: 
encoded_492034: 
encoded_492036: 
encoded_492038: 4
conv2d_transpose_12_492042:@ (
conv2d_transpose_12_492044:@4
conv2d_transpose_13_492047: @(
conv2d_transpose_13_492049: 4
conv2d_transpose_14_492052: (
conv2d_transpose_14_492054:(
decoded_492057:
decoded_492059:
identity??.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall?+conv2d_transpose_12/StatefulPartitionedCall?+conv2d_transpose_13/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?decoded/StatefulPartitionedCall?encoded/StatefulPartitionedCall?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallconv2d_16_inputconv2d_16_491978conv2d_16_491980*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_491348?
activation_16/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_491359?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0batch_normalization_12_491984batch_normalization_12_491986batch_normalization_12_491988batch_normalization_12_491990*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_490917?
dropout_8/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_491375?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0conv2d_17_491994conv2d_17_491996*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_491387?
activation_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_491398?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall&activation_17/PartitionedCall:output:0batch_normalization_13_492000batch_normalization_13_492002batch_normalization_13_492004batch_normalization_13_492006*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_490981?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv2d_18_492009conv2d_18_492011*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_491419?
activation_18/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_491430?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0batch_normalization_14_492015batch_normalization_14_492017batch_normalization_14_492019batch_normalization_14_492021*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_491045?
dropout_9/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_491446?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0conv2d_19_492025conv2d_19_492027*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_491458?
activation_19/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_491469?
encoded/CastCast&activation_19/PartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
encoded/StatefulPartitionedCallStatefulPartitionedCallencoded/Cast:y:0encoded_492032encoded_492034encoded_492036encoded_492038*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_encoded_layer_call_and_return_conditional_losses_491109?
conv2d_transpose_12/CastCast(encoded/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_12/Cast:y:0conv2d_transpose_12_492042conv2d_transpose_12_492044*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_491189?
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0conv2d_transpose_13_492047conv2d_transpose_13_492049*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_491234?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_13/StatefulPartitionedCall:output:0conv2d_transpose_14_492052conv2d_transpose_14_492054*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_491279?
decoded/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0decoded_492057decoded_492059*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_decoded_layer_call_and_return_conditional_losses_491324?
IdentityIdentity(decoded/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall ^decoded/StatefulPartitionedCall ^encoded/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2B
decoded/StatefulPartitionedCalldecoded/StatefulPartitionedCall2B
encoded/StatefulPartitionedCallencoded/StatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_16_input
?
e
I__inference_activation_17_layer_call_and_return_conditional_losses_491398

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????@@ b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@ :W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_14_layer_call_fn_493237

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
GPU2 *0J 8? *X
fSRQ
O__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_491279?
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
O__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_491189

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
value	B :@y
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
:@ *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
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
?
J
.__inference_activation_19_layer_call_fn_493075

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
:????????? * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_491469h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
-__inference_sequential_4_layer_call_fn_492366

inputs!
unknown:
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

unknown_29:

unknown_30:
identity??StatefulPartitionedCall?
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
:???????????*:
_read_only_resource_inputs
	
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_491839y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_492915

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
?
e
I__inference_activation_18_layer_call_and_return_conditional_losses_491430

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????  @b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
*__inference_conv2d_16_layer_call_fn_492733

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_491348y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?:
"__inference__traced_restore_493869
file_prefix;
!assignvariableop_conv2d_16_kernel:/
!assignvariableop_1_conv2d_16_bias:=
/assignvariableop_2_batch_normalization_12_gamma:<
.assignvariableop_3_batch_normalization_12_beta:C
5assignvariableop_4_batch_normalization_12_moving_mean:G
9assignvariableop_5_batch_normalization_12_moving_variance:=
#assignvariableop_6_conv2d_17_kernel: /
!assignvariableop_7_conv2d_17_bias: =
/assignvariableop_8_batch_normalization_13_gamma: <
.assignvariableop_9_batch_normalization_13_beta: D
6assignvariableop_10_batch_normalization_13_moving_mean: H
:assignvariableop_11_batch_normalization_13_moving_variance: >
$assignvariableop_12_conv2d_18_kernel: @0
"assignvariableop_13_conv2d_18_bias:@>
0assignvariableop_14_batch_normalization_14_gamma:@=
/assignvariableop_15_batch_normalization_14_beta:@D
6assignvariableop_16_batch_normalization_14_moving_mean:@H
:assignvariableop_17_batch_normalization_14_moving_variance:@>
$assignvariableop_18_conv2d_19_kernel:@ 0
"assignvariableop_19_conv2d_19_bias: /
!assignvariableop_20_encoded_gamma: .
 assignvariableop_21_encoded_beta: 5
'assignvariableop_22_encoded_moving_mean: 9
+assignvariableop_23_encoded_moving_variance: H
.assignvariableop_24_conv2d_transpose_12_kernel:@ :
,assignvariableop_25_conv2d_transpose_12_bias:@H
.assignvariableop_26_conv2d_transpose_13_kernel: @:
,assignvariableop_27_conv2d_transpose_13_bias: H
.assignvariableop_28_conv2d_transpose_14_kernel: :
,assignvariableop_29_conv2d_transpose_14_bias:<
"assignvariableop_30_decoded_kernel:.
 assignvariableop_31_decoded_bias:'
assignvariableop_32_adam_iter:	 )
assignvariableop_33_adam_beta_1: )
assignvariableop_34_adam_beta_2: (
assignvariableop_35_adam_decay: 0
&assignvariableop_36_adam_learning_rate: #
assignvariableop_37_total: #
assignvariableop_38_count: E
+assignvariableop_39_adam_conv2d_16_kernel_m:7
)assignvariableop_40_adam_conv2d_16_bias_m:E
7assignvariableop_41_adam_batch_normalization_12_gamma_m:D
6assignvariableop_42_adam_batch_normalization_12_beta_m:E
+assignvariableop_43_adam_conv2d_17_kernel_m: 7
)assignvariableop_44_adam_conv2d_17_bias_m: E
7assignvariableop_45_adam_batch_normalization_13_gamma_m: D
6assignvariableop_46_adam_batch_normalization_13_beta_m: E
+assignvariableop_47_adam_conv2d_18_kernel_m: @7
)assignvariableop_48_adam_conv2d_18_bias_m:@E
7assignvariableop_49_adam_batch_normalization_14_gamma_m:@D
6assignvariableop_50_adam_batch_normalization_14_beta_m:@E
+assignvariableop_51_adam_conv2d_19_kernel_m:@ 7
)assignvariableop_52_adam_conv2d_19_bias_m: 6
(assignvariableop_53_adam_encoded_gamma_m: 5
'assignvariableop_54_adam_encoded_beta_m: O
5assignvariableop_55_adam_conv2d_transpose_12_kernel_m:@ A
3assignvariableop_56_adam_conv2d_transpose_12_bias_m:@O
5assignvariableop_57_adam_conv2d_transpose_13_kernel_m: @A
3assignvariableop_58_adam_conv2d_transpose_13_bias_m: O
5assignvariableop_59_adam_conv2d_transpose_14_kernel_m: A
3assignvariableop_60_adam_conv2d_transpose_14_bias_m:C
)assignvariableop_61_adam_decoded_kernel_m:5
'assignvariableop_62_adam_decoded_bias_m:E
+assignvariableop_63_adam_conv2d_16_kernel_v:7
)assignvariableop_64_adam_conv2d_16_bias_v:E
7assignvariableop_65_adam_batch_normalization_12_gamma_v:D
6assignvariableop_66_adam_batch_normalization_12_beta_v:E
+assignvariableop_67_adam_conv2d_17_kernel_v: 7
)assignvariableop_68_adam_conv2d_17_bias_v: E
7assignvariableop_69_adam_batch_normalization_13_gamma_v: D
6assignvariableop_70_adam_batch_normalization_13_beta_v: E
+assignvariableop_71_adam_conv2d_18_kernel_v: @7
)assignvariableop_72_adam_conv2d_18_bias_v:@E
7assignvariableop_73_adam_batch_normalization_14_gamma_v:@D
6assignvariableop_74_adam_batch_normalization_14_beta_v:@E
+assignvariableop_75_adam_conv2d_19_kernel_v:@ 7
)assignvariableop_76_adam_conv2d_19_bias_v: 6
(assignvariableop_77_adam_encoded_gamma_v: 5
'assignvariableop_78_adam_encoded_beta_v: O
5assignvariableop_79_adam_conv2d_transpose_12_kernel_v:@ A
3assignvariableop_80_adam_conv2d_transpose_12_bias_v:@O
5assignvariableop_81_adam_conv2d_transpose_13_kernel_v: @A
3assignvariableop_82_adam_conv2d_transpose_13_bias_v: O
5assignvariableop_83_adam_conv2d_transpose_14_kernel_v: A
3assignvariableop_84_adam_conv2d_transpose_14_bias_v:C
)assignvariableop_85_adam_decoded_kernel_v:5
'assignvariableop_86_adam_decoded_bias_v:
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
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_16_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_16_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_12_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_12_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_12_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_12_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_17_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_17_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_13_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_13_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_13_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_13_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_18_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_18_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_14_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_14_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_14_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_14_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_19_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_19_biasIdentity_19:output:0"/device:CPU:0*
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
AssignVariableOp_24AssignVariableOp.assignvariableop_24_conv2d_transpose_12_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp,assignvariableop_25_conv2d_transpose_12_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp.assignvariableop_26_conv2d_transpose_13_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp,assignvariableop_27_conv2d_transpose_13_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp.assignvariableop_28_conv2d_transpose_14_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp,assignvariableop_29_conv2d_transpose_14_biasIdentity_29:output:0"/device:CPU:0*
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
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_16_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_16_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp7assignvariableop_41_adam_batch_normalization_12_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_batch_normalization_12_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_17_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_17_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp7assignvariableop_45_adam_batch_normalization_13_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adam_batch_normalization_13_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_18_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_18_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp7assignvariableop_49_adam_batch_normalization_14_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adam_batch_normalization_14_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv2d_19_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv2d_19_bias_mIdentity_52:output:0"/device:CPU:0*
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
AssignVariableOp_55AssignVariableOp5assignvariableop_55_adam_conv2d_transpose_12_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp3assignvariableop_56_adam_conv2d_transpose_12_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp5assignvariableop_57_adam_conv2d_transpose_13_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp3assignvariableop_58_adam_conv2d_transpose_13_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp5assignvariableop_59_adam_conv2d_transpose_14_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp3assignvariableop_60_adam_conv2d_transpose_14_bias_mIdentity_60:output:0"/device:CPU:0*
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
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_conv2d_16_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_conv2d_16_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp7assignvariableop_65_adam_batch_normalization_12_gamma_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp6assignvariableop_66_adam_batch_normalization_12_beta_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_conv2d_17_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_conv2d_17_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp7assignvariableop_69_adam_batch_normalization_13_gamma_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp6assignvariableop_70_adam_batch_normalization_13_beta_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_conv2d_18_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_conv2d_18_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp7assignvariableop_73_adam_batch_normalization_14_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp6assignvariableop_74_adam_batch_normalization_14_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_conv2d_19_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_conv2d_19_bias_vIdentity_76:output:0"/device:CPU:0*
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
AssignVariableOp_79AssignVariableOp5assignvariableop_79_adam_conv2d_transpose_12_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOp3assignvariableop_80_adam_conv2d_transpose_12_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOp5assignvariableop_81_adam_conv2d_transpose_13_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOp3assignvariableop_82_adam_conv2d_transpose_13_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOp5assignvariableop_83_adam_conv2d_transpose_14_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOp3assignvariableop_84_adam_conv2d_transpose_14_bias_vIdentity_84:output:0"/device:CPU:0*
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
?
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_492830

inputs

identity_1X
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????e

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:???????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_12_layer_call_fn_492779

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
GPU2 *0J 8? *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_490948?
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
?

?
E__inference_conv2d_16_layer_call_and_return_conditional_losses_492743

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_492228
conv2d_16_input!
unknown:
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

unknown_29:

unknown_30:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:???????????*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2 *0J 8? **
f%R#
!__inference__wrapped_model_490895y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_16_input
?

?
E__inference_conv2d_18_layer_call_and_return_conditional_losses_491419

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
:?????????  @*
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
:?????????  @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????  @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
?
C__inference_encoded_layer_call_and_return_conditional_losses_493124

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
?
e
I__inference_activation_19_layer_call_and_return_conditional_losses_493080

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:????????? b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
-__inference_sequential_4_layer_call_fn_491570
conv2d_16_input!
unknown:
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

unknown_29:

unknown_30:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:???????????*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_491503y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_16_input
?
F
*__inference_dropout_9_layer_call_fn_493029

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
:?????????  @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_491446h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
(__inference_encoded_layer_call_fn_493106

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
C__inference_encoded_layer_call_and_return_conditional_losses_491140?
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
?
J
.__inference_activation_18_layer_call_fn_492957

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
:?????????  @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_491430h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?

?
E__inference_conv2d_18_layer_call_and_return_conditional_losses_492952

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
:?????????  @*
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
:?????????  @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????  @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
?
C__inference_encoded_layer_call_and_return_conditional_losses_491140

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
?

d
E__inference_dropout_9_layer_call_and_return_conditional_losses_493051

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
:?????????  @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????  @*
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
:?????????  @w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  @q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  @a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
*__inference_conv2d_17_layer_call_fn_492851

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
:?????????@@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_491387w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_14_layer_call_fn_492988

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
GPU2 *0J 8? *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_491076?
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

d
E__inference_dropout_8_layer_call_and_return_conditional_losses_491661

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??n
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:???????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????y
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????s
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:???????????c
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_19_layer_call_fn_493060

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
:????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_491458w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_493024

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
?
?
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_493006

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
?!
?
O__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_491279

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
?!
?
O__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_491234

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
??
?'
__inference__traced_save_493598
file_prefix/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableop/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop;
7savev2_batch_normalization_13_gamma_read_readvariableop:
6savev2_batch_normalization_13_beta_read_readvariableopA
=savev2_batch_normalization_13_moving_mean_read_readvariableopE
Asavev2_batch_normalization_13_moving_variance_read_readvariableop/
+savev2_conv2d_18_kernel_read_readvariableop-
)savev2_conv2d_18_bias_read_readvariableop;
7savev2_batch_normalization_14_gamma_read_readvariableop:
6savev2_batch_normalization_14_beta_read_readvariableopA
=savev2_batch_normalization_14_moving_mean_read_readvariableopE
Asavev2_batch_normalization_14_moving_variance_read_readvariableop/
+savev2_conv2d_19_kernel_read_readvariableop-
)savev2_conv2d_19_bias_read_readvariableop,
(savev2_encoded_gamma_read_readvariableop+
'savev2_encoded_beta_read_readvariableop2
.savev2_encoded_moving_mean_read_readvariableop6
2savev2_encoded_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_12_kernel_read_readvariableop7
3savev2_conv2d_transpose_12_bias_read_readvariableop9
5savev2_conv2d_transpose_13_kernel_read_readvariableop7
3savev2_conv2d_transpose_13_bias_read_readvariableop9
5savev2_conv2d_transpose_14_kernel_read_readvariableop7
3savev2_conv2d_transpose_14_bias_read_readvariableop-
)savev2_decoded_kernel_read_readvariableop+
'savev2_decoded_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_16_kernel_m_read_readvariableop4
0savev2_adam_conv2d_16_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_12_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_12_beta_m_read_readvariableop6
2savev2_adam_conv2d_17_kernel_m_read_readvariableop4
0savev2_adam_conv2d_17_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_13_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_13_beta_m_read_readvariableop6
2savev2_adam_conv2d_18_kernel_m_read_readvariableop4
0savev2_adam_conv2d_18_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_14_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_14_beta_m_read_readvariableop6
2savev2_adam_conv2d_19_kernel_m_read_readvariableop4
0savev2_adam_conv2d_19_bias_m_read_readvariableop3
/savev2_adam_encoded_gamma_m_read_readvariableop2
.savev2_adam_encoded_beta_m_read_readvariableop@
<savev2_adam_conv2d_transpose_12_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_12_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_13_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_13_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_14_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_14_bias_m_read_readvariableop4
0savev2_adam_decoded_kernel_m_read_readvariableop2
.savev2_adam_decoded_bias_m_read_readvariableop6
2savev2_adam_conv2d_16_kernel_v_read_readvariableop4
0savev2_adam_conv2d_16_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_12_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_12_beta_v_read_readvariableop6
2savev2_adam_conv2d_17_kernel_v_read_readvariableop4
0savev2_adam_conv2d_17_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_13_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_13_beta_v_read_readvariableop6
2savev2_adam_conv2d_18_kernel_v_read_readvariableop4
0savev2_adam_conv2d_18_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_14_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_14_beta_v_read_readvariableop6
2savev2_adam_conv2d_19_kernel_v_read_readvariableop4
0savev2_adam_conv2d_19_bias_v_read_readvariableop3
/savev2_adam_encoded_gamma_v_read_readvariableop2
.savev2_adam_encoded_beta_v_read_readvariableop@
<savev2_adam_conv2d_transpose_12_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_12_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_13_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_13_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_14_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_14_bias_v_read_readvariableop4
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop7savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableop7savev2_batch_normalization_13_gamma_read_readvariableop6savev2_batch_normalization_13_beta_read_readvariableop=savev2_batch_normalization_13_moving_mean_read_readvariableopAsavev2_batch_normalization_13_moving_variance_read_readvariableop+savev2_conv2d_18_kernel_read_readvariableop)savev2_conv2d_18_bias_read_readvariableop7savev2_batch_normalization_14_gamma_read_readvariableop6savev2_batch_normalization_14_beta_read_readvariableop=savev2_batch_normalization_14_moving_mean_read_readvariableopAsavev2_batch_normalization_14_moving_variance_read_readvariableop+savev2_conv2d_19_kernel_read_readvariableop)savev2_conv2d_19_bias_read_readvariableop(savev2_encoded_gamma_read_readvariableop'savev2_encoded_beta_read_readvariableop.savev2_encoded_moving_mean_read_readvariableop2savev2_encoded_moving_variance_read_readvariableop5savev2_conv2d_transpose_12_kernel_read_readvariableop3savev2_conv2d_transpose_12_bias_read_readvariableop5savev2_conv2d_transpose_13_kernel_read_readvariableop3savev2_conv2d_transpose_13_bias_read_readvariableop5savev2_conv2d_transpose_14_kernel_read_readvariableop3savev2_conv2d_transpose_14_bias_read_readvariableop)savev2_decoded_kernel_read_readvariableop'savev2_decoded_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_16_kernel_m_read_readvariableop0savev2_adam_conv2d_16_bias_m_read_readvariableop>savev2_adam_batch_normalization_12_gamma_m_read_readvariableop=savev2_adam_batch_normalization_12_beta_m_read_readvariableop2savev2_adam_conv2d_17_kernel_m_read_readvariableop0savev2_adam_conv2d_17_bias_m_read_readvariableop>savev2_adam_batch_normalization_13_gamma_m_read_readvariableop=savev2_adam_batch_normalization_13_beta_m_read_readvariableop2savev2_adam_conv2d_18_kernel_m_read_readvariableop0savev2_adam_conv2d_18_bias_m_read_readvariableop>savev2_adam_batch_normalization_14_gamma_m_read_readvariableop=savev2_adam_batch_normalization_14_beta_m_read_readvariableop2savev2_adam_conv2d_19_kernel_m_read_readvariableop0savev2_adam_conv2d_19_bias_m_read_readvariableop/savev2_adam_encoded_gamma_m_read_readvariableop.savev2_adam_encoded_beta_m_read_readvariableop<savev2_adam_conv2d_transpose_12_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_12_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_13_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_13_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_14_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_14_bias_m_read_readvariableop0savev2_adam_decoded_kernel_m_read_readvariableop.savev2_adam_decoded_bias_m_read_readvariableop2savev2_adam_conv2d_16_kernel_v_read_readvariableop0savev2_adam_conv2d_16_bias_v_read_readvariableop>savev2_adam_batch_normalization_12_gamma_v_read_readvariableop=savev2_adam_batch_normalization_12_beta_v_read_readvariableop2savev2_adam_conv2d_17_kernel_v_read_readvariableop0savev2_adam_conv2d_17_bias_v_read_readvariableop>savev2_adam_batch_normalization_13_gamma_v_read_readvariableop=savev2_adam_batch_normalization_13_beta_v_read_readvariableop2savev2_adam_conv2d_18_kernel_v_read_readvariableop0savev2_adam_conv2d_18_bias_v_read_readvariableop>savev2_adam_batch_normalization_14_gamma_v_read_readvariableop=savev2_adam_batch_normalization_14_beta_v_read_readvariableop2savev2_adam_conv2d_19_kernel_v_read_readvariableop0savev2_adam_conv2d_19_bias_v_read_readvariableop/savev2_adam_encoded_gamma_v_read_readvariableop.savev2_adam_encoded_beta_v_read_readvariableop<savev2_adam_conv2d_transpose_12_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_12_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_13_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_13_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_14_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_14_bias_v_read_readvariableop0savev2_adam_decoded_kernel_v_read_readvariableop.savev2_adam_decoded_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: ::::::: : : : : : : @:@:@:@:@:@:@ : : : : : :@ :@: @: : :::: : : : : : : ::::: : : : : @:@:@:@:@ : : : :@ :@: @: : :::::::: : : : : @:@:@:@:@ : : : :@ :@: @: : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 
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
::  

_output_shapes
::!
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
:: )
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
:: ?

_output_shapes
::,@(
&
_output_shapes
:: A
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
:: W

_output_shapes
::X

_output_shapes
: 
?
c
*__inference_dropout_8_layer_call_fn_492825

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_491661y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
J
.__inference_activation_16_layer_call_fn_492748

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_491359j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?]
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_491503

inputs*
conv2d_16_491349:
conv2d_16_491351:+
batch_normalization_12_491361:+
batch_normalization_12_491363:+
batch_normalization_12_491365:+
batch_normalization_12_491367:*
conv2d_17_491388: 
conv2d_17_491390: +
batch_normalization_13_491400: +
batch_normalization_13_491402: +
batch_normalization_13_491404: +
batch_normalization_13_491406: *
conv2d_18_491420: @
conv2d_18_491422:@+
batch_normalization_14_491432:@+
batch_normalization_14_491434:@+
batch_normalization_14_491436:@+
batch_normalization_14_491438:@*
conv2d_19_491459:@ 
conv2d_19_491461: 
encoded_491472: 
encoded_491474: 
encoded_491476: 
encoded_491478: 4
conv2d_transpose_12_491482:@ (
conv2d_transpose_12_491484:@4
conv2d_transpose_13_491487: @(
conv2d_transpose_13_491489: 4
conv2d_transpose_14_491492: (
conv2d_transpose_14_491494:(
decoded_491497:
decoded_491499:
identity??.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall?+conv2d_transpose_12/StatefulPartitionedCall?+conv2d_transpose_13/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?decoded/StatefulPartitionedCall?encoded/StatefulPartitionedCall?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_16_491349conv2d_16_491351*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_491348?
activation_16/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_491359?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0batch_normalization_12_491361batch_normalization_12_491363batch_normalization_12_491365batch_normalization_12_491367*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_490917?
dropout_8/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_491375?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0conv2d_17_491388conv2d_17_491390*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_491387?
activation_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_491398?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall&activation_17/PartitionedCall:output:0batch_normalization_13_491400batch_normalization_13_491402batch_normalization_13_491404batch_normalization_13_491406*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_490981?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv2d_18_491420conv2d_18_491422*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_491419?
activation_18/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_491430?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0batch_normalization_14_491432batch_normalization_14_491434batch_normalization_14_491436batch_normalization_14_491438*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_491045?
dropout_9/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_491446?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0conv2d_19_491459conv2d_19_491461*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_491458?
activation_19/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_491469?
encoded/CastCast&activation_19/PartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
encoded/StatefulPartitionedCallStatefulPartitionedCallencoded/Cast:y:0encoded_491472encoded_491474encoded_491476encoded_491478*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_encoded_layer_call_and_return_conditional_losses_491109?
conv2d_transpose_12/CastCast(encoded/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_12/Cast:y:0conv2d_transpose_12_491482conv2d_transpose_12_491484*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_491189?
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0conv2d_transpose_13_491487conv2d_transpose_13_491489*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_491234?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_13/StatefulPartitionedCall:output:0conv2d_transpose_14_491492conv2d_transpose_14_491494*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_491279?
decoded/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0decoded_491497decoded_491499*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_decoded_layer_call_and_return_conditional_losses_491324?
IdentityIdentity(decoded/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall ^decoded/StatefulPartitionedCall ^encoded/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2B
decoded/StatefulPartitionedCalldecoded/StatefulPartitionedCall2B
encoded/StatefulPartitionedCallencoded/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

d
E__inference_dropout_9_layer_call_and_return_conditional_losses_491606

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
:?????????  @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????  @*
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
:?????????  @w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  @q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  @a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?

?
E__inference_conv2d_16_layer_call_and_return_conditional_losses_491348

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_491375

inputs

identity_1X
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????e

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:???????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?!
?
O__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_493228

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
?`
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_492151
conv2d_16_input*
conv2d_16_492066:
conv2d_16_492068:+
batch_normalization_12_492072:+
batch_normalization_12_492074:+
batch_normalization_12_492076:+
batch_normalization_12_492078:*
conv2d_17_492082: 
conv2d_17_492084: +
batch_normalization_13_492088: +
batch_normalization_13_492090: +
batch_normalization_13_492092: +
batch_normalization_13_492094: *
conv2d_18_492097: @
conv2d_18_492099:@+
batch_normalization_14_492103:@+
batch_normalization_14_492105:@+
batch_normalization_14_492107:@+
batch_normalization_14_492109:@*
conv2d_19_492113:@ 
conv2d_19_492115: 
encoded_492120: 
encoded_492122: 
encoded_492124: 
encoded_492126: 4
conv2d_transpose_12_492130:@ (
conv2d_transpose_12_492132:@4
conv2d_transpose_13_492135: @(
conv2d_transpose_13_492137: 4
conv2d_transpose_14_492140: (
conv2d_transpose_14_492142:(
decoded_492145:
decoded_492147:
identity??.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall?+conv2d_transpose_12/StatefulPartitionedCall?+conv2d_transpose_13/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?decoded/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?encoded/StatefulPartitionedCall?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallconv2d_16_inputconv2d_16_492066conv2d_16_492068*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_491348?
activation_16/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_491359?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0batch_normalization_12_492072batch_normalization_12_492074batch_normalization_12_492076batch_normalization_12_492078*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_490948?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_491661?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0conv2d_17_492082conv2d_17_492084*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_491387?
activation_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_491398?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall&activation_17/PartitionedCall:output:0batch_normalization_13_492088batch_normalization_13_492090batch_normalization_13_492092batch_normalization_13_492094*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_491012?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv2d_18_492097conv2d_18_492099*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_491419?
activation_18/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_491430?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0batch_normalization_14_492103batch_normalization_14_492105batch_normalization_14_492107batch_normalization_14_492109*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_491076?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_491606?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0conv2d_19_492113conv2d_19_492115*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_491458?
activation_19/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_491469?
encoded/CastCast&activation_19/PartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
encoded/StatefulPartitionedCallStatefulPartitionedCallencoded/Cast:y:0encoded_492120encoded_492122encoded_492124encoded_492126*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_encoded_layer_call_and_return_conditional_losses_491140?
conv2d_transpose_12/CastCast(encoded/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_12/Cast:y:0conv2d_transpose_12_492130conv2d_transpose_12_492132*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_491189?
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0conv2d_transpose_13_492135conv2d_transpose_13_492137*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_491234?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_13/StatefulPartitionedCall:output:0conv2d_transpose_14_492140conv2d_transpose_14_492142*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_491279?
decoded/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0decoded_492145decoded_492147*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_decoded_layer_call_and_return_conditional_losses_491324?
IdentityIdentity(decoded/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall ^decoded/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall ^encoded/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2B
decoded/StatefulPartitionedCalldecoded/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2B
encoded/StatefulPartitionedCallencoded/StatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_16_input
?!
?
O__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_493185

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
value	B :@y
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
:@ *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
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
(__inference_encoded_layer_call_fn_493093

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
C__inference_encoded_layer_call_and_return_conditional_losses_491109?
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
?
F
*__inference_dropout_8_layer_call_fn_492820

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_491375j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?!
?
C__inference_decoded_layer_call_and_return_conditional_losses_491324

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
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
value	B :y
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
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
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
?`
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_491839

inputs*
conv2d_16_491754:
conv2d_16_491756:+
batch_normalization_12_491760:+
batch_normalization_12_491762:+
batch_normalization_12_491764:+
batch_normalization_12_491766:*
conv2d_17_491770: 
conv2d_17_491772: +
batch_normalization_13_491776: +
batch_normalization_13_491778: +
batch_normalization_13_491780: +
batch_normalization_13_491782: *
conv2d_18_491785: @
conv2d_18_491787:@+
batch_normalization_14_491791:@+
batch_normalization_14_491793:@+
batch_normalization_14_491795:@+
batch_normalization_14_491797:@*
conv2d_19_491801:@ 
conv2d_19_491803: 
encoded_491808: 
encoded_491810: 
encoded_491812: 
encoded_491814: 4
conv2d_transpose_12_491818:@ (
conv2d_transpose_12_491820:@4
conv2d_transpose_13_491823: @(
conv2d_transpose_13_491825: 4
conv2d_transpose_14_491828: (
conv2d_transpose_14_491830:(
decoded_491833:
decoded_491835:
identity??.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall?+conv2d_transpose_12/StatefulPartitionedCall?+conv2d_transpose_13/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?decoded/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?encoded/StatefulPartitionedCall?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_16_491754conv2d_16_491756*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_491348?
activation_16/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_491359?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0batch_normalization_12_491760batch_normalization_12_491762batch_normalization_12_491764batch_normalization_12_491766*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_490948?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_491661?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0conv2d_17_491770conv2d_17_491772*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_491387?
activation_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_491398?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall&activation_17/PartitionedCall:output:0batch_normalization_13_491776batch_normalization_13_491778batch_normalization_13_491780batch_normalization_13_491782*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_491012?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv2d_18_491785conv2d_18_491787*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_491419?
activation_18/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_491430?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0batch_normalization_14_491791batch_normalization_14_491793batch_normalization_14_491795batch_normalization_14_491797*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_491076?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_491606?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0conv2d_19_491801conv2d_19_491803*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_491458?
activation_19/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_491469?
encoded/CastCast&activation_19/PartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
encoded/StatefulPartitionedCallStatefulPartitionedCallencoded/Cast:y:0encoded_491808encoded_491810encoded_491812encoded_491814*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_encoded_layer_call_and_return_conditional_losses_491140?
conv2d_transpose_12/CastCast(encoded/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_12/Cast:y:0conv2d_transpose_12_491818conv2d_transpose_12_491820*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_491189?
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0conv2d_transpose_13_491823conv2d_transpose_13_491825*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_491234?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_13/StatefulPartitionedCall:output:0conv2d_transpose_14_491828conv2d_transpose_14_491830*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_491279?
decoded/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0decoded_491833decoded_491835*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_decoded_layer_call_and_return_conditional_losses_491324?
IdentityIdentity(decoded/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall ^decoded/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall ^encoded/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2B
decoded/StatefulPartitionedCalldecoded/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2B
encoded/StatefulPartitionedCallencoded/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_491012

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
?
?
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_490917

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
?
e
I__inference_activation_16_layer_call_and_return_conditional_losses_491359

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:???????????d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_18_layer_call_fn_492942

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
:?????????  @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_491419w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_490948

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
?!
?
O__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_493271

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
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_491076

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
E__inference_conv2d_17_layer_call_and_return_conditional_losses_491387

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
:?????????@@ *
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
:?????????@@ g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_492538

inputsB
(conv2d_16_conv2d_readvariableop_resource:7
)conv2d_16_biasadd_readvariableop_resource:<
.batch_normalization_12_readvariableop_resource:>
0batch_normalization_12_readvariableop_1_resource:M
?batch_normalization_12_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_17_conv2d_readvariableop_resource: 7
)conv2d_17_biasadd_readvariableop_resource: <
.batch_normalization_13_readvariableop_resource: >
0batch_normalization_13_readvariableop_1_resource: M
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_18_conv2d_readvariableop_resource: @7
)conv2d_18_biasadd_readvariableop_resource:@<
.batch_normalization_14_readvariableop_resource:@>
0batch_normalization_14_readvariableop_1_resource:@M
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:@B
(conv2d_19_conv2d_readvariableop_resource:@ 7
)conv2d_19_biasadd_readvariableop_resource: -
encoded_readvariableop_resource: /
!encoded_readvariableop_1_resource: >
0encoded_fusedbatchnormv3_readvariableop_resource: @
2encoded_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_12_conv2d_transpose_readvariableop_resource:@ A
3conv2d_transpose_12_biasadd_readvariableop_resource:@V
<conv2d_transpose_13_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_13_biasadd_readvariableop_resource: V
<conv2d_transpose_14_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_14_biasadd_readvariableop_resource:J
0decoded_conv2d_transpose_readvariableop_resource:5
'decoded_biasadd_readvariableop_resource:
identity??6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_12/ReadVariableOp?'batch_normalization_12/ReadVariableOp_1?6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_13/ReadVariableOp?'batch_normalization_13/ReadVariableOp_1?6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_14/ReadVariableOp?'batch_normalization_14/ReadVariableOp_1? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp? conv2d_18/BiasAdd/ReadVariableOp?conv2d_18/Conv2D/ReadVariableOp? conv2d_19/BiasAdd/ReadVariableOp?conv2d_19/Conv2D/ReadVariableOp?*conv2d_transpose_12/BiasAdd/ReadVariableOp?3conv2d_transpose_12/conv2d_transpose/ReadVariableOp?*conv2d_transpose_13/BiasAdd/ReadVariableOp?3conv2d_transpose_13/conv2d_transpose/ReadVariableOp?*conv2d_transpose_14/BiasAdd/ReadVariableOp?3conv2d_transpose_14/conv2d_transpose/ReadVariableOp?decoded/BiasAdd/ReadVariableOp?'decoded/conv2d_transpose/ReadVariableOp?'encoded/FusedBatchNormV3/ReadVariableOp?)encoded/FusedBatchNormV3/ReadVariableOp_1?encoded/ReadVariableOp?encoded/ReadVariableOp_1?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_16/Conv2DConv2Dinputs'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????r
activation_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3 activation_16/Relu:activations:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
dropout_8/IdentityIdentity+batch_normalization_12/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:????????????
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_17/Conv2DConv2Ddropout_8/Identity:output:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ p
activation_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ ?
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3 activation_17/Relu:activations:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( ?
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_18/Conv2DConv2D+batch_normalization_13/FusedBatchNormV3:y:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @p
activation_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @?
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3 activation_18/Relu:activations:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( ?
dropout_9/IdentityIdentity+batch_normalization_14/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @?
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
conv2d_19/Conv2DConv2Ddropout_9/Identity:output:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? p
activation_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 
encoded/CastCast activation_19/Relu:activations:0*

DstT0*

SrcT0*/
_output_shapes
:????????? r
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
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
conv2d_transpose_12/CastCastencoded/FusedBatchNormV3:y:0*

DstT0*

SrcT0*/
_output_shapes
:????????? e
conv2d_transpose_12/ShapeShapeconv2d_transpose_12/Cast:y:0*
T0*
_output_shapes
:q
'conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_12/strided_sliceStridedSlice"conv2d_transpose_12/Shape:output:00conv2d_transpose_12/strided_slice/stack:output:02conv2d_transpose_12/strided_slice/stack_1:output:02conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose_12/stackPack*conv2d_transpose_12/strided_slice:output:0$conv2d_transpose_12/stack/1:output:0$conv2d_transpose_12/stack/2:output:0$conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_12/strided_slice_1StridedSlice"conv2d_transpose_12/stack:output:02conv2d_transpose_12/strided_slice_1/stack:output:04conv2d_transpose_12/strided_slice_1/stack_1:output:04conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
$conv2d_transpose_12/conv2d_transposeConv2DBackpropInput"conv2d_transpose_12/stack:output:0;conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0conv2d_transpose_12/Cast:y:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
*conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_transpose_12/BiasAddBiasAdd-conv2d_transpose_12/conv2d_transpose:output:02conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
conv2d_transpose_12/ReluRelu$conv2d_transpose_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @o
conv2d_transpose_13/ShapeShape&conv2d_transpose_12/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_13/strided_sliceStridedSlice"conv2d_transpose_13/Shape:output:00conv2d_transpose_13/strided_slice/stack:output:02conv2d_transpose_13/strided_slice/stack_1:output:02conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@]
conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@]
conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_13/stackPack*conv2d_transpose_13/strided_slice:output:0$conv2d_transpose_13/stack/1:output:0$conv2d_transpose_13/stack/2:output:0$conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_13/strided_slice_1StridedSlice"conv2d_transpose_13/stack:output:02conv2d_transpose_13/strided_slice_1/stack:output:04conv2d_transpose_13/strided_slice_1/stack_1:output:04conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
$conv2d_transpose_13/conv2d_transposeConv2DBackpropInput"conv2d_transpose_13/stack:output:0;conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_12/Relu:activations:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
*conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_13/BiasAddBiasAdd-conv2d_transpose_13/conv2d_transpose:output:02conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
conv2d_transpose_13/ReluRelu$conv2d_transpose_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ o
conv2d_transpose_14/ShapeShape&conv2d_transpose_13/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_14/strided_sliceStridedSlice"conv2d_transpose_14/Shape:output:00conv2d_transpose_14/strided_slice/stack:output:02conv2d_transpose_14/strided_slice/stack_1:output:02conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?^
conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?]
conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_14/stackPack*conv2d_transpose_14/strided_slice:output:0$conv2d_transpose_14/stack/1:output:0$conv2d_transpose_14/stack/2:output:0$conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_14/strided_slice_1StridedSlice"conv2d_transpose_14/stack:output:02conv2d_transpose_14/strided_slice_1/stack:output:04conv2d_transpose_14/strided_slice_1/stack_1:output:04conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
$conv2d_transpose_14/conv2d_transposeConv2DBackpropInput"conv2d_transpose_14/stack:output:0;conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_13/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
*conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_14/BiasAddBiasAdd-conv2d_transpose_14/conv2d_transpose:output:02conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
conv2d_transpose_14/ReluRelu$conv2d_transpose_14/BiasAdd:output:0*
T0*1
_output_shapes
:???????????c
decoded/ShapeShape&conv2d_transpose_14/Relu:activations:0*
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
B :?R
decoded/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?Q
decoded/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
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
:*
dtype0?
decoded/conv2d_transposeConv2DBackpropInputdecoded/stack:output:0/decoded/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_14/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
decoded/BiasAdd/ReadVariableOpReadVariableOp'decoded_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
decoded/BiasAddBiasAdd!decoded/conv2d_transpose:output:0&decoded/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????p
decoded/SigmoidSigmoiddecoded/BiasAdd:output:0*
T0*1
_output_shapes
:???????????l
IdentityIdentitydecoded/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp7^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_1!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp+^conv2d_transpose_12/BiasAdd/ReadVariableOp4^conv2d_transpose_12/conv2d_transpose/ReadVariableOp+^conv2d_transpose_13/BiasAdd/ReadVariableOp4^conv2d_transpose_13/conv2d_transpose/ReadVariableOp+^conv2d_transpose_14/BiasAdd/ReadVariableOp4^conv2d_transpose_14/conv2d_transpose/ReadVariableOp^decoded/BiasAdd/ReadVariableOp(^decoded/conv2d_transpose/ReadVariableOp(^encoded/FusedBatchNormV3/ReadVariableOp*^encoded/FusedBatchNormV3/ReadVariableOp_1^encoded/ReadVariableOp^encoded/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2X
*conv2d_transpose_12/BiasAdd/ReadVariableOp*conv2d_transpose_12/BiasAdd/ReadVariableOp2j
3conv2d_transpose_12/conv2d_transpose/ReadVariableOp3conv2d_transpose_12/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_13/BiasAdd/ReadVariableOp*conv2d_transpose_13/BiasAdd/ReadVariableOp2j
3conv2d_transpose_13/conv2d_transpose/ReadVariableOp3conv2d_transpose_13/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_14/BiasAdd/ReadVariableOp*conv2d_transpose_14/BiasAdd/ReadVariableOp2j
3conv2d_transpose_14/conv2d_transpose/ReadVariableOp3conv2d_transpose_14/conv2d_transpose/ReadVariableOp2@
decoded/BiasAdd/ReadVariableOpdecoded/BiasAdd/ReadVariableOp2R
'decoded/conv2d_transpose/ReadVariableOp'decoded/conv2d_transpose/ReadVariableOp2R
'encoded/FusedBatchNormV3/ReadVariableOp'encoded/FusedBatchNormV3/ReadVariableOp2V
)encoded/FusedBatchNormV3/ReadVariableOp_1)encoded/FusedBatchNormV3/ReadVariableOp_120
encoded/ReadVariableOpencoded/ReadVariableOp24
encoded/ReadVariableOp_1encoded/ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
I__inference_activation_19_layer_call_and_return_conditional_losses_491469

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:????????? b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
J
.__inference_activation_17_layer_call_fn_492866

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
:?????????@@ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_491398h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@ :W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_12_layer_call_fn_493151

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
GPU2 *0J 8? *X
fSRQ
O__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_491189?
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
?!
?
C__inference_decoded_layer_call_and_return_conditional_losses_493314

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
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
value	B :y
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
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
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
?
?
(__inference_decoded_layer_call_fn_493280

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_decoded_layer_call_and_return_conditional_losses_491324?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
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
?
?
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_492797

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
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_492933

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
?
?
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_490981

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
?	
?
7__inference_batch_normalization_14_layer_call_fn_492975

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
GPU2 *0J 8? *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_491045?
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
?
e
I__inference_activation_16_layer_call_and_return_conditional_losses_492753

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:???????????d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_17_layer_call_and_return_conditional_losses_492861

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
:?????????@@ *
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
:?????????@@ g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_13_layer_call_fn_492884

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
GPU2 *0J 8? *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_490981?
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
?
?
4__inference_conv2d_transpose_13_layer_call_fn_493194

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
GPU2 *0J 8? *X
fSRQ
O__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_491234?
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
??
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_492724

inputsB
(conv2d_16_conv2d_readvariableop_resource:7
)conv2d_16_biasadd_readvariableop_resource:<
.batch_normalization_12_readvariableop_resource:>
0batch_normalization_12_readvariableop_1_resource:M
?batch_normalization_12_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_17_conv2d_readvariableop_resource: 7
)conv2d_17_biasadd_readvariableop_resource: <
.batch_normalization_13_readvariableop_resource: >
0batch_normalization_13_readvariableop_1_resource: M
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_18_conv2d_readvariableop_resource: @7
)conv2d_18_biasadd_readvariableop_resource:@<
.batch_normalization_14_readvariableop_resource:@>
0batch_normalization_14_readvariableop_1_resource:@M
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:@B
(conv2d_19_conv2d_readvariableop_resource:@ 7
)conv2d_19_biasadd_readvariableop_resource: -
encoded_readvariableop_resource: /
!encoded_readvariableop_1_resource: >
0encoded_fusedbatchnormv3_readvariableop_resource: @
2encoded_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_12_conv2d_transpose_readvariableop_resource:@ A
3conv2d_transpose_12_biasadd_readvariableop_resource:@V
<conv2d_transpose_13_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_13_biasadd_readvariableop_resource: V
<conv2d_transpose_14_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_14_biasadd_readvariableop_resource:J
0decoded_conv2d_transpose_readvariableop_resource:5
'decoded_biasadd_readvariableop_resource:
identity??%batch_normalization_12/AssignNewValue?'batch_normalization_12/AssignNewValue_1?6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_12/ReadVariableOp?'batch_normalization_12/ReadVariableOp_1?%batch_normalization_13/AssignNewValue?'batch_normalization_13/AssignNewValue_1?6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_13/ReadVariableOp?'batch_normalization_13/ReadVariableOp_1?%batch_normalization_14/AssignNewValue?'batch_normalization_14/AssignNewValue_1?6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_14/ReadVariableOp?'batch_normalization_14/ReadVariableOp_1? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp? conv2d_18/BiasAdd/ReadVariableOp?conv2d_18/Conv2D/ReadVariableOp? conv2d_19/BiasAdd/ReadVariableOp?conv2d_19/Conv2D/ReadVariableOp?*conv2d_transpose_12/BiasAdd/ReadVariableOp?3conv2d_transpose_12/conv2d_transpose/ReadVariableOp?*conv2d_transpose_13/BiasAdd/ReadVariableOp?3conv2d_transpose_13/conv2d_transpose/ReadVariableOp?*conv2d_transpose_14/BiasAdd/ReadVariableOp?3conv2d_transpose_14/conv2d_transpose/ReadVariableOp?decoded/BiasAdd/ReadVariableOp?'decoded/conv2d_transpose/ReadVariableOp?encoded/AssignNewValue?encoded/AssignNewValue_1?'encoded/FusedBatchNormV3/ReadVariableOp?)encoded/FusedBatchNormV3/ReadVariableOp_1?encoded/ReadVariableOp?encoded/ReadVariableOp_1?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_16/Conv2DConv2Dinputs'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????r
activation_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3 activation_16/Relu:activations:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_12/AssignNewValueAssignVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource4batch_normalization_12/FusedBatchNormV3:batch_mean:07^batch_normalization_12/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_12/AssignNewValue_1AssignVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_12/FusedBatchNormV3:batch_variance:09^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(\
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_8/dropout/MulMul+batch_normalization_12/FusedBatchNormV3:y:0 dropout_8/dropout/Const:output:0*
T0*1
_output_shapes
:???????????r
dropout_8/dropout/ShapeShape+batch_normalization_12/FusedBatchNormV3:y:0*
T0*
_output_shapes
:?
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype0e
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:????????????
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:????????????
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*1
_output_shapes
:????????????
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_17/Conv2DConv2Ddropout_8/dropout/Mul_1:z:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ p
activation_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ ?
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3 activation_17/Relu:activations:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_13/AssignNewValueAssignVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource4batch_normalization_13/FusedBatchNormV3:batch_mean:07^batch_normalization_13/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_13/AssignNewValue_1AssignVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_13/FusedBatchNormV3:batch_variance:09^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_18/Conv2DConv2D+batch_normalization_13/FusedBatchNormV3:y:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @p
activation_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @?
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3 activation_18/Relu:activations:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_14/AssignNewValueAssignVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource4batch_normalization_14/FusedBatchNormV3:batch_mean:07^batch_normalization_14/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_14/AssignNewValue_1AssignVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_14/FusedBatchNormV3:batch_variance:09^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(\
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_9/dropout/MulMul+batch_normalization_14/FusedBatchNormV3:y:0 dropout_9/dropout/Const:output:0*
T0*/
_output_shapes
:?????????  @r
dropout_9/dropout/ShapeShape+batch_normalization_14/FusedBatchNormV3:y:0*
T0*
_output_shapes
:?
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????  @*
dtype0e
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  @?
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  @?
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  @?
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
conv2d_19/Conv2DConv2Ddropout_9/dropout/Mul_1:z:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? p
activation_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 
encoded/CastCast activation_19/Relu:activations:0*

DstT0*

SrcT0*/
_output_shapes
:????????? r
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
7:????????? : : : : :*
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
conv2d_transpose_12/CastCastencoded/FusedBatchNormV3:y:0*

DstT0*

SrcT0*/
_output_shapes
:????????? e
conv2d_transpose_12/ShapeShapeconv2d_transpose_12/Cast:y:0*
T0*
_output_shapes
:q
'conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_12/strided_sliceStridedSlice"conv2d_transpose_12/Shape:output:00conv2d_transpose_12/strided_slice/stack:output:02conv2d_transpose_12/strided_slice/stack_1:output:02conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose_12/stackPack*conv2d_transpose_12/strided_slice:output:0$conv2d_transpose_12/stack/1:output:0$conv2d_transpose_12/stack/2:output:0$conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_12/strided_slice_1StridedSlice"conv2d_transpose_12/stack:output:02conv2d_transpose_12/strided_slice_1/stack:output:04conv2d_transpose_12/strided_slice_1/stack_1:output:04conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
$conv2d_transpose_12/conv2d_transposeConv2DBackpropInput"conv2d_transpose_12/stack:output:0;conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0conv2d_transpose_12/Cast:y:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
*conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_transpose_12/BiasAddBiasAdd-conv2d_transpose_12/conv2d_transpose:output:02conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
conv2d_transpose_12/ReluRelu$conv2d_transpose_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @o
conv2d_transpose_13/ShapeShape&conv2d_transpose_12/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_13/strided_sliceStridedSlice"conv2d_transpose_13/Shape:output:00conv2d_transpose_13/strided_slice/stack:output:02conv2d_transpose_13/strided_slice/stack_1:output:02conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@]
conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@]
conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_13/stackPack*conv2d_transpose_13/strided_slice:output:0$conv2d_transpose_13/stack/1:output:0$conv2d_transpose_13/stack/2:output:0$conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_13/strided_slice_1StridedSlice"conv2d_transpose_13/stack:output:02conv2d_transpose_13/strided_slice_1/stack:output:04conv2d_transpose_13/strided_slice_1/stack_1:output:04conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
$conv2d_transpose_13/conv2d_transposeConv2DBackpropInput"conv2d_transpose_13/stack:output:0;conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_12/Relu:activations:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
*conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_13/BiasAddBiasAdd-conv2d_transpose_13/conv2d_transpose:output:02conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
conv2d_transpose_13/ReluRelu$conv2d_transpose_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ o
conv2d_transpose_14/ShapeShape&conv2d_transpose_13/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_14/strided_sliceStridedSlice"conv2d_transpose_14/Shape:output:00conv2d_transpose_14/strided_slice/stack:output:02conv2d_transpose_14/strided_slice/stack_1:output:02conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?^
conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?]
conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_14/stackPack*conv2d_transpose_14/strided_slice:output:0$conv2d_transpose_14/stack/1:output:0$conv2d_transpose_14/stack/2:output:0$conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_14/strided_slice_1StridedSlice"conv2d_transpose_14/stack:output:02conv2d_transpose_14/strided_slice_1/stack:output:04conv2d_transpose_14/strided_slice_1/stack_1:output:04conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
$conv2d_transpose_14/conv2d_transposeConv2DBackpropInput"conv2d_transpose_14/stack:output:0;conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_13/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
*conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_14/BiasAddBiasAdd-conv2d_transpose_14/conv2d_transpose:output:02conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
conv2d_transpose_14/ReluRelu$conv2d_transpose_14/BiasAdd:output:0*
T0*1
_output_shapes
:???????????c
decoded/ShapeShape&conv2d_transpose_14/Relu:activations:0*
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
B :?R
decoded/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?Q
decoded/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
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
:*
dtype0?
decoded/conv2d_transposeConv2DBackpropInputdecoded/stack:output:0/decoded/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_14/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
decoded/BiasAdd/ReadVariableOpReadVariableOp'decoded_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
decoded/BiasAddBiasAdd!decoded/conv2d_transpose:output:0&decoded/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????p
decoded/SigmoidSigmoiddecoded/BiasAdd:output:0*
T0*1
_output_shapes
:???????????l
IdentityIdentitydecoded/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp&^batch_normalization_12/AssignNewValue(^batch_normalization_12/AssignNewValue_17^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_1&^batch_normalization_13/AssignNewValue(^batch_normalization_13/AssignNewValue_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_1&^batch_normalization_14/AssignNewValue(^batch_normalization_14/AssignNewValue_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_1!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp+^conv2d_transpose_12/BiasAdd/ReadVariableOp4^conv2d_transpose_12/conv2d_transpose/ReadVariableOp+^conv2d_transpose_13/BiasAdd/ReadVariableOp4^conv2d_transpose_13/conv2d_transpose/ReadVariableOp+^conv2d_transpose_14/BiasAdd/ReadVariableOp4^conv2d_transpose_14/conv2d_transpose/ReadVariableOp^decoded/BiasAdd/ReadVariableOp(^decoded/conv2d_transpose/ReadVariableOp^encoded/AssignNewValue^encoded/AssignNewValue_1(^encoded/FusedBatchNormV3/ReadVariableOp*^encoded/FusedBatchNormV3/ReadVariableOp_1^encoded/ReadVariableOp^encoded/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_12/AssignNewValue%batch_normalization_12/AssignNewValue2R
'batch_normalization_12/AssignNewValue_1'batch_normalization_12/AssignNewValue_12p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12N
%batch_normalization_13/AssignNewValue%batch_normalization_13/AssignNewValue2R
'batch_normalization_13/AssignNewValue_1'batch_normalization_13/AssignNewValue_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12N
%batch_normalization_14/AssignNewValue%batch_normalization_14/AssignNewValue2R
'batch_normalization_14/AssignNewValue_1'batch_normalization_14/AssignNewValue_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2X
*conv2d_transpose_12/BiasAdd/ReadVariableOp*conv2d_transpose_12/BiasAdd/ReadVariableOp2j
3conv2d_transpose_12/conv2d_transpose/ReadVariableOp3conv2d_transpose_12/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_13/BiasAdd/ReadVariableOp*conv2d_transpose_13/BiasAdd/ReadVariableOp2j
3conv2d_transpose_13/conv2d_transpose/ReadVariableOp3conv2d_transpose_13/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_14/BiasAdd/ReadVariableOp*conv2d_transpose_14/BiasAdd/ReadVariableOp2j
3conv2d_transpose_14/conv2d_transpose/ReadVariableOp3conv2d_transpose_14/conv2d_transpose/ReadVariableOp2@
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
:???????????
 
_user_specified_nameinputs
??
?"
!__inference__wrapped_model_490895
conv2d_16_inputO
5sequential_4_conv2d_16_conv2d_readvariableop_resource:D
6sequential_4_conv2d_16_biasadd_readvariableop_resource:I
;sequential_4_batch_normalization_12_readvariableop_resource:K
=sequential_4_batch_normalization_12_readvariableop_1_resource:Z
Lsequential_4_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:\
Nsequential_4_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_17_conv2d_readvariableop_resource: D
6sequential_4_conv2d_17_biasadd_readvariableop_resource: I
;sequential_4_batch_normalization_13_readvariableop_resource: K
=sequential_4_batch_normalization_13_readvariableop_1_resource: Z
Lsequential_4_batch_normalization_13_fusedbatchnormv3_readvariableop_resource: \
Nsequential_4_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource: O
5sequential_4_conv2d_18_conv2d_readvariableop_resource: @D
6sequential_4_conv2d_18_biasadd_readvariableop_resource:@I
;sequential_4_batch_normalization_14_readvariableop_resource:@K
=sequential_4_batch_normalization_14_readvariableop_1_resource:@Z
Lsequential_4_batch_normalization_14_fusedbatchnormv3_readvariableop_resource:@\
Nsequential_4_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:@O
5sequential_4_conv2d_19_conv2d_readvariableop_resource:@ D
6sequential_4_conv2d_19_biasadd_readvariableop_resource: :
,sequential_4_encoded_readvariableop_resource: <
.sequential_4_encoded_readvariableop_1_resource: K
=sequential_4_encoded_fusedbatchnormv3_readvariableop_resource: M
?sequential_4_encoded_fusedbatchnormv3_readvariableop_1_resource: c
Isequential_4_conv2d_transpose_12_conv2d_transpose_readvariableop_resource:@ N
@sequential_4_conv2d_transpose_12_biasadd_readvariableop_resource:@c
Isequential_4_conv2d_transpose_13_conv2d_transpose_readvariableop_resource: @N
@sequential_4_conv2d_transpose_13_biasadd_readvariableop_resource: c
Isequential_4_conv2d_transpose_14_conv2d_transpose_readvariableop_resource: N
@sequential_4_conv2d_transpose_14_biasadd_readvariableop_resource:W
=sequential_4_decoded_conv2d_transpose_readvariableop_resource:B
4sequential_4_decoded_biasadd_readvariableop_resource:
identity??Csequential_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp?Esequential_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?2sequential_4/batch_normalization_12/ReadVariableOp?4sequential_4/batch_normalization_12/ReadVariableOp_1?Csequential_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?Esequential_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?2sequential_4/batch_normalization_13/ReadVariableOp?4sequential_4/batch_normalization_13/ReadVariableOp_1?Csequential_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp?Esequential_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?2sequential_4/batch_normalization_14/ReadVariableOp?4sequential_4/batch_normalization_14/ReadVariableOp_1?-sequential_4/conv2d_16/BiasAdd/ReadVariableOp?,sequential_4/conv2d_16/Conv2D/ReadVariableOp?-sequential_4/conv2d_17/BiasAdd/ReadVariableOp?,sequential_4/conv2d_17/Conv2D/ReadVariableOp?-sequential_4/conv2d_18/BiasAdd/ReadVariableOp?,sequential_4/conv2d_18/Conv2D/ReadVariableOp?-sequential_4/conv2d_19/BiasAdd/ReadVariableOp?,sequential_4/conv2d_19/Conv2D/ReadVariableOp?7sequential_4/conv2d_transpose_12/BiasAdd/ReadVariableOp?@sequential_4/conv2d_transpose_12/conv2d_transpose/ReadVariableOp?7sequential_4/conv2d_transpose_13/BiasAdd/ReadVariableOp?@sequential_4/conv2d_transpose_13/conv2d_transpose/ReadVariableOp?7sequential_4/conv2d_transpose_14/BiasAdd/ReadVariableOp?@sequential_4/conv2d_transpose_14/conv2d_transpose/ReadVariableOp?+sequential_4/decoded/BiasAdd/ReadVariableOp?4sequential_4/decoded/conv2d_transpose/ReadVariableOp?4sequential_4/encoded/FusedBatchNormV3/ReadVariableOp?6sequential_4/encoded/FusedBatchNormV3/ReadVariableOp_1?#sequential_4/encoded/ReadVariableOp?%sequential_4/encoded/ReadVariableOp_1?
,sequential_4/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_4/conv2d_16/Conv2DConv2Dconv2d_16_input4sequential_4/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
-sequential_4/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_4/conv2d_16/BiasAddBiasAdd&sequential_4/conv2d_16/Conv2D:output:05sequential_4/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
sequential_4/activation_16/ReluRelu'sequential_4/conv2d_16/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
2sequential_4/batch_normalization_12/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_12_readvariableop_resource*
_output_shapes
:*
dtype0?
4sequential_4/batch_normalization_12/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_12_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Csequential_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Esequential_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
4sequential_4/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3-sequential_4/activation_16/Relu:activations:0:sequential_4/batch_normalization_12/ReadVariableOp:value:0<sequential_4/batch_normalization_12/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
sequential_4/dropout_8/IdentityIdentity8sequential_4/batch_normalization_12/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:????????????
,sequential_4/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
sequential_4/conv2d_17/Conv2DConv2D(sequential_4/dropout_8/Identity:output:04sequential_4/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
-sequential_4/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_4/conv2d_17/BiasAddBiasAdd&sequential_4/conv2d_17/Conv2D:output:05sequential_4/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
sequential_4/activation_17/ReluRelu'sequential_4/conv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ ?
2sequential_4/batch_normalization_13/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_13_readvariableop_resource*
_output_shapes
: *
dtype0?
4sequential_4/batch_normalization_13/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_13_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Csequential_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Esequential_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
4sequential_4/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3-sequential_4/activation_17/Relu:activations:0:sequential_4/batch_normalization_13/ReadVariableOp:value:0<sequential_4/batch_normalization_13/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( ?
,sequential_4/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
sequential_4/conv2d_18/Conv2DConv2D8sequential_4/batch_normalization_13/FusedBatchNormV3:y:04sequential_4/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
-sequential_4/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_4/conv2d_18/BiasAddBiasAdd&sequential_4/conv2d_18/Conv2D:output:05sequential_4/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
sequential_4/activation_18/ReluRelu'sequential_4/conv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @?
2sequential_4/batch_normalization_14/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_14_readvariableop_resource*
_output_shapes
:@*
dtype0?
4sequential_4/batch_normalization_14/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_14_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Csequential_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Esequential_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
4sequential_4/batch_normalization_14/FusedBatchNormV3FusedBatchNormV3-sequential_4/activation_18/Relu:activations:0:sequential_4/batch_normalization_14/ReadVariableOp:value:0<sequential_4/batch_normalization_14/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( ?
sequential_4/dropout_9/IdentityIdentity8sequential_4/batch_normalization_14/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @?
,sequential_4/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
sequential_4/conv2d_19/Conv2DConv2D(sequential_4/dropout_9/Identity:output:04sequential_4/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
-sequential_4/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_4/conv2d_19/BiasAddBiasAdd&sequential_4/conv2d_19/Conv2D:output:05sequential_4/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
sequential_4/activation_19/ReluRelu'sequential_4/conv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
sequential_4/encoded/CastCast-sequential_4/activation_19/Relu:activations:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
#sequential_4/encoded/ReadVariableOpReadVariableOp,sequential_4_encoded_readvariableop_resource*
_output_shapes
: *
dtype0?
%sequential_4/encoded/ReadVariableOp_1ReadVariableOp.sequential_4_encoded_readvariableop_1_resource*
_output_shapes
: *
dtype0?
4sequential_4/encoded/FusedBatchNormV3/ReadVariableOpReadVariableOp=sequential_4_encoded_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
6sequential_4/encoded/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp?sequential_4_encoded_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
%sequential_4/encoded/FusedBatchNormV3FusedBatchNormV3sequential_4/encoded/Cast:y:0+sequential_4/encoded/ReadVariableOp:value:0-sequential_4/encoded/ReadVariableOp_1:value:0<sequential_4/encoded/FusedBatchNormV3/ReadVariableOp:value:0>sequential_4/encoded/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
%sequential_4/conv2d_transpose_12/CastCast)sequential_4/encoded/FusedBatchNormV3:y:0*

DstT0*

SrcT0*/
_output_shapes
:????????? 
&sequential_4/conv2d_transpose_12/ShapeShape)sequential_4/conv2d_transpose_12/Cast:y:0*
T0*
_output_shapes
:~
4sequential_4/conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6sequential_4/conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential_4/conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.sequential_4/conv2d_transpose_12/strided_sliceStridedSlice/sequential_4/conv2d_transpose_12/Shape:output:0=sequential_4/conv2d_transpose_12/strided_slice/stack:output:0?sequential_4/conv2d_transpose_12/strided_slice/stack_1:output:0?sequential_4/conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_4/conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B : j
(sequential_4/conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B : j
(sequential_4/conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
&sequential_4/conv2d_transpose_12/stackPack7sequential_4/conv2d_transpose_12/strided_slice:output:01sequential_4/conv2d_transpose_12/stack/1:output:01sequential_4/conv2d_transpose_12/stack/2:output:01sequential_4/conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:?
6sequential_4/conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8sequential_4/conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_4/conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0sequential_4/conv2d_transpose_12/strided_slice_1StridedSlice/sequential_4/conv2d_transpose_12/stack:output:0?sequential_4/conv2d_transpose_12/strided_slice_1/stack:output:0Asequential_4/conv2d_transpose_12/strided_slice_1/stack_1:output:0Asequential_4/conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_4/conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_4_conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
1sequential_4/conv2d_transpose_12/conv2d_transposeConv2DBackpropInput/sequential_4/conv2d_transpose_12/stack:output:0Hsequential_4/conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0)sequential_4/conv2d_transpose_12/Cast:y:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
7sequential_4/conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOp@sequential_4_conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
(sequential_4/conv2d_transpose_12/BiasAddBiasAdd:sequential_4/conv2d_transpose_12/conv2d_transpose:output:0?sequential_4/conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
%sequential_4/conv2d_transpose_12/ReluRelu1sequential_4/conv2d_transpose_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @?
&sequential_4/conv2d_transpose_13/ShapeShape3sequential_4/conv2d_transpose_12/Relu:activations:0*
T0*
_output_shapes
:~
4sequential_4/conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6sequential_4/conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential_4/conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.sequential_4/conv2d_transpose_13/strided_sliceStridedSlice/sequential_4/conv2d_transpose_13/Shape:output:0=sequential_4/conv2d_transpose_13/strided_slice/stack:output:0?sequential_4/conv2d_transpose_13/strided_slice/stack_1:output:0?sequential_4/conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_4/conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@j
(sequential_4/conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@j
(sequential_4/conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_4/conv2d_transpose_13/stackPack7sequential_4/conv2d_transpose_13/strided_slice:output:01sequential_4/conv2d_transpose_13/stack/1:output:01sequential_4/conv2d_transpose_13/stack/2:output:01sequential_4/conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:?
6sequential_4/conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8sequential_4/conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_4/conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0sequential_4/conv2d_transpose_13/strided_slice_1StridedSlice/sequential_4/conv2d_transpose_13/stack:output:0?sequential_4/conv2d_transpose_13/strided_slice_1/stack:output:0Asequential_4/conv2d_transpose_13/strided_slice_1/stack_1:output:0Asequential_4/conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_4/conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_4_conv2d_transpose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
1sequential_4/conv2d_transpose_13/conv2d_transposeConv2DBackpropInput/sequential_4/conv2d_transpose_13/stack:output:0Hsequential_4/conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:03sequential_4/conv2d_transpose_12/Relu:activations:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
7sequential_4/conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOp@sequential_4_conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
(sequential_4/conv2d_transpose_13/BiasAddBiasAdd:sequential_4/conv2d_transpose_13/conv2d_transpose:output:0?sequential_4/conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
%sequential_4/conv2d_transpose_13/ReluRelu1sequential_4/conv2d_transpose_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ ?
&sequential_4/conv2d_transpose_14/ShapeShape3sequential_4/conv2d_transpose_13/Relu:activations:0*
T0*
_output_shapes
:~
4sequential_4/conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6sequential_4/conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential_4/conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.sequential_4/conv2d_transpose_14/strided_sliceStridedSlice/sequential_4/conv2d_transpose_14/Shape:output:0=sequential_4/conv2d_transpose_14/strided_slice/stack:output:0?sequential_4/conv2d_transpose_14/strided_slice/stack_1:output:0?sequential_4/conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
(sequential_4/conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?k
(sequential_4/conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?j
(sequential_4/conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
&sequential_4/conv2d_transpose_14/stackPack7sequential_4/conv2d_transpose_14/strided_slice:output:01sequential_4/conv2d_transpose_14/stack/1:output:01sequential_4/conv2d_transpose_14/stack/2:output:01sequential_4/conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:?
6sequential_4/conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8sequential_4/conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_4/conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0sequential_4/conv2d_transpose_14/strided_slice_1StridedSlice/sequential_4/conv2d_transpose_14/stack:output:0?sequential_4/conv2d_transpose_14/strided_slice_1/stack:output:0Asequential_4/conv2d_transpose_14/strided_slice_1/stack_1:output:0Asequential_4/conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_4/conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_4_conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
1sequential_4/conv2d_transpose_14/conv2d_transposeConv2DBackpropInput/sequential_4/conv2d_transpose_14/stack:output:0Hsequential_4/conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:03sequential_4/conv2d_transpose_13/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
7sequential_4/conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOp@sequential_4_conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
(sequential_4/conv2d_transpose_14/BiasAddBiasAdd:sequential_4/conv2d_transpose_14/conv2d_transpose:output:0?sequential_4/conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
%sequential_4/conv2d_transpose_14/ReluRelu1sequential_4/conv2d_transpose_14/BiasAdd:output:0*
T0*1
_output_shapes
:???????????}
sequential_4/decoded/ShapeShape3sequential_4/conv2d_transpose_14/Relu:activations:0*
T0*
_output_shapes
:r
(sequential_4/decoded/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_4/decoded/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_4/decoded/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"sequential_4/decoded/strided_sliceStridedSlice#sequential_4/decoded/Shape:output:01sequential_4/decoded/strided_slice/stack:output:03sequential_4/decoded/strided_slice/stack_1:output:03sequential_4/decoded/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
sequential_4/decoded/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?_
sequential_4/decoded/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?^
sequential_4/decoded/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
sequential_4/decoded/stackPack+sequential_4/decoded/strided_slice:output:0%sequential_4/decoded/stack/1:output:0%sequential_4/decoded/stack/2:output:0%sequential_4/decoded/stack/3:output:0*
N*
T0*
_output_shapes
:t
*sequential_4/decoded/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_4/decoded/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_4/decoded/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$sequential_4/decoded/strided_slice_1StridedSlice#sequential_4/decoded/stack:output:03sequential_4/decoded/strided_slice_1/stack:output:05sequential_4/decoded/strided_slice_1/stack_1:output:05sequential_4/decoded/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
4sequential_4/decoded/conv2d_transpose/ReadVariableOpReadVariableOp=sequential_4_decoded_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
%sequential_4/decoded/conv2d_transposeConv2DBackpropInput#sequential_4/decoded/stack:output:0<sequential_4/decoded/conv2d_transpose/ReadVariableOp:value:03sequential_4/conv2d_transpose_14/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
+sequential_4/decoded/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_decoded_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_4/decoded/BiasAddBiasAdd.sequential_4/decoded/conv2d_transpose:output:03sequential_4/decoded/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
sequential_4/decoded/SigmoidSigmoid%sequential_4/decoded/BiasAdd:output:0*
T0*1
_output_shapes
:???????????y
IdentityIdentity sequential_4/decoded/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOpD^sequential_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_12/ReadVariableOp5^sequential_4/batch_normalization_12/ReadVariableOp_1D^sequential_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_13/ReadVariableOp5^sequential_4/batch_normalization_13/ReadVariableOp_1D^sequential_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_14/ReadVariableOp5^sequential_4/batch_normalization_14/ReadVariableOp_1.^sequential_4/conv2d_16/BiasAdd/ReadVariableOp-^sequential_4/conv2d_16/Conv2D/ReadVariableOp.^sequential_4/conv2d_17/BiasAdd/ReadVariableOp-^sequential_4/conv2d_17/Conv2D/ReadVariableOp.^sequential_4/conv2d_18/BiasAdd/ReadVariableOp-^sequential_4/conv2d_18/Conv2D/ReadVariableOp.^sequential_4/conv2d_19/BiasAdd/ReadVariableOp-^sequential_4/conv2d_19/Conv2D/ReadVariableOp8^sequential_4/conv2d_transpose_12/BiasAdd/ReadVariableOpA^sequential_4/conv2d_transpose_12/conv2d_transpose/ReadVariableOp8^sequential_4/conv2d_transpose_13/BiasAdd/ReadVariableOpA^sequential_4/conv2d_transpose_13/conv2d_transpose/ReadVariableOp8^sequential_4/conv2d_transpose_14/BiasAdd/ReadVariableOpA^sequential_4/conv2d_transpose_14/conv2d_transpose/ReadVariableOp,^sequential_4/decoded/BiasAdd/ReadVariableOp5^sequential_4/decoded/conv2d_transpose/ReadVariableOp5^sequential_4/encoded/FusedBatchNormV3/ReadVariableOp7^sequential_4/encoded/FusedBatchNormV3/ReadVariableOp_1$^sequential_4/encoded/ReadVariableOp&^sequential_4/encoded/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2?
Csequential_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2?
Esequential_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_12/ReadVariableOp2sequential_4/batch_normalization_12/ReadVariableOp2l
4sequential_4/batch_normalization_12/ReadVariableOp_14sequential_4/batch_normalization_12/ReadVariableOp_12?
Csequential_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2?
Esequential_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_13/ReadVariableOp2sequential_4/batch_normalization_13/ReadVariableOp2l
4sequential_4/batch_normalization_13/ReadVariableOp_14sequential_4/batch_normalization_13/ReadVariableOp_12?
Csequential_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp2?
Esequential_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_14/ReadVariableOp2sequential_4/batch_normalization_14/ReadVariableOp2l
4sequential_4/batch_normalization_14/ReadVariableOp_14sequential_4/batch_normalization_14/ReadVariableOp_12^
-sequential_4/conv2d_16/BiasAdd/ReadVariableOp-sequential_4/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_16/Conv2D/ReadVariableOp,sequential_4/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_17/BiasAdd/ReadVariableOp-sequential_4/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_17/Conv2D/ReadVariableOp,sequential_4/conv2d_17/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_18/BiasAdd/ReadVariableOp-sequential_4/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_18/Conv2D/ReadVariableOp,sequential_4/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_19/BiasAdd/ReadVariableOp-sequential_4/conv2d_19/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_19/Conv2D/ReadVariableOp,sequential_4/conv2d_19/Conv2D/ReadVariableOp2r
7sequential_4/conv2d_transpose_12/BiasAdd/ReadVariableOp7sequential_4/conv2d_transpose_12/BiasAdd/ReadVariableOp2?
@sequential_4/conv2d_transpose_12/conv2d_transpose/ReadVariableOp@sequential_4/conv2d_transpose_12/conv2d_transpose/ReadVariableOp2r
7sequential_4/conv2d_transpose_13/BiasAdd/ReadVariableOp7sequential_4/conv2d_transpose_13/BiasAdd/ReadVariableOp2?
@sequential_4/conv2d_transpose_13/conv2d_transpose/ReadVariableOp@sequential_4/conv2d_transpose_13/conv2d_transpose/ReadVariableOp2r
7sequential_4/conv2d_transpose_14/BiasAdd/ReadVariableOp7sequential_4/conv2d_transpose_14/BiasAdd/ReadVariableOp2?
@sequential_4/conv2d_transpose_14/conv2d_transpose/ReadVariableOp@sequential_4/conv2d_transpose_14/conv2d_transpose/ReadVariableOp2Z
+sequential_4/decoded/BiasAdd/ReadVariableOp+sequential_4/decoded/BiasAdd/ReadVariableOp2l
4sequential_4/decoded/conv2d_transpose/ReadVariableOp4sequential_4/decoded/conv2d_transpose/ReadVariableOp2l
4sequential_4/encoded/FusedBatchNormV3/ReadVariableOp4sequential_4/encoded/FusedBatchNormV3/ReadVariableOp2p
6sequential_4/encoded/FusedBatchNormV3/ReadVariableOp_16sequential_4/encoded/FusedBatchNormV3/ReadVariableOp_12J
#sequential_4/encoded/ReadVariableOp#sequential_4/encoded/ReadVariableOp2N
%sequential_4/encoded/ReadVariableOp_1%sequential_4/encoded/ReadVariableOp_1:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_16_input"?	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
U
conv2d_16_inputB
!serving_default_conv2d_16_input:0???????????E
decoded:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
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
?trace_32?
-__inference_sequential_4_layer_call_fn_491570
-__inference_sequential_4_layer_call_fn_492297
-__inference_sequential_4_layer_call_fn_492366
-__inference_sequential_4_layer_call_fn_491975?
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
H__inference_sequential_4_layer_call_and_return_conditional_losses_492538
H__inference_sequential_4_layer_call_and_return_conditional_losses_492724
H__inference_sequential_4_layer_call_and_return_conditional_losses_492063
H__inference_sequential_4_layer_call_and_return_conditional_losses_492151?
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
!__inference__wrapped_model_490895conv2d_16_input"?
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
*__inference_conv2d_16_layer_call_fn_492733?
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
E__inference_conv2d_16_layer_call_and_return_conditional_losses_492743?
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
*:(2conv2d_16/kernel
:2conv2d_16/bias
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
.__inference_activation_16_layer_call_fn_492748?
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
I__inference_activation_16_layer_call_and_return_conditional_losses_492753?
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
7__inference_batch_normalization_12_layer_call_fn_492766
7__inference_batch_normalization_12_layer_call_fn_492779?
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
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_492797
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_492815?
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
*:(2batch_normalization_12/gamma
):'2batch_normalization_12/beta
2:0 (2"batch_normalization_12/moving_mean
6:4 (2&batch_normalization_12/moving_variance
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
*__inference_dropout_8_layer_call_fn_492820
*__inference_dropout_8_layer_call_fn_492825?
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
?
?trace_0
?trace_12?
E__inference_dropout_8_layer_call_and_return_conditional_losses_492830
E__inference_dropout_8_layer_call_and_return_conditional_losses_492842?
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
*__inference_conv2d_17_layer_call_fn_492851?
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
E__inference_conv2d_17_layer_call_and_return_conditional_losses_492861?
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
*:( 2conv2d_17/kernel
: 2conv2d_17/bias
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
.__inference_activation_17_layer_call_fn_492866?
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
I__inference_activation_17_layer_call_and_return_conditional_losses_492871?
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
7__inference_batch_normalization_13_layer_call_fn_492884
7__inference_batch_normalization_13_layer_call_fn_492897?
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
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_492915
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_492933?
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
*:( 2batch_normalization_13/gamma
):' 2batch_normalization_13/beta
2:0  (2"batch_normalization_13/moving_mean
6:4  (2&batch_normalization_13/moving_variance
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
*__inference_conv2d_18_layer_call_fn_492942?
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
E__inference_conv2d_18_layer_call_and_return_conditional_losses_492952?
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
*:( @2conv2d_18/kernel
:@2conv2d_18/bias
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
.__inference_activation_18_layer_call_fn_492957?
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
I__inference_activation_18_layer_call_and_return_conditional_losses_492962?
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
7__inference_batch_normalization_14_layer_call_fn_492975
7__inference_batch_normalization_14_layer_call_fn_492988?
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
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_493006
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_493024?
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
*:(@2batch_normalization_14/gamma
):'@2batch_normalization_14/beta
2:0@ (2"batch_normalization_14/moving_mean
6:4@ (2&batch_normalization_14/moving_variance
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
*__inference_dropout_9_layer_call_fn_493029
*__inference_dropout_9_layer_call_fn_493034?
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
E__inference_dropout_9_layer_call_and_return_conditional_losses_493039
E__inference_dropout_9_layer_call_and_return_conditional_losses_493051?
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
*__inference_conv2d_19_layer_call_fn_493060?
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
E__inference_conv2d_19_layer_call_and_return_conditional_losses_493070?
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
*:(@ 2conv2d_19/kernel
: 2conv2d_19/bias
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
.__inference_activation_19_layer_call_fn_493075?
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
I__inference_activation_19_layer_call_and_return_conditional_losses_493080?
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
(__inference_encoded_layer_call_fn_493093
(__inference_encoded_layer_call_fn_493106?
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
C__inference_encoded_layer_call_and_return_conditional_losses_493124
C__inference_encoded_layer_call_and_return_conditional_losses_493142?
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
4__inference_conv2d_transpose_12_layer_call_fn_493151?
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
O__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_493185?
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
4:2@ 2conv2d_transpose_12/kernel
&:$@2conv2d_transpose_12/bias
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
4__inference_conv2d_transpose_13_layer_call_fn_493194?
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
O__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_493228?
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
4:2 @2conv2d_transpose_13/kernel
&:$ 2conv2d_transpose_13/bias
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
4__inference_conv2d_transpose_14_layer_call_fn_493237?
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
O__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_493271?
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
4:2 2conv2d_transpose_14/kernel
&:$2conv2d_transpose_14/bias
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
(__inference_decoded_layer_call_fn_493280?
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
C__inference_decoded_layer_call_and_return_conditional_losses_493314?
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
(:&2decoded/kernel
:2decoded/bias
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
-__inference_sequential_4_layer_call_fn_491570conv2d_16_input"?
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
-__inference_sequential_4_layer_call_fn_492297inputs"?
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
-__inference_sequential_4_layer_call_fn_492366inputs"?
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
-__inference_sequential_4_layer_call_fn_491975conv2d_16_input"?
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
H__inference_sequential_4_layer_call_and_return_conditional_losses_492538inputs"?
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
H__inference_sequential_4_layer_call_and_return_conditional_losses_492724inputs"?
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
H__inference_sequential_4_layer_call_and_return_conditional_losses_492063conv2d_16_input"?
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
H__inference_sequential_4_layer_call_and_return_conditional_losses_492151conv2d_16_input"?
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
$__inference_signature_wrapper_492228conv2d_16_input"?
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
*__inference_conv2d_16_layer_call_fn_492733inputs"?
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
E__inference_conv2d_16_layer_call_and_return_conditional_losses_492743inputs"?
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
.__inference_activation_16_layer_call_fn_492748inputs"?
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
I__inference_activation_16_layer_call_and_return_conditional_losses_492753inputs"?
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
7__inference_batch_normalization_12_layer_call_fn_492766inputs"?
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
7__inference_batch_normalization_12_layer_call_fn_492779inputs"?
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
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_492797inputs"?
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
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_492815inputs"?
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
*__inference_dropout_8_layer_call_fn_492820inputs"?
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
*__inference_dropout_8_layer_call_fn_492825inputs"?
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
E__inference_dropout_8_layer_call_and_return_conditional_losses_492830inputs"?
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
E__inference_dropout_8_layer_call_and_return_conditional_losses_492842inputs"?
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
*__inference_conv2d_17_layer_call_fn_492851inputs"?
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
E__inference_conv2d_17_layer_call_and_return_conditional_losses_492861inputs"?
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
.__inference_activation_17_layer_call_fn_492866inputs"?
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
I__inference_activation_17_layer_call_and_return_conditional_losses_492871inputs"?
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
7__inference_batch_normalization_13_layer_call_fn_492884inputs"?
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
7__inference_batch_normalization_13_layer_call_fn_492897inputs"?
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
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_492915inputs"?
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
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_492933inputs"?
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
*__inference_conv2d_18_layer_call_fn_492942inputs"?
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
E__inference_conv2d_18_layer_call_and_return_conditional_losses_492952inputs"?
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
.__inference_activation_18_layer_call_fn_492957inputs"?
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
I__inference_activation_18_layer_call_and_return_conditional_losses_492962inputs"?
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
7__inference_batch_normalization_14_layer_call_fn_492975inputs"?
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
7__inference_batch_normalization_14_layer_call_fn_492988inputs"?
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
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_493006inputs"?
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
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_493024inputs"?
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
*__inference_dropout_9_layer_call_fn_493029inputs"?
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
*__inference_dropout_9_layer_call_fn_493034inputs"?
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
E__inference_dropout_9_layer_call_and_return_conditional_losses_493039inputs"?
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
E__inference_dropout_9_layer_call_and_return_conditional_losses_493051inputs"?
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
*__inference_conv2d_19_layer_call_fn_493060inputs"?
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
E__inference_conv2d_19_layer_call_and_return_conditional_losses_493070inputs"?
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
.__inference_activation_19_layer_call_fn_493075inputs"?
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
I__inference_activation_19_layer_call_and_return_conditional_losses_493080inputs"?
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
(__inference_encoded_layer_call_fn_493093inputs"?
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
(__inference_encoded_layer_call_fn_493106inputs"?
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
C__inference_encoded_layer_call_and_return_conditional_losses_493124inputs"?
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
C__inference_encoded_layer_call_and_return_conditional_losses_493142inputs"?
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
4__inference_conv2d_transpose_12_layer_call_fn_493151inputs"?
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
O__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_493185inputs"?
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
4__inference_conv2d_transpose_13_layer_call_fn_493194inputs"?
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
O__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_493228inputs"?
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
4__inference_conv2d_transpose_14_layer_call_fn_493237inputs"?
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
O__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_493271inputs"?
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
(__inference_decoded_layer_call_fn_493280inputs"?
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
C__inference_decoded_layer_call_and_return_conditional_losses_493314inputs"?
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
/:-2Adam/conv2d_16/kernel/m
!:2Adam/conv2d_16/bias/m
/:-2#Adam/batch_normalization_12/gamma/m
.:,2"Adam/batch_normalization_12/beta/m
/:- 2Adam/conv2d_17/kernel/m
!: 2Adam/conv2d_17/bias/m
/:- 2#Adam/batch_normalization_13/gamma/m
.:, 2"Adam/batch_normalization_13/beta/m
/:- @2Adam/conv2d_18/kernel/m
!:@2Adam/conv2d_18/bias/m
/:-@2#Adam/batch_normalization_14/gamma/m
.:,@2"Adam/batch_normalization_14/beta/m
/:-@ 2Adam/conv2d_19/kernel/m
!: 2Adam/conv2d_19/bias/m
 : 2Adam/encoded/gamma/m
: 2Adam/encoded/beta/m
9:7@ 2!Adam/conv2d_transpose_12/kernel/m
+:)@2Adam/conv2d_transpose_12/bias/m
9:7 @2!Adam/conv2d_transpose_13/kernel/m
+:) 2Adam/conv2d_transpose_13/bias/m
9:7 2!Adam/conv2d_transpose_14/kernel/m
+:)2Adam/conv2d_transpose_14/bias/m
-:+2Adam/decoded/kernel/m
:2Adam/decoded/bias/m
/:-2Adam/conv2d_16/kernel/v
!:2Adam/conv2d_16/bias/v
/:-2#Adam/batch_normalization_12/gamma/v
.:,2"Adam/batch_normalization_12/beta/v
/:- 2Adam/conv2d_17/kernel/v
!: 2Adam/conv2d_17/bias/v
/:- 2#Adam/batch_normalization_13/gamma/v
.:, 2"Adam/batch_normalization_13/beta/v
/:- @2Adam/conv2d_18/kernel/v
!:@2Adam/conv2d_18/bias/v
/:-@2#Adam/batch_normalization_14/gamma/v
.:,@2"Adam/batch_normalization_14/beta/v
/:-@ 2Adam/conv2d_19/kernel/v
!: 2Adam/conv2d_19/bias/v
 : 2Adam/encoded/gamma/v
: 2Adam/encoded/beta/v
9:7@ 2!Adam/conv2d_transpose_12/kernel/v
+:)@2Adam/conv2d_transpose_12/bias/v
9:7 @2!Adam/conv2d_transpose_13/kernel/v
+:) 2Adam/conv2d_transpose_13/bias/v
9:7 2!Adam/conv2d_transpose_14/kernel/v
+:)2Adam/conv2d_transpose_14/bias/v
-:+2Adam/decoded/kernel/v
:2Adam/decoded/bias/v?
!__inference__wrapped_model_490895?,"#2345CDSTUV]^mnop~????????????B??
8?5
3?0
conv2d_16_input???????????
? ";?8
6
decoded+?(
decoded????????????
I__inference_activation_16_layer_call_and_return_conditional_losses_492753l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
.__inference_activation_16_layer_call_fn_492748_9?6
/?,
*?'
inputs???????????
? ""?????????????
I__inference_activation_17_layer_call_and_return_conditional_losses_492871h7?4
-?*
(?%
inputs?????????@@ 
? "-?*
#? 
0?????????@@ 
? ?
.__inference_activation_17_layer_call_fn_492866[7?4
-?*
(?%
inputs?????????@@ 
? " ??????????@@ ?
I__inference_activation_18_layer_call_and_return_conditional_losses_492962h7?4
-?*
(?%
inputs?????????  @
? "-?*
#? 
0?????????  @
? ?
.__inference_activation_18_layer_call_fn_492957[7?4
-?*
(?%
inputs?????????  @
? " ??????????  @?
I__inference_activation_19_layer_call_and_return_conditional_losses_493080h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
.__inference_activation_19_layer_call_fn_493075[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_492797?2345M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_492815?2345M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
7__inference_batch_normalization_12_layer_call_fn_492766?2345M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
7__inference_batch_normalization_12_layer_call_fn_492779?2345M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_492915?STUVM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_492933?STUVM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
7__inference_batch_normalization_13_layer_call_fn_492884?STUVM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
7__inference_batch_normalization_13_layer_call_fn_492897?STUVM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_493006?mnopM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_493024?mnopM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
7__inference_batch_normalization_14_layer_call_fn_492975?mnopM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
7__inference_batch_normalization_14_layer_call_fn_492988?mnopM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
E__inference_conv2d_16_layer_call_and_return_conditional_losses_492743p"#9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
*__inference_conv2d_16_layer_call_fn_492733c"#9?6
/?,
*?'
inputs???????????
? ""?????????????
E__inference_conv2d_17_layer_call_and_return_conditional_losses_492861nCD9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????@@ 
? ?
*__inference_conv2d_17_layer_call_fn_492851aCD9?6
/?,
*?'
inputs???????????
? " ??????????@@ ?
E__inference_conv2d_18_layer_call_and_return_conditional_losses_492952l]^7?4
-?*
(?%
inputs?????????@@ 
? "-?*
#? 
0?????????  @
? ?
*__inference_conv2d_18_layer_call_fn_492942_]^7?4
-?*
(?%
inputs?????????@@ 
? " ??????????  @?
E__inference_conv2d_19_layer_call_and_return_conditional_losses_493070l~7?4
-?*
(?%
inputs?????????  @
? "-?*
#? 
0????????? 
? ?
*__inference_conv2d_19_layer_call_fn_493060_~7?4
-?*
(?%
inputs?????????  @
? " ?????????? ?
O__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_493185???I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????@
? ?
4__inference_conv2d_transpose_12_layer_call_fn_493151???I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+???????????????????????????@?
O__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_493228???I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
4__inference_conv2d_transpose_13_layer_call_fn_493194???I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
O__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_493271???I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
4__inference_conv2d_transpose_14_layer_call_fn_493237???I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
C__inference_decoded_layer_call_and_return_conditional_losses_493314???I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
(__inference_decoded_layer_call_fn_493280???I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
E__inference_dropout_8_layer_call_and_return_conditional_losses_492830p=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
E__inference_dropout_8_layer_call_and_return_conditional_losses_492842p=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
*__inference_dropout_8_layer_call_fn_492820c=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
*__inference_dropout_8_layer_call_fn_492825c=?:
3?0
*?'
inputs???????????
p
? ""?????????????
E__inference_dropout_9_layer_call_and_return_conditional_losses_493039l;?8
1?.
(?%
inputs?????????  @
p 
? "-?*
#? 
0?????????  @
? ?
E__inference_dropout_9_layer_call_and_return_conditional_losses_493051l;?8
1?.
(?%
inputs?????????  @
p
? "-?*
#? 
0?????????  @
? ?
*__inference_dropout_9_layer_call_fn_493029_;?8
1?.
(?%
inputs?????????  @
p 
? " ??????????  @?
*__inference_dropout_9_layer_call_fn_493034_;?8
1?.
(?%
inputs?????????  @
p
? " ??????????  @?
C__inference_encoded_layer_call_and_return_conditional_losses_493124?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
C__inference_encoded_layer_call_and_return_conditional_losses_493142?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
(__inference_encoded_layer_call_fn_493093?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
(__inference_encoded_layer_call_fn_493106?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
H__inference_sequential_4_layer_call_and_return_conditional_losses_492063?,"#2345CDSTUV]^mnop~????????????J?G
@?=
3?0
conv2d_16_input???????????
p 

 
? "/?,
%?"
0???????????
? ?
H__inference_sequential_4_layer_call_and_return_conditional_losses_492151?,"#2345CDSTUV]^mnop~????????????J?G
@?=
3?0
conv2d_16_input???????????
p

 
? "/?,
%?"
0???????????
? ?
H__inference_sequential_4_layer_call_and_return_conditional_losses_492538?,"#2345CDSTUV]^mnop~????????????A?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
H__inference_sequential_4_layer_call_and_return_conditional_losses_492724?,"#2345CDSTUV]^mnop~????????????A?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????
? ?
-__inference_sequential_4_layer_call_fn_491570?,"#2345CDSTUV]^mnop~????????????J?G
@?=
3?0
conv2d_16_input???????????
p 

 
? ""?????????????
-__inference_sequential_4_layer_call_fn_491975?,"#2345CDSTUV]^mnop~????????????J?G
@?=
3?0
conv2d_16_input???????????
p

 
? ""?????????????
-__inference_sequential_4_layer_call_fn_492297?,"#2345CDSTUV]^mnop~????????????A?>
7?4
*?'
inputs???????????
p 

 
? ""?????????????
-__inference_sequential_4_layer_call_fn_492366?,"#2345CDSTUV]^mnop~????????????A?>
7?4
*?'
inputs???????????
p

 
? ""?????????????
$__inference_signature_wrapper_492228?,"#2345CDSTUV]^mnop~????????????U?R
? 
K?H
F
conv2d_16_input3?0
conv2d_16_input???????????";?8
6
decoded+?(
decoded???????????