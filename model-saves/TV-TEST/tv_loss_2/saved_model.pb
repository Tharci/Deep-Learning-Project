Ֆ(
??
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
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
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
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8??"
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
shape:*&
shared_nameAdam/decoded/kernel/v
?
)Adam/decoded/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decoded/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_20/bias/v
?
3Adam/conv2d_transpose_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_20/bias/v*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_20/kernel/v
?
5Adam/conv2d_transpose_20/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_20/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_19/bias/v
?
3Adam/conv2d_transpose_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_19/bias/v*
_output_shapes
: *
dtype0
?
!Adam/conv2d_transpose_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/conv2d_transpose_19/kernel/v
?
5Adam/conv2d_transpose_19/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_19/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_transpose_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/conv2d_transpose_18/bias/v
?
3Adam/conv2d_transpose_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_18/bias/v*
_output_shapes
:@*
dtype0
?
!Adam/conv2d_transpose_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *2
shared_name#!Adam/conv2d_transpose_18/kernel/v
?
5Adam/conv2d_transpose_18/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_18/kernel/v*&
_output_shapes
:@ *
dtype0
?
"Adam/batch_normalization_48/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_48/beta/v
?
6Adam/batch_normalization_48/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_48/beta/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_48/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_48/gamma/v
?
7Adam/batch_normalization_48/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_48/gamma/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_48/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_48/bias/v
{
)Adam/conv2d_48/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_48/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *(
shared_nameAdam/conv2d_48/kernel/v
?
+Adam/conv2d_48/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/kernel/v*&
_output_shapes
:@ *
dtype0
?
"Adam/batch_normalization_47/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_47/beta/v
?
6Adam/batch_normalization_47/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_47/beta/v*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_47/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_47/gamma/v
?
7Adam/batch_normalization_47/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_47/gamma/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_47/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_47/bias/v
{
)Adam/conv2d_47/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_47/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_47/kernel/v
?
+Adam/conv2d_47/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/kernel/v*&
_output_shapes
: @*
dtype0
?
"Adam/batch_normalization_46/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_46/beta/v
?
6Adam/batch_normalization_46/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_46/beta/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_46/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_46/gamma/v
?
7Adam/batch_normalization_46/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_46/gamma/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_46/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_46/bias/v
{
)Adam/conv2d_46/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_46/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_46/kernel/v
?
+Adam/conv2d_46/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/kernel/v*&
_output_shapes
:  *
dtype0
?
"Adam/batch_normalization_45/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_45/beta/v
?
6Adam/batch_normalization_45/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_45/beta/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_45/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_45/gamma/v
?
7Adam/batch_normalization_45/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_45/gamma/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_45/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_45/bias/v
{
)Adam/conv2d_45/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_45/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_45/kernel/v
?
+Adam/conv2d_45/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/kernel/v*&
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_44/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_44/beta/v
?
6Adam/batch_normalization_44/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_44/beta/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_44/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_44/gamma/v
?
7Adam/batch_normalization_44/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_44/gamma/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_44/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_44/bias/v
{
)Adam/conv2d_44/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_44/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_44/kernel/v
?
+Adam/conv2d_44/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/kernel/v*&
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_43/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_43/beta/v
?
6Adam/batch_normalization_43/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_43/beta/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_43/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_43/gamma/v
?
7Adam/batch_normalization_43/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_43/gamma/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_43/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_43/bias/v
{
)Adam/conv2d_43/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_43/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_43/kernel/v
?
+Adam/conv2d_43/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/kernel/v*&
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_42/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_42/beta/v
?
6Adam/batch_normalization_42/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_42/beta/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_42/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_42/gamma/v
?
7Adam/batch_normalization_42/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_42/gamma/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_42/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_42/bias/v
{
)Adam/conv2d_42/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_42/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_42/kernel/v
?
+Adam/conv2d_42/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/kernel/v*&
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
shape:*&
shared_nameAdam/decoded/kernel/m
?
)Adam/decoded/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decoded/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_20/bias/m
?
3Adam/conv2d_transpose_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_20/bias/m*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_20/kernel/m
?
5Adam/conv2d_transpose_20/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_20/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_19/bias/m
?
3Adam/conv2d_transpose_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_19/bias/m*
_output_shapes
: *
dtype0
?
!Adam/conv2d_transpose_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/conv2d_transpose_19/kernel/m
?
5Adam/conv2d_transpose_19/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_19/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_transpose_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/conv2d_transpose_18/bias/m
?
3Adam/conv2d_transpose_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_18/bias/m*
_output_shapes
:@*
dtype0
?
!Adam/conv2d_transpose_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *2
shared_name#!Adam/conv2d_transpose_18/kernel/m
?
5Adam/conv2d_transpose_18/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_18/kernel/m*&
_output_shapes
:@ *
dtype0
?
"Adam/batch_normalization_48/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_48/beta/m
?
6Adam/batch_normalization_48/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_48/beta/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_48/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_48/gamma/m
?
7Adam/batch_normalization_48/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_48/gamma/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_48/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_48/bias/m
{
)Adam/conv2d_48/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_48/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *(
shared_nameAdam/conv2d_48/kernel/m
?
+Adam/conv2d_48/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/kernel/m*&
_output_shapes
:@ *
dtype0
?
"Adam/batch_normalization_47/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_47/beta/m
?
6Adam/batch_normalization_47/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_47/beta/m*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_47/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_47/gamma/m
?
7Adam/batch_normalization_47/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_47/gamma/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_47/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_47/bias/m
{
)Adam/conv2d_47/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_47/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_47/kernel/m
?
+Adam/conv2d_47/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/kernel/m*&
_output_shapes
: @*
dtype0
?
"Adam/batch_normalization_46/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_46/beta/m
?
6Adam/batch_normalization_46/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_46/beta/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_46/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_46/gamma/m
?
7Adam/batch_normalization_46/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_46/gamma/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_46/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_46/bias/m
{
)Adam/conv2d_46/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_46/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_46/kernel/m
?
+Adam/conv2d_46/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/kernel/m*&
_output_shapes
:  *
dtype0
?
"Adam/batch_normalization_45/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_45/beta/m
?
6Adam/batch_normalization_45/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_45/beta/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_45/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_45/gamma/m
?
7Adam/batch_normalization_45/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_45/gamma/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_45/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_45/bias/m
{
)Adam/conv2d_45/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_45/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_45/kernel/m
?
+Adam/conv2d_45/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/kernel/m*&
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_44/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_44/beta/m
?
6Adam/batch_normalization_44/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_44/beta/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_44/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_44/gamma/m
?
7Adam/batch_normalization_44/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_44/gamma/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_44/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_44/bias/m
{
)Adam/conv2d_44/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_44/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_44/kernel/m
?
+Adam/conv2d_44/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/kernel/m*&
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_43/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_43/beta/m
?
6Adam/batch_normalization_43/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_43/beta/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_43/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_43/gamma/m
?
7Adam/batch_normalization_43/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_43/gamma/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_43/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_43/bias/m
{
)Adam/conv2d_43/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_43/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_43/kernel/m
?
+Adam/conv2d_43/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/kernel/m*&
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_42/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_42/beta/m
?
6Adam/batch_normalization_42/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_42/beta/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_42/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_42/gamma/m
?
7Adam/batch_normalization_42/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_42/gamma/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_42/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_42/bias/m
{
)Adam/conv2d_42/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_42/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_42/kernel/m
?
+Adam/conv2d_42/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/kernel/m*&
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
shape:*
shared_namedecoded/kernel
y
"decoded/kernel/Read/ReadVariableOpReadVariableOpdecoded/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_20/bias
?
,conv2d_transpose_20/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_20/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_20/kernel
?
.conv2d_transpose_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_20/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_19/bias
?
,conv2d_transpose_19/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_19/bias*
_output_shapes
: *
dtype0
?
conv2d_transpose_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameconv2d_transpose_19/kernel
?
.conv2d_transpose_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_19/kernel*&
_output_shapes
: @*
dtype0
?
conv2d_transpose_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv2d_transpose_18/bias
?
,conv2d_transpose_18/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_18/bias*
_output_shapes
:@*
dtype0
?
conv2d_transpose_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *+
shared_nameconv2d_transpose_18/kernel
?
.conv2d_transpose_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_18/kernel*&
_output_shapes
:@ *
dtype0
?
&batch_normalization_48/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_48/moving_variance
?
:batch_normalization_48/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_48/moving_variance*
_output_shapes
: *
dtype0
?
"batch_normalization_48/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_48/moving_mean
?
6batch_normalization_48/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_48/moving_mean*
_output_shapes
: *
dtype0
?
batch_normalization_48/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_48/beta
?
/batch_normalization_48/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_48/beta*
_output_shapes
: *
dtype0
?
batch_normalization_48/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_48/gamma
?
0batch_normalization_48/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_48/gamma*
_output_shapes
: *
dtype0
t
conv2d_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_48/bias
m
"conv2d_48/bias/Read/ReadVariableOpReadVariableOpconv2d_48/bias*
_output_shapes
: *
dtype0
?
conv2d_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *!
shared_nameconv2d_48/kernel
}
$conv2d_48/kernel/Read/ReadVariableOpReadVariableOpconv2d_48/kernel*&
_output_shapes
:@ *
dtype0
?
&batch_normalization_47/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_47/moving_variance
?
:batch_normalization_47/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_47/moving_variance*
_output_shapes
:@*
dtype0
?
"batch_normalization_47/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_47/moving_mean
?
6batch_normalization_47/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_47/moving_mean*
_output_shapes
:@*
dtype0
?
batch_normalization_47/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_47/beta
?
/batch_normalization_47/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_47/beta*
_output_shapes
:@*
dtype0
?
batch_normalization_47/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_47/gamma
?
0batch_normalization_47/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_47/gamma*
_output_shapes
:@*
dtype0
t
conv2d_47/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_47/bias
m
"conv2d_47/bias/Read/ReadVariableOpReadVariableOpconv2d_47/bias*
_output_shapes
:@*
dtype0
?
conv2d_47/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_47/kernel
}
$conv2d_47/kernel/Read/ReadVariableOpReadVariableOpconv2d_47/kernel*&
_output_shapes
: @*
dtype0
?
&batch_normalization_46/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_46/moving_variance
?
:batch_normalization_46/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_46/moving_variance*
_output_shapes
: *
dtype0
?
"batch_normalization_46/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_46/moving_mean
?
6batch_normalization_46/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_46/moving_mean*
_output_shapes
: *
dtype0
?
batch_normalization_46/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_46/beta
?
/batch_normalization_46/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_46/beta*
_output_shapes
: *
dtype0
?
batch_normalization_46/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_46/gamma
?
0batch_normalization_46/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_46/gamma*
_output_shapes
: *
dtype0
t
conv2d_46/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_46/bias
m
"conv2d_46/bias/Read/ReadVariableOpReadVariableOpconv2d_46/bias*
_output_shapes
: *
dtype0
?
conv2d_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_46/kernel
}
$conv2d_46/kernel/Read/ReadVariableOpReadVariableOpconv2d_46/kernel*&
_output_shapes
:  *
dtype0
?
&batch_normalization_45/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_45/moving_variance
?
:batch_normalization_45/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_45/moving_variance*
_output_shapes
: *
dtype0
?
"batch_normalization_45/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_45/moving_mean
?
6batch_normalization_45/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_45/moving_mean*
_output_shapes
: *
dtype0
?
batch_normalization_45/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_45/beta
?
/batch_normalization_45/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_45/beta*
_output_shapes
: *
dtype0
?
batch_normalization_45/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_45/gamma
?
0batch_normalization_45/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_45/gamma*
_output_shapes
: *
dtype0
t
conv2d_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_45/bias
m
"conv2d_45/bias/Read/ReadVariableOpReadVariableOpconv2d_45/bias*
_output_shapes
: *
dtype0
?
conv2d_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_45/kernel
}
$conv2d_45/kernel/Read/ReadVariableOpReadVariableOpconv2d_45/kernel*&
_output_shapes
: *
dtype0
?
&batch_normalization_44/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_44/moving_variance
?
:batch_normalization_44/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_44/moving_variance*
_output_shapes
:*
dtype0
?
"batch_normalization_44/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_44/moving_mean
?
6batch_normalization_44/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_44/moving_mean*
_output_shapes
:*
dtype0
?
batch_normalization_44/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_44/beta
?
/batch_normalization_44/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_44/beta*
_output_shapes
:*
dtype0
?
batch_normalization_44/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_44/gamma
?
0batch_normalization_44/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_44/gamma*
_output_shapes
:*
dtype0
t
conv2d_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_44/bias
m
"conv2d_44/bias/Read/ReadVariableOpReadVariableOpconv2d_44/bias*
_output_shapes
:*
dtype0
?
conv2d_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_44/kernel
}
$conv2d_44/kernel/Read/ReadVariableOpReadVariableOpconv2d_44/kernel*&
_output_shapes
:*
dtype0
?
&batch_normalization_43/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_43/moving_variance
?
:batch_normalization_43/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_43/moving_variance*
_output_shapes
:*
dtype0
?
"batch_normalization_43/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_43/moving_mean
?
6batch_normalization_43/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_43/moving_mean*
_output_shapes
:*
dtype0
?
batch_normalization_43/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_43/beta
?
/batch_normalization_43/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_43/beta*
_output_shapes
:*
dtype0
?
batch_normalization_43/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_43/gamma
?
0batch_normalization_43/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_43/gamma*
_output_shapes
:*
dtype0
t
conv2d_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_43/bias
m
"conv2d_43/bias/Read/ReadVariableOpReadVariableOpconv2d_43/bias*
_output_shapes
:*
dtype0
?
conv2d_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_43/kernel
}
$conv2d_43/kernel/Read/ReadVariableOpReadVariableOpconv2d_43/kernel*&
_output_shapes
:*
dtype0
?
&batch_normalization_42/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_42/moving_variance
?
:batch_normalization_42/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_42/moving_variance*
_output_shapes
:*
dtype0
?
"batch_normalization_42/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_42/moving_mean
?
6batch_normalization_42/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_42/moving_mean*
_output_shapes
:*
dtype0
?
batch_normalization_42/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_42/beta
?
/batch_normalization_42/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_42/beta*
_output_shapes
:*
dtype0
?
batch_normalization_42/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_42/gamma
?
0batch_normalization_42/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_42/gamma*
_output_shapes
:*
dtype0
t
conv2d_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_42/bias
m
"conv2d_42/bias/Read/ReadVariableOpReadVariableOpconv2d_42/bias*
_output_shapes
:*
dtype0
?
conv2d_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_42/kernel
}
$conv2d_42/kernel/Read/ReadVariableOpReadVariableOpconv2d_42/kernel*&
_output_shapes
:*
dtype0
?
serving_default_conv2d_42_inputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_42_inputconv2d_42/kernelconv2d_42/biasbatch_normalization_42/gammabatch_normalization_42/beta"batch_normalization_42/moving_mean&batch_normalization_42/moving_varianceconv2d_43/kernelconv2d_43/biasbatch_normalization_43/gammabatch_normalization_43/beta"batch_normalization_43/moving_mean&batch_normalization_43/moving_varianceconv2d_44/kernelconv2d_44/biasbatch_normalization_44/gammabatch_normalization_44/beta"batch_normalization_44/moving_mean&batch_normalization_44/moving_varianceconv2d_45/kernelconv2d_45/biasbatch_normalization_45/gammabatch_normalization_45/beta"batch_normalization_45/moving_mean&batch_normalization_45/moving_varianceconv2d_46/kernelconv2d_46/biasbatch_normalization_46/gammabatch_normalization_46/beta"batch_normalization_46/moving_mean&batch_normalization_46/moving_varianceconv2d_47/kernelconv2d_47/biasbatch_normalization_47/gammabatch_normalization_47/beta"batch_normalization_47/moving_mean&batch_normalization_47/moving_varianceconv2d_48/kernelconv2d_48/biasbatch_normalization_48/gammabatch_normalization_48/beta"batch_normalization_48/moving_mean&batch_normalization_48/moving_varianceconv2d_transpose_18/kernelconv2d_transpose_18/biasconv2d_transpose_19/kernelconv2d_transpose_19/biasconv2d_transpose_20/kernelconv2d_transpose_20/biasdecoded/kerneldecoded/bias*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*2
config_proto" 

CPU

GPU2 *0J 8? *.
f)R'
%__inference_signature_wrapper_2597352

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
valueޛBڛ Bқ
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer-17
layer_with_weights-12
layer-18
layer_with_weights-13
layer-19
layer-20
layer_with_weights-14
layer-21
layer-22
layer_with_weights-15
layer-23
layer-24
layer_with_weights-16
layer-25
layer-26
layer_with_weights-17
layer-27
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_default_save_signature
$	optimizer
%
signatures*
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias
 ._jit_compiled_convolution_op*
?
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5axis
	6gamma
7beta
8moving_mean
9moving_variance*
?
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses* 
?
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias
 H_jit_compiled_convolution_op*
?
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance*
?
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses* 
?
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias
 b_jit_compiled_convolution_op*
?
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses
iaxis
	jgamma
kbeta
lmoving_mean
mmoving_variance*
?
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses* 
?
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

zkernel
{bias
 |_jit_compiled_convolution_op*
?
}	variables
~trainable_variables
regularization_losses
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
+?&call_and_return_all_conditional_losses* 
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
+?&call_and_return_all_conditional_losses* 
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
+?&call_and_return_all_conditional_losses* 
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
+?&call_and_return_all_conditional_losses* 
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
+?&call_and_return_all_conditional_losses* 
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
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
!?_jit_compiled_convolution_op*
?
,0
-1
62
73
84
95
F6
G7
P8
Q9
R10
S11
`12
a13
j14
k15
l16
m17
z18
{19
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
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49*
?
,0
-1
62
73
F4
G5
P6
Q7
`8
a9
j10
k11
z12
{13
?14
?15
?16
?17
?18
?19
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
?31
?32
?33
?34
?35*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
#_default_save_signature
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
* 
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate,m?-m?6m?7m?Fm?Gm?Pm?Qm?`m?am?jm?km?zm?{m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?,v?-v?6v?7v?Fv?Gv?Pv?Qv?`v?av?jv?kv?zv?{v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?*

?serving_default* 

,0
-1*

,0
-1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_42/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_42/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
60
71
82
93*

60
71*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_42/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_42/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_42/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_42/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

F0
G1*

F0
G1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_43/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_43/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
P0
Q1
R2
S3*

P0
Q1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_43/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_43/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_43/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_43/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

`0
a1*

`0
a1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_44/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_44/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
j0
k1
l2
m3*

j0
k1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_44/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_44/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_44/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_44/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

z0
{1*

z0
{1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_45/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_45/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
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
}	variables
~trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_45/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_45/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_45/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_45/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
?trace_0* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_46/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_46/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_46/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_46/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_46/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_46/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
a[
VARIABLE_VALUEconv2d_47/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_47/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_47/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_47/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_47/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE&batch_normalization_47/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
a[
VARIABLE_VALUEconv2d_48/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_48/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_48/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_48/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_48/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE&batch_normalization_48/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
ke
VARIABLE_VALUEconv2d_transpose_18/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEconv2d_transpose_18/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
ke
VARIABLE_VALUEconv2d_transpose_19/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEconv2d_transpose_19/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
ke
VARIABLE_VALUEconv2d_transpose_20/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEconv2d_transpose_20/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEdecoded/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdecoded/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
r
80
91
R2
S3
l4
m5
?6
?7
?8
?9
?10
?11
?12
?13*
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
17
18
19
20
21
22
23
24
25
26
27*

?0*
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

80
91*
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
R0
S1*
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
l0
m1*
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
?	variables
?	keras_api

?total

?count*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_42/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_42/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_42/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_42/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_43/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_43/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_43/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_43/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_44/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_44/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_44/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_44/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_45/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_45/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_45/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_45/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_46/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_46/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_46/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_46/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_47/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_47/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_47/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_47/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_48/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_48/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_48/gamma/mRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_48/beta/mQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_18/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_18/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_19/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_19/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_20/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_20/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/decoded/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/decoded/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_42/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_42/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_42/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_42/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_43/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_43/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_43/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_43/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_44/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_44/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_44/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_44/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_45/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_45/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_45/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_45/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_46/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_46/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_46/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_46/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_47/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_47/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_47/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_47/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_48/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_48/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_48/gamma/vRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_48/beta/vQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_18/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_18/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_19/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_19/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_20/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_20/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/decoded/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/decoded/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?3
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_42/kernel/Read/ReadVariableOp"conv2d_42/bias/Read/ReadVariableOp0batch_normalization_42/gamma/Read/ReadVariableOp/batch_normalization_42/beta/Read/ReadVariableOp6batch_normalization_42/moving_mean/Read/ReadVariableOp:batch_normalization_42/moving_variance/Read/ReadVariableOp$conv2d_43/kernel/Read/ReadVariableOp"conv2d_43/bias/Read/ReadVariableOp0batch_normalization_43/gamma/Read/ReadVariableOp/batch_normalization_43/beta/Read/ReadVariableOp6batch_normalization_43/moving_mean/Read/ReadVariableOp:batch_normalization_43/moving_variance/Read/ReadVariableOp$conv2d_44/kernel/Read/ReadVariableOp"conv2d_44/bias/Read/ReadVariableOp0batch_normalization_44/gamma/Read/ReadVariableOp/batch_normalization_44/beta/Read/ReadVariableOp6batch_normalization_44/moving_mean/Read/ReadVariableOp:batch_normalization_44/moving_variance/Read/ReadVariableOp$conv2d_45/kernel/Read/ReadVariableOp"conv2d_45/bias/Read/ReadVariableOp0batch_normalization_45/gamma/Read/ReadVariableOp/batch_normalization_45/beta/Read/ReadVariableOp6batch_normalization_45/moving_mean/Read/ReadVariableOp:batch_normalization_45/moving_variance/Read/ReadVariableOp$conv2d_46/kernel/Read/ReadVariableOp"conv2d_46/bias/Read/ReadVariableOp0batch_normalization_46/gamma/Read/ReadVariableOp/batch_normalization_46/beta/Read/ReadVariableOp6batch_normalization_46/moving_mean/Read/ReadVariableOp:batch_normalization_46/moving_variance/Read/ReadVariableOp$conv2d_47/kernel/Read/ReadVariableOp"conv2d_47/bias/Read/ReadVariableOp0batch_normalization_47/gamma/Read/ReadVariableOp/batch_normalization_47/beta/Read/ReadVariableOp6batch_normalization_47/moving_mean/Read/ReadVariableOp:batch_normalization_47/moving_variance/Read/ReadVariableOp$conv2d_48/kernel/Read/ReadVariableOp"conv2d_48/bias/Read/ReadVariableOp0batch_normalization_48/gamma/Read/ReadVariableOp/batch_normalization_48/beta/Read/ReadVariableOp6batch_normalization_48/moving_mean/Read/ReadVariableOp:batch_normalization_48/moving_variance/Read/ReadVariableOp.conv2d_transpose_18/kernel/Read/ReadVariableOp,conv2d_transpose_18/bias/Read/ReadVariableOp.conv2d_transpose_19/kernel/Read/ReadVariableOp,conv2d_transpose_19/bias/Read/ReadVariableOp.conv2d_transpose_20/kernel/Read/ReadVariableOp,conv2d_transpose_20/bias/Read/ReadVariableOp"decoded/kernel/Read/ReadVariableOp decoded/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_42/kernel/m/Read/ReadVariableOp)Adam/conv2d_42/bias/m/Read/ReadVariableOp7Adam/batch_normalization_42/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_42/beta/m/Read/ReadVariableOp+Adam/conv2d_43/kernel/m/Read/ReadVariableOp)Adam/conv2d_43/bias/m/Read/ReadVariableOp7Adam/batch_normalization_43/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_43/beta/m/Read/ReadVariableOp+Adam/conv2d_44/kernel/m/Read/ReadVariableOp)Adam/conv2d_44/bias/m/Read/ReadVariableOp7Adam/batch_normalization_44/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_44/beta/m/Read/ReadVariableOp+Adam/conv2d_45/kernel/m/Read/ReadVariableOp)Adam/conv2d_45/bias/m/Read/ReadVariableOp7Adam/batch_normalization_45/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_45/beta/m/Read/ReadVariableOp+Adam/conv2d_46/kernel/m/Read/ReadVariableOp)Adam/conv2d_46/bias/m/Read/ReadVariableOp7Adam/batch_normalization_46/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_46/beta/m/Read/ReadVariableOp+Adam/conv2d_47/kernel/m/Read/ReadVariableOp)Adam/conv2d_47/bias/m/Read/ReadVariableOp7Adam/batch_normalization_47/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_47/beta/m/Read/ReadVariableOp+Adam/conv2d_48/kernel/m/Read/ReadVariableOp)Adam/conv2d_48/bias/m/Read/ReadVariableOp7Adam/batch_normalization_48/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_48/beta/m/Read/ReadVariableOp5Adam/conv2d_transpose_18/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_18/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_19/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_19/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_20/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_20/bias/m/Read/ReadVariableOp)Adam/decoded/kernel/m/Read/ReadVariableOp'Adam/decoded/bias/m/Read/ReadVariableOp+Adam/conv2d_42/kernel/v/Read/ReadVariableOp)Adam/conv2d_42/bias/v/Read/ReadVariableOp7Adam/batch_normalization_42/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_42/beta/v/Read/ReadVariableOp+Adam/conv2d_43/kernel/v/Read/ReadVariableOp)Adam/conv2d_43/bias/v/Read/ReadVariableOp7Adam/batch_normalization_43/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_43/beta/v/Read/ReadVariableOp+Adam/conv2d_44/kernel/v/Read/ReadVariableOp)Adam/conv2d_44/bias/v/Read/ReadVariableOp7Adam/batch_normalization_44/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_44/beta/v/Read/ReadVariableOp+Adam/conv2d_45/kernel/v/Read/ReadVariableOp)Adam/conv2d_45/bias/v/Read/ReadVariableOp7Adam/batch_normalization_45/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_45/beta/v/Read/ReadVariableOp+Adam/conv2d_46/kernel/v/Read/ReadVariableOp)Adam/conv2d_46/bias/v/Read/ReadVariableOp7Adam/batch_normalization_46/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_46/beta/v/Read/ReadVariableOp+Adam/conv2d_47/kernel/v/Read/ReadVariableOp)Adam/conv2d_47/bias/v/Read/ReadVariableOp7Adam/batch_normalization_47/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_47/beta/v/Read/ReadVariableOp+Adam/conv2d_48/kernel/v/Read/ReadVariableOp)Adam/conv2d_48/bias/v/Read/ReadVariableOp7Adam/batch_normalization_48/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_48/beta/v/Read/ReadVariableOp5Adam/conv2d_transpose_18/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_18/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_19/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_19/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_20/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_20/bias/v/Read/ReadVariableOp)Adam/decoded/kernel/v/Read/ReadVariableOp'Adam/decoded/bias/v/Read/ReadVariableOpConst*?
Tin?
?2?	*
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
GPU2 *0J 8? *)
f$R"
 __inference__traced_save_2599274
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_42/kernelconv2d_42/biasbatch_normalization_42/gammabatch_normalization_42/beta"batch_normalization_42/moving_mean&batch_normalization_42/moving_varianceconv2d_43/kernelconv2d_43/biasbatch_normalization_43/gammabatch_normalization_43/beta"batch_normalization_43/moving_mean&batch_normalization_43/moving_varianceconv2d_44/kernelconv2d_44/biasbatch_normalization_44/gammabatch_normalization_44/beta"batch_normalization_44/moving_mean&batch_normalization_44/moving_varianceconv2d_45/kernelconv2d_45/biasbatch_normalization_45/gammabatch_normalization_45/beta"batch_normalization_45/moving_mean&batch_normalization_45/moving_varianceconv2d_46/kernelconv2d_46/biasbatch_normalization_46/gammabatch_normalization_46/beta"batch_normalization_46/moving_mean&batch_normalization_46/moving_varianceconv2d_47/kernelconv2d_47/biasbatch_normalization_47/gammabatch_normalization_47/beta"batch_normalization_47/moving_mean&batch_normalization_47/moving_varianceconv2d_48/kernelconv2d_48/biasbatch_normalization_48/gammabatch_normalization_48/beta"batch_normalization_48/moving_mean&batch_normalization_48/moving_varianceconv2d_transpose_18/kernelconv2d_transpose_18/biasconv2d_transpose_19/kernelconv2d_transpose_19/biasconv2d_transpose_20/kernelconv2d_transpose_20/biasdecoded/kerneldecoded/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_42/kernel/mAdam/conv2d_42/bias/m#Adam/batch_normalization_42/gamma/m"Adam/batch_normalization_42/beta/mAdam/conv2d_43/kernel/mAdam/conv2d_43/bias/m#Adam/batch_normalization_43/gamma/m"Adam/batch_normalization_43/beta/mAdam/conv2d_44/kernel/mAdam/conv2d_44/bias/m#Adam/batch_normalization_44/gamma/m"Adam/batch_normalization_44/beta/mAdam/conv2d_45/kernel/mAdam/conv2d_45/bias/m#Adam/batch_normalization_45/gamma/m"Adam/batch_normalization_45/beta/mAdam/conv2d_46/kernel/mAdam/conv2d_46/bias/m#Adam/batch_normalization_46/gamma/m"Adam/batch_normalization_46/beta/mAdam/conv2d_47/kernel/mAdam/conv2d_47/bias/m#Adam/batch_normalization_47/gamma/m"Adam/batch_normalization_47/beta/mAdam/conv2d_48/kernel/mAdam/conv2d_48/bias/m#Adam/batch_normalization_48/gamma/m"Adam/batch_normalization_48/beta/m!Adam/conv2d_transpose_18/kernel/mAdam/conv2d_transpose_18/bias/m!Adam/conv2d_transpose_19/kernel/mAdam/conv2d_transpose_19/bias/m!Adam/conv2d_transpose_20/kernel/mAdam/conv2d_transpose_20/bias/mAdam/decoded/kernel/mAdam/decoded/bias/mAdam/conv2d_42/kernel/vAdam/conv2d_42/bias/v#Adam/batch_normalization_42/gamma/v"Adam/batch_normalization_42/beta/vAdam/conv2d_43/kernel/vAdam/conv2d_43/bias/v#Adam/batch_normalization_43/gamma/v"Adam/batch_normalization_43/beta/vAdam/conv2d_44/kernel/vAdam/conv2d_44/bias/v#Adam/batch_normalization_44/gamma/v"Adam/batch_normalization_44/beta/vAdam/conv2d_45/kernel/vAdam/conv2d_45/bias/v#Adam/batch_normalization_45/gamma/v"Adam/batch_normalization_45/beta/vAdam/conv2d_46/kernel/vAdam/conv2d_46/bias/v#Adam/batch_normalization_46/gamma/v"Adam/batch_normalization_46/beta/vAdam/conv2d_47/kernel/vAdam/conv2d_47/bias/v#Adam/batch_normalization_47/gamma/v"Adam/batch_normalization_47/beta/vAdam/conv2d_48/kernel/vAdam/conv2d_48/bias/v#Adam/batch_normalization_48/gamma/v"Adam/batch_normalization_48/beta/v!Adam/conv2d_transpose_18/kernel/vAdam/conv2d_transpose_18/bias/v!Adam/conv2d_transpose_19/kernel/vAdam/conv2d_transpose_19/bias/v!Adam/conv2d_transpose_20/kernel/vAdam/conv2d_transpose_20/bias/vAdam/decoded/kernel/vAdam/decoded/bias/v*?
Tin?
?2?*
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
GPU2 *0J 8? *,
f'R%
#__inference__traced_restore_2599671??
?
f
J__inference_activation_62_layer_call_and_return_conditional_losses_2598821

inputs
identityQ
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:???????????i
IdentityIdentityLeakyRelu:activations:0*
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
?
K
/__inference_activation_55_layer_call_fn_2598205

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
GPU2 *0J 8? *S
fNRL
J__inference_activation_55_layer_call_and_return_conditional_losses_2596083j
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
?
?
+__inference_conv2d_45_layer_call_fn_2598310

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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_45_layer_call_and_return_conditional_losses_2596127w
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
F__inference_conv2d_46_layer_call_and_return_conditional_losses_2598411

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
r
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
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?

?
F__inference_conv2d_43_layer_call_and_return_conditional_losses_2598138

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
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
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
I__inference_sequential_6_layer_call_and_return_conditional_losses_2597105
conv2d_42_input+
conv2d_42_2596974:
conv2d_42_2596976:,
batch_normalization_42_2596979:,
batch_normalization_42_2596981:,
batch_normalization_42_2596983:,
batch_normalization_42_2596985:+
conv2d_43_2596989:
conv2d_43_2596991:,
batch_normalization_43_2596994:,
batch_normalization_43_2596996:,
batch_normalization_43_2596998:,
batch_normalization_43_2597000:+
conv2d_44_2597004:
conv2d_44_2597006:,
batch_normalization_44_2597009:,
batch_normalization_44_2597011:,
batch_normalization_44_2597013:,
batch_normalization_44_2597015:+
conv2d_45_2597019: 
conv2d_45_2597021: ,
batch_normalization_45_2597024: ,
batch_normalization_45_2597026: ,
batch_normalization_45_2597028: ,
batch_normalization_45_2597030: +
conv2d_46_2597034:  
conv2d_46_2597036: ,
batch_normalization_46_2597039: ,
batch_normalization_46_2597041: ,
batch_normalization_46_2597043: ,
batch_normalization_46_2597045: +
conv2d_47_2597049: @
conv2d_47_2597051:@,
batch_normalization_47_2597054:@,
batch_normalization_47_2597056:@,
batch_normalization_47_2597058:@,
batch_normalization_47_2597060:@+
conv2d_48_2597064:@ 
conv2d_48_2597066: ,
batch_normalization_48_2597069: ,
batch_normalization_48_2597071: ,
batch_normalization_48_2597073: ,
batch_normalization_48_2597075: 5
conv2d_transpose_18_2597081:@ )
conv2d_transpose_18_2597083:@5
conv2d_transpose_19_2597087: @)
conv2d_transpose_19_2597089: 5
conv2d_transpose_20_2597093: )
conv2d_transpose_20_2597095:)
decoded_2597099:
decoded_2597101:
identity??.batch_normalization_42/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_44/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?.batch_normalization_46/StatefulPartitionedCall?.batch_normalization_47/StatefulPartitionedCall?.batch_normalization_48/StatefulPartitionedCall?!conv2d_42/StatefulPartitionedCall?!conv2d_43/StatefulPartitionedCall?!conv2d_44/StatefulPartitionedCall?!conv2d_45/StatefulPartitionedCall?!conv2d_46/StatefulPartitionedCall?!conv2d_47/StatefulPartitionedCall?!conv2d_48/StatefulPartitionedCall?+conv2d_transpose_18/StatefulPartitionedCall?+conv2d_transpose_19/StatefulPartitionedCall?+conv2d_transpose_20/StatefulPartitionedCall?decoded/StatefulPartitionedCall?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCallconv2d_42_inputconv2d_42_2596974conv2d_42_2596976*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_42_layer_call_and_return_conditional_losses_2596031?
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0batch_normalization_42_2596979batch_normalization_42_2596981batch_normalization_42_2596983batch_normalization_42_2596985*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2595411?
activation_54/PartitionedCallPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_54_layer_call_and_return_conditional_losses_2596051?
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall&activation_54/PartitionedCall:output:0conv2d_43_2596989conv2d_43_2596991*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_43_layer_call_and_return_conditional_losses_2596063?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0batch_normalization_43_2596994batch_normalization_43_2596996batch_normalization_43_2596998batch_normalization_43_2597000*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2595475?
activation_55/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_55_layer_call_and_return_conditional_losses_2596083?
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCall&activation_55/PartitionedCall:output:0conv2d_44_2597004conv2d_44_2597006*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_44_layer_call_and_return_conditional_losses_2596095?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0batch_normalization_44_2597009batch_normalization_44_2597011batch_normalization_44_2597013batch_normalization_44_2597015*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2595539?
activation_56/PartitionedCallPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_56_layer_call_and_return_conditional_losses_2596115?
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCall&activation_56/PartitionedCall:output:0conv2d_45_2597019conv2d_45_2597021*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_45_layer_call_and_return_conditional_losses_2596127?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0batch_normalization_45_2597024batch_normalization_45_2597026batch_normalization_45_2597028batch_normalization_45_2597030*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2595603?
activation_57/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_57_layer_call_and_return_conditional_losses_2596147?
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCall&activation_57/PartitionedCall:output:0conv2d_46_2597034conv2d_46_2597036*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_46_layer_call_and_return_conditional_losses_2596159?
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0batch_normalization_46_2597039batch_normalization_46_2597041batch_normalization_46_2597043batch_normalization_46_2597045*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2595667?
activation_58/PartitionedCallPartitionedCall7batch_normalization_46/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_58_layer_call_and_return_conditional_losses_2596179?
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall&activation_58/PartitionedCall:output:0conv2d_47_2597049conv2d_47_2597051*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_47_layer_call_and_return_conditional_losses_2596191?
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0batch_normalization_47_2597054batch_normalization_47_2597056batch_normalization_47_2597058batch_normalization_47_2597060*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2595731?
activation_59/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_59_layer_call_and_return_conditional_losses_2596211?
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCall&activation_59/PartitionedCall:output:0conv2d_48_2597064conv2d_48_2597066*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_48_layer_call_and_return_conditional_losses_2596223?
.batch_normalization_48/StatefulPartitionedCallStatefulPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0batch_normalization_48_2597069batch_normalization_48_2597071batch_normalization_48_2597073batch_normalization_48_2597075*
Tin	
2*
Tout
2*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2595795?
encoded/CastCast7batch_normalization_48/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
encoded/PartitionedCallPartitionedCallencoded/Cast:y:0*
Tin
2*
Tout
2*
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
GPU2 *0J 8? *M
fHRF
D__inference_encoded_layer_call_and_return_conditional_losses_2596244?
conv2d_transpose_18/CastCast encoded/PartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
+conv2d_transpose_18/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_18/Cast:y:0conv2d_transpose_18_2597081conv2d_transpose_18_2597083*
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
GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_2595874?
activation_60/PartitionedCallPartitionedCall4conv2d_transpose_18/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_60_layer_call_and_return_conditional_losses_2596257?
+conv2d_transpose_19/StatefulPartitionedCallStatefulPartitionedCall&activation_60/PartitionedCall:output:0conv2d_transpose_19_2597087conv2d_transpose_19_2597089*
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
GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_2595918?
activation_61/PartitionedCallPartitionedCall4conv2d_transpose_19/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_61_layer_call_and_return_conditional_losses_2596269?
+conv2d_transpose_20/StatefulPartitionedCallStatefulPartitionedCall&activation_61/PartitionedCall:output:0conv2d_transpose_20_2597093conv2d_transpose_20_2597095*
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
GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_2595962?
activation_62/PartitionedCallPartitionedCall4conv2d_transpose_20/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_62_layer_call_and_return_conditional_losses_2596281?
decoded/StatefulPartitionedCallStatefulPartitionedCall&activation_62/PartitionedCall:output:0decoded_2597099decoded_2597101*
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
GPU2 *0J 8? *M
fHRF
D__inference_decoded_layer_call_and_return_conditional_losses_2596007?
IdentityIdentity(decoded/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_46/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_48/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall"^conv2d_44/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall,^conv2d_transpose_18/StatefulPartitionedCall,^conv2d_transpose_19/StatefulPartitionedCall,^conv2d_transpose_20/StatefulPartitionedCall ^decoded/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_46/StatefulPartitionedCall.batch_normalization_46/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_48/StatefulPartitionedCall.batch_normalization_48/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2Z
+conv2d_transpose_18/StatefulPartitionedCall+conv2d_transpose_18/StatefulPartitionedCall2Z
+conv2d_transpose_19/StatefulPartitionedCall+conv2d_transpose_19/StatefulPartitionedCall2Z
+conv2d_transpose_20/StatefulPartitionedCall+conv2d_transpose_20/StatefulPartitionedCall2B
decoded/StatefulPartitionedCalldecoded/StatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_42_input
??
?7
"__inference__wrapped_model_2595389
conv2d_42_inputO
5sequential_6_conv2d_42_conv2d_readvariableop_resource:D
6sequential_6_conv2d_42_biasadd_readvariableop_resource:I
;sequential_6_batch_normalization_42_readvariableop_resource:K
=sequential_6_batch_normalization_42_readvariableop_1_resource:Z
Lsequential_6_batch_normalization_42_fusedbatchnormv3_readvariableop_resource:\
Nsequential_6_batch_normalization_42_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_6_conv2d_43_conv2d_readvariableop_resource:D
6sequential_6_conv2d_43_biasadd_readvariableop_resource:I
;sequential_6_batch_normalization_43_readvariableop_resource:K
=sequential_6_batch_normalization_43_readvariableop_1_resource:Z
Lsequential_6_batch_normalization_43_fusedbatchnormv3_readvariableop_resource:\
Nsequential_6_batch_normalization_43_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_6_conv2d_44_conv2d_readvariableop_resource:D
6sequential_6_conv2d_44_biasadd_readvariableop_resource:I
;sequential_6_batch_normalization_44_readvariableop_resource:K
=sequential_6_batch_normalization_44_readvariableop_1_resource:Z
Lsequential_6_batch_normalization_44_fusedbatchnormv3_readvariableop_resource:\
Nsequential_6_batch_normalization_44_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_6_conv2d_45_conv2d_readvariableop_resource: D
6sequential_6_conv2d_45_biasadd_readvariableop_resource: I
;sequential_6_batch_normalization_45_readvariableop_resource: K
=sequential_6_batch_normalization_45_readvariableop_1_resource: Z
Lsequential_6_batch_normalization_45_fusedbatchnormv3_readvariableop_resource: \
Nsequential_6_batch_normalization_45_fusedbatchnormv3_readvariableop_1_resource: O
5sequential_6_conv2d_46_conv2d_readvariableop_resource:  D
6sequential_6_conv2d_46_biasadd_readvariableop_resource: I
;sequential_6_batch_normalization_46_readvariableop_resource: K
=sequential_6_batch_normalization_46_readvariableop_1_resource: Z
Lsequential_6_batch_normalization_46_fusedbatchnormv3_readvariableop_resource: \
Nsequential_6_batch_normalization_46_fusedbatchnormv3_readvariableop_1_resource: O
5sequential_6_conv2d_47_conv2d_readvariableop_resource: @D
6sequential_6_conv2d_47_biasadd_readvariableop_resource:@I
;sequential_6_batch_normalization_47_readvariableop_resource:@K
=sequential_6_batch_normalization_47_readvariableop_1_resource:@Z
Lsequential_6_batch_normalization_47_fusedbatchnormv3_readvariableop_resource:@\
Nsequential_6_batch_normalization_47_fusedbatchnormv3_readvariableop_1_resource:@O
5sequential_6_conv2d_48_conv2d_readvariableop_resource:@ D
6sequential_6_conv2d_48_biasadd_readvariableop_resource: I
;sequential_6_batch_normalization_48_readvariableop_resource: K
=sequential_6_batch_normalization_48_readvariableop_1_resource: Z
Lsequential_6_batch_normalization_48_fusedbatchnormv3_readvariableop_resource: \
Nsequential_6_batch_normalization_48_fusedbatchnormv3_readvariableop_1_resource: c
Isequential_6_conv2d_transpose_18_conv2d_transpose_readvariableop_resource:@ N
@sequential_6_conv2d_transpose_18_biasadd_readvariableop_resource:@c
Isequential_6_conv2d_transpose_19_conv2d_transpose_readvariableop_resource: @N
@sequential_6_conv2d_transpose_19_biasadd_readvariableop_resource: c
Isequential_6_conv2d_transpose_20_conv2d_transpose_readvariableop_resource: N
@sequential_6_conv2d_transpose_20_biasadd_readvariableop_resource:W
=sequential_6_decoded_conv2d_transpose_readvariableop_resource:B
4sequential_6_decoded_biasadd_readvariableop_resource:
identity??Csequential_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOp?Esequential_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1?2sequential_6/batch_normalization_42/ReadVariableOp?4sequential_6/batch_normalization_42/ReadVariableOp_1?Csequential_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOp?Esequential_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?2sequential_6/batch_normalization_43/ReadVariableOp?4sequential_6/batch_normalization_43/ReadVariableOp_1?Csequential_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOp?Esequential_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1?2sequential_6/batch_normalization_44/ReadVariableOp?4sequential_6/batch_normalization_44/ReadVariableOp_1?Csequential_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOp?Esequential_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?2sequential_6/batch_normalization_45/ReadVariableOp?4sequential_6/batch_normalization_45/ReadVariableOp_1?Csequential_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOp?Esequential_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1?2sequential_6/batch_normalization_46/ReadVariableOp?4sequential_6/batch_normalization_46/ReadVariableOp_1?Csequential_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOp?Esequential_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1?2sequential_6/batch_normalization_47/ReadVariableOp?4sequential_6/batch_normalization_47/ReadVariableOp_1?Csequential_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOp?Esequential_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?2sequential_6/batch_normalization_48/ReadVariableOp?4sequential_6/batch_normalization_48/ReadVariableOp_1?-sequential_6/conv2d_42/BiasAdd/ReadVariableOp?,sequential_6/conv2d_42/Conv2D/ReadVariableOp?-sequential_6/conv2d_43/BiasAdd/ReadVariableOp?,sequential_6/conv2d_43/Conv2D/ReadVariableOp?-sequential_6/conv2d_44/BiasAdd/ReadVariableOp?,sequential_6/conv2d_44/Conv2D/ReadVariableOp?-sequential_6/conv2d_45/BiasAdd/ReadVariableOp?,sequential_6/conv2d_45/Conv2D/ReadVariableOp?-sequential_6/conv2d_46/BiasAdd/ReadVariableOp?,sequential_6/conv2d_46/Conv2D/ReadVariableOp?-sequential_6/conv2d_47/BiasAdd/ReadVariableOp?,sequential_6/conv2d_47/Conv2D/ReadVariableOp?-sequential_6/conv2d_48/BiasAdd/ReadVariableOp?,sequential_6/conv2d_48/Conv2D/ReadVariableOp?7sequential_6/conv2d_transpose_18/BiasAdd/ReadVariableOp?@sequential_6/conv2d_transpose_18/conv2d_transpose/ReadVariableOp?7sequential_6/conv2d_transpose_19/BiasAdd/ReadVariableOp?@sequential_6/conv2d_transpose_19/conv2d_transpose/ReadVariableOp?7sequential_6/conv2d_transpose_20/BiasAdd/ReadVariableOp?@sequential_6/conv2d_transpose_20/conv2d_transpose/ReadVariableOp?+sequential_6/decoded/BiasAdd/ReadVariableOp?4sequential_6/decoded/conv2d_transpose/ReadVariableOp?
,sequential_6/conv2d_42/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_6/conv2d_42/Conv2DConv2Dconv2d_42_input4sequential_6/conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
-sequential_6/conv2d_42/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_6/conv2d_42/BiasAddBiasAdd&sequential_6/conv2d_42/Conv2D:output:05sequential_6/conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
2sequential_6/batch_normalization_42/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_42_readvariableop_resource*
_output_shapes
:*
dtype0?
4sequential_6/batch_normalization_42/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_42_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Csequential_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_42_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Esequential_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_42_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
4sequential_6/batch_normalization_42/FusedBatchNormV3FusedBatchNormV3'sequential_6/conv2d_42/BiasAdd:output:0:sequential_6/batch_normalization_42/ReadVariableOp:value:0<sequential_6/batch_normalization_42/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
$sequential_6/activation_54/LeakyRelu	LeakyRelu8sequential_6/batch_normalization_42/FusedBatchNormV3:y:0*1
_output_shapes
:????????????
,sequential_6/conv2d_43/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_43_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_6/conv2d_43/Conv2DConv2D2sequential_6/activation_54/LeakyRelu:activations:04sequential_6/conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
-sequential_6/conv2d_43/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_6/conv2d_43/BiasAddBiasAdd&sequential_6/conv2d_43/Conv2D:output:05sequential_6/conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
2sequential_6/batch_normalization_43/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_43_readvariableop_resource*
_output_shapes
:*
dtype0?
4sequential_6/batch_normalization_43/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_43_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Csequential_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_43_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Esequential_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_43_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
4sequential_6/batch_normalization_43/FusedBatchNormV3FusedBatchNormV3'sequential_6/conv2d_43/BiasAdd:output:0:sequential_6/batch_normalization_43/ReadVariableOp:value:0<sequential_6/batch_normalization_43/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
$sequential_6/activation_55/LeakyRelu	LeakyRelu8sequential_6/batch_normalization_43/FusedBatchNormV3:y:0*1
_output_shapes
:????????????
,sequential_6/conv2d_44/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_44_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_6/conv2d_44/Conv2DConv2D2sequential_6/activation_55/LeakyRelu:activations:04sequential_6/conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
-sequential_6/conv2d_44/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_6/conv2d_44/BiasAddBiasAdd&sequential_6/conv2d_44/Conv2D:output:05sequential_6/conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
2sequential_6/batch_normalization_44/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_44_readvariableop_resource*
_output_shapes
:*
dtype0?
4sequential_6/batch_normalization_44/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_44_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Csequential_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_44_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Esequential_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_44_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
4sequential_6/batch_normalization_44/FusedBatchNormV3FusedBatchNormV3'sequential_6/conv2d_44/BiasAdd:output:0:sequential_6/batch_normalization_44/ReadVariableOp:value:0<sequential_6/batch_normalization_44/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
$sequential_6/activation_56/LeakyRelu	LeakyRelu8sequential_6/batch_normalization_44/FusedBatchNormV3:y:0*1
_output_shapes
:????????????
,sequential_6/conv2d_45/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_45_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
sequential_6/conv2d_45/Conv2DConv2D2sequential_6/activation_56/LeakyRelu:activations:04sequential_6/conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
-sequential_6/conv2d_45/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_45_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_6/conv2d_45/BiasAddBiasAdd&sequential_6/conv2d_45/Conv2D:output:05sequential_6/conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
2sequential_6/batch_normalization_45/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_45_readvariableop_resource*
_output_shapes
: *
dtype0?
4sequential_6/batch_normalization_45/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_45_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Csequential_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_45_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Esequential_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_45_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
4sequential_6/batch_normalization_45/FusedBatchNormV3FusedBatchNormV3'sequential_6/conv2d_45/BiasAdd:output:0:sequential_6/batch_normalization_45/ReadVariableOp:value:0<sequential_6/batch_normalization_45/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( ?
$sequential_6/activation_57/LeakyRelu	LeakyRelu8sequential_6/batch_normalization_45/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@ ?
,sequential_6/conv2d_46/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
sequential_6/conv2d_46/Conv2DConv2D2sequential_6/activation_57/LeakyRelu:activations:04sequential_6/conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
-sequential_6/conv2d_46/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_46_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_6/conv2d_46/BiasAddBiasAdd&sequential_6/conv2d_46/Conv2D:output:05sequential_6/conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
2sequential_6/batch_normalization_46/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_46_readvariableop_resource*
_output_shapes
: *
dtype0?
4sequential_6/batch_normalization_46/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_46_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Csequential_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_46_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Esequential_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_46_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
4sequential_6/batch_normalization_46/FusedBatchNormV3FusedBatchNormV3'sequential_6/conv2d_46/BiasAdd:output:0:sequential_6/batch_normalization_46/ReadVariableOp:value:0<sequential_6/batch_normalization_46/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( ?
$sequential_6/activation_58/LeakyRelu	LeakyRelu8sequential_6/batch_normalization_46/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@ ?
,sequential_6/conv2d_47/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_47_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
sequential_6/conv2d_47/Conv2DConv2D2sequential_6/activation_58/LeakyRelu:activations:04sequential_6/conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
-sequential_6/conv2d_47/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_47_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_6/conv2d_47/BiasAddBiasAdd&sequential_6/conv2d_47/Conv2D:output:05sequential_6/conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
2sequential_6/batch_normalization_47/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_47_readvariableop_resource*
_output_shapes
:@*
dtype0?
4sequential_6/batch_normalization_47/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_47_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Csequential_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Esequential_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
4sequential_6/batch_normalization_47/FusedBatchNormV3FusedBatchNormV3'sequential_6/conv2d_47/BiasAdd:output:0:sequential_6/batch_normalization_47/ReadVariableOp:value:0<sequential_6/batch_normalization_47/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( ?
$sequential_6/activation_59/LeakyRelu	LeakyRelu8sequential_6/batch_normalization_47/FusedBatchNormV3:y:0*/
_output_shapes
:?????????  @?
,sequential_6/conv2d_48/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
sequential_6/conv2d_48/Conv2DConv2D2sequential_6/activation_59/LeakyRelu:activations:04sequential_6/conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
-sequential_6/conv2d_48/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_48_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_6/conv2d_48/BiasAddBiasAdd&sequential_6/conv2d_48/Conv2D:output:05sequential_6/conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
2sequential_6/batch_normalization_48/ReadVariableOpReadVariableOp;sequential_6_batch_normalization_48_readvariableop_resource*
_output_shapes
: *
dtype0?
4sequential_6/batch_normalization_48/ReadVariableOp_1ReadVariableOp=sequential_6_batch_normalization_48_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Csequential_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_6_batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Esequential_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_6_batch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
4sequential_6/batch_normalization_48/FusedBatchNormV3FusedBatchNormV3'sequential_6/conv2d_48/BiasAdd:output:0:sequential_6/batch_normalization_48/ReadVariableOp:value:0<sequential_6/batch_normalization_48/ReadVariableOp_1:value:0Ksequential_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0Msequential_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
sequential_6/encoded/CastCast8sequential_6/batch_normalization_48/FusedBatchNormV3:y:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
sequential_6/encoded/LeakyRelu	LeakyRelusequential_6/encoded/Cast:y:0*
T0*/
_output_shapes
:????????? ?
%sequential_6/conv2d_transpose_18/CastCast,sequential_6/encoded/LeakyRelu:activations:0*

DstT0*

SrcT0*/
_output_shapes
:????????? 
&sequential_6/conv2d_transpose_18/ShapeShape)sequential_6/conv2d_transpose_18/Cast:y:0*
T0*
_output_shapes
:~
4sequential_6/conv2d_transpose_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6sequential_6/conv2d_transpose_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential_6/conv2d_transpose_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.sequential_6/conv2d_transpose_18/strided_sliceStridedSlice/sequential_6/conv2d_transpose_18/Shape:output:0=sequential_6/conv2d_transpose_18/strided_slice/stack:output:0?sequential_6/conv2d_transpose_18/strided_slice/stack_1:output:0?sequential_6/conv2d_transpose_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_6/conv2d_transpose_18/stack/1Const*
_output_shapes
: *
dtype0*
value	B : j
(sequential_6/conv2d_transpose_18/stack/2Const*
_output_shapes
: *
dtype0*
value	B : j
(sequential_6/conv2d_transpose_18/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
&sequential_6/conv2d_transpose_18/stackPack7sequential_6/conv2d_transpose_18/strided_slice:output:01sequential_6/conv2d_transpose_18/stack/1:output:01sequential_6/conv2d_transpose_18/stack/2:output:01sequential_6/conv2d_transpose_18/stack/3:output:0*
N*
T0*
_output_shapes
:?
6sequential_6/conv2d_transpose_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8sequential_6/conv2d_transpose_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_6/conv2d_transpose_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0sequential_6/conv2d_transpose_18/strided_slice_1StridedSlice/sequential_6/conv2d_transpose_18/stack:output:0?sequential_6/conv2d_transpose_18/strided_slice_1/stack:output:0Asequential_6/conv2d_transpose_18/strided_slice_1/stack_1:output:0Asequential_6/conv2d_transpose_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_6/conv2d_transpose_18/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_6_conv2d_transpose_18_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
1sequential_6/conv2d_transpose_18/conv2d_transposeConv2DBackpropInput/sequential_6/conv2d_transpose_18/stack:output:0Hsequential_6/conv2d_transpose_18/conv2d_transpose/ReadVariableOp:value:0)sequential_6/conv2d_transpose_18/Cast:y:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
7sequential_6/conv2d_transpose_18/BiasAdd/ReadVariableOpReadVariableOp@sequential_6_conv2d_transpose_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
(sequential_6/conv2d_transpose_18/BiasAddBiasAdd:sequential_6/conv2d_transpose_18/conv2d_transpose:output:0?sequential_6/conv2d_transpose_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
$sequential_6/activation_60/LeakyRelu	LeakyRelu1sequential_6/conv2d_transpose_18/BiasAdd:output:0*/
_output_shapes
:?????????  @?
&sequential_6/conv2d_transpose_19/ShapeShape2sequential_6/activation_60/LeakyRelu:activations:0*
T0*
_output_shapes
:~
4sequential_6/conv2d_transpose_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6sequential_6/conv2d_transpose_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential_6/conv2d_transpose_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.sequential_6/conv2d_transpose_19/strided_sliceStridedSlice/sequential_6/conv2d_transpose_19/Shape:output:0=sequential_6/conv2d_transpose_19/strided_slice/stack:output:0?sequential_6/conv2d_transpose_19/strided_slice/stack_1:output:0?sequential_6/conv2d_transpose_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_6/conv2d_transpose_19/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@j
(sequential_6/conv2d_transpose_19/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@j
(sequential_6/conv2d_transpose_19/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_6/conv2d_transpose_19/stackPack7sequential_6/conv2d_transpose_19/strided_slice:output:01sequential_6/conv2d_transpose_19/stack/1:output:01sequential_6/conv2d_transpose_19/stack/2:output:01sequential_6/conv2d_transpose_19/stack/3:output:0*
N*
T0*
_output_shapes
:?
6sequential_6/conv2d_transpose_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8sequential_6/conv2d_transpose_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_6/conv2d_transpose_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0sequential_6/conv2d_transpose_19/strided_slice_1StridedSlice/sequential_6/conv2d_transpose_19/stack:output:0?sequential_6/conv2d_transpose_19/strided_slice_1/stack:output:0Asequential_6/conv2d_transpose_19/strided_slice_1/stack_1:output:0Asequential_6/conv2d_transpose_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_6/conv2d_transpose_19/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_6_conv2d_transpose_19_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
1sequential_6/conv2d_transpose_19/conv2d_transposeConv2DBackpropInput/sequential_6/conv2d_transpose_19/stack:output:0Hsequential_6/conv2d_transpose_19/conv2d_transpose/ReadVariableOp:value:02sequential_6/activation_60/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
7sequential_6/conv2d_transpose_19/BiasAdd/ReadVariableOpReadVariableOp@sequential_6_conv2d_transpose_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
(sequential_6/conv2d_transpose_19/BiasAddBiasAdd:sequential_6/conv2d_transpose_19/conv2d_transpose:output:0?sequential_6/conv2d_transpose_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
$sequential_6/activation_61/LeakyRelu	LeakyRelu1sequential_6/conv2d_transpose_19/BiasAdd:output:0*/
_output_shapes
:?????????@@ ?
&sequential_6/conv2d_transpose_20/ShapeShape2sequential_6/activation_61/LeakyRelu:activations:0*
T0*
_output_shapes
:~
4sequential_6/conv2d_transpose_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6sequential_6/conv2d_transpose_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential_6/conv2d_transpose_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.sequential_6/conv2d_transpose_20/strided_sliceStridedSlice/sequential_6/conv2d_transpose_20/Shape:output:0=sequential_6/conv2d_transpose_20/strided_slice/stack:output:0?sequential_6/conv2d_transpose_20/strided_slice/stack_1:output:0?sequential_6/conv2d_transpose_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
(sequential_6/conv2d_transpose_20/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?k
(sequential_6/conv2d_transpose_20/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?j
(sequential_6/conv2d_transpose_20/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
&sequential_6/conv2d_transpose_20/stackPack7sequential_6/conv2d_transpose_20/strided_slice:output:01sequential_6/conv2d_transpose_20/stack/1:output:01sequential_6/conv2d_transpose_20/stack/2:output:01sequential_6/conv2d_transpose_20/stack/3:output:0*
N*
T0*
_output_shapes
:?
6sequential_6/conv2d_transpose_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8sequential_6/conv2d_transpose_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_6/conv2d_transpose_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0sequential_6/conv2d_transpose_20/strided_slice_1StridedSlice/sequential_6/conv2d_transpose_20/stack:output:0?sequential_6/conv2d_transpose_20/strided_slice_1/stack:output:0Asequential_6/conv2d_transpose_20/strided_slice_1/stack_1:output:0Asequential_6/conv2d_transpose_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_6/conv2d_transpose_20/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_6_conv2d_transpose_20_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
1sequential_6/conv2d_transpose_20/conv2d_transposeConv2DBackpropInput/sequential_6/conv2d_transpose_20/stack:output:0Hsequential_6/conv2d_transpose_20/conv2d_transpose/ReadVariableOp:value:02sequential_6/activation_61/LeakyRelu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
7sequential_6/conv2d_transpose_20/BiasAdd/ReadVariableOpReadVariableOp@sequential_6_conv2d_transpose_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
(sequential_6/conv2d_transpose_20/BiasAddBiasAdd:sequential_6/conv2d_transpose_20/conv2d_transpose:output:0?sequential_6/conv2d_transpose_20/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
$sequential_6/activation_62/LeakyRelu	LeakyRelu1sequential_6/conv2d_transpose_20/BiasAdd:output:0*1
_output_shapes
:???????????|
sequential_6/decoded/ShapeShape2sequential_6/activation_62/LeakyRelu:activations:0*
T0*
_output_shapes
:r
(sequential_6/decoded/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_6/decoded/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_6/decoded/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"sequential_6/decoded/strided_sliceStridedSlice#sequential_6/decoded/Shape:output:01sequential_6/decoded/strided_slice/stack:output:03sequential_6/decoded/strided_slice/stack_1:output:03sequential_6/decoded/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
sequential_6/decoded/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?_
sequential_6/decoded/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?^
sequential_6/decoded/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
sequential_6/decoded/stackPack+sequential_6/decoded/strided_slice:output:0%sequential_6/decoded/stack/1:output:0%sequential_6/decoded/stack/2:output:0%sequential_6/decoded/stack/3:output:0*
N*
T0*
_output_shapes
:t
*sequential_6/decoded/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_6/decoded/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_6/decoded/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$sequential_6/decoded/strided_slice_1StridedSlice#sequential_6/decoded/stack:output:03sequential_6/decoded/strided_slice_1/stack:output:05sequential_6/decoded/strided_slice_1/stack_1:output:05sequential_6/decoded/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
4sequential_6/decoded/conv2d_transpose/ReadVariableOpReadVariableOp=sequential_6_decoded_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
%sequential_6/decoded/conv2d_transposeConv2DBackpropInput#sequential_6/decoded/stack:output:0<sequential_6/decoded/conv2d_transpose/ReadVariableOp:value:02sequential_6/activation_62/LeakyRelu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
+sequential_6/decoded/BiasAdd/ReadVariableOpReadVariableOp4sequential_6_decoded_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_6/decoded/BiasAddBiasAdd.sequential_6/decoded/conv2d_transpose:output:03sequential_6/decoded/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
sequential_6/decoded/TanhTanh%sequential_6/decoded/BiasAdd:output:0*
T0*1
_output_shapes
:???????????v
IdentityIdentitysequential_6/decoded/Tanh:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOpD^sequential_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_42/ReadVariableOp5^sequential_6/batch_normalization_42/ReadVariableOp_1D^sequential_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_43/ReadVariableOp5^sequential_6/batch_normalization_43/ReadVariableOp_1D^sequential_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_44/ReadVariableOp5^sequential_6/batch_normalization_44/ReadVariableOp_1D^sequential_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_45/ReadVariableOp5^sequential_6/batch_normalization_45/ReadVariableOp_1D^sequential_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_46/ReadVariableOp5^sequential_6/batch_normalization_46/ReadVariableOp_1D^sequential_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_47/ReadVariableOp5^sequential_6/batch_normalization_47/ReadVariableOp_1D^sequential_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOpF^sequential_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_13^sequential_6/batch_normalization_48/ReadVariableOp5^sequential_6/batch_normalization_48/ReadVariableOp_1.^sequential_6/conv2d_42/BiasAdd/ReadVariableOp-^sequential_6/conv2d_42/Conv2D/ReadVariableOp.^sequential_6/conv2d_43/BiasAdd/ReadVariableOp-^sequential_6/conv2d_43/Conv2D/ReadVariableOp.^sequential_6/conv2d_44/BiasAdd/ReadVariableOp-^sequential_6/conv2d_44/Conv2D/ReadVariableOp.^sequential_6/conv2d_45/BiasAdd/ReadVariableOp-^sequential_6/conv2d_45/Conv2D/ReadVariableOp.^sequential_6/conv2d_46/BiasAdd/ReadVariableOp-^sequential_6/conv2d_46/Conv2D/ReadVariableOp.^sequential_6/conv2d_47/BiasAdd/ReadVariableOp-^sequential_6/conv2d_47/Conv2D/ReadVariableOp.^sequential_6/conv2d_48/BiasAdd/ReadVariableOp-^sequential_6/conv2d_48/Conv2D/ReadVariableOp8^sequential_6/conv2d_transpose_18/BiasAdd/ReadVariableOpA^sequential_6/conv2d_transpose_18/conv2d_transpose/ReadVariableOp8^sequential_6/conv2d_transpose_19/BiasAdd/ReadVariableOpA^sequential_6/conv2d_transpose_19/conv2d_transpose/ReadVariableOp8^sequential_6/conv2d_transpose_20/BiasAdd/ReadVariableOpA^sequential_6/conv2d_transpose_20/conv2d_transpose/ReadVariableOp,^sequential_6/decoded/BiasAdd/ReadVariableOp5^sequential_6/decoded/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2?
Csequential_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOp2?
Esequential_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_42/ReadVariableOp2sequential_6/batch_normalization_42/ReadVariableOp2l
4sequential_6/batch_normalization_42/ReadVariableOp_14sequential_6/batch_normalization_42/ReadVariableOp_12?
Csequential_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOp2?
Esequential_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_43/ReadVariableOp2sequential_6/batch_normalization_43/ReadVariableOp2l
4sequential_6/batch_normalization_43/ReadVariableOp_14sequential_6/batch_normalization_43/ReadVariableOp_12?
Csequential_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOp2?
Esequential_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_44/ReadVariableOp2sequential_6/batch_normalization_44/ReadVariableOp2l
4sequential_6/batch_normalization_44/ReadVariableOp_14sequential_6/batch_normalization_44/ReadVariableOp_12?
Csequential_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOp2?
Esequential_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_45/ReadVariableOp2sequential_6/batch_normalization_45/ReadVariableOp2l
4sequential_6/batch_normalization_45/ReadVariableOp_14sequential_6/batch_normalization_45/ReadVariableOp_12?
Csequential_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOp2?
Esequential_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_46/ReadVariableOp2sequential_6/batch_normalization_46/ReadVariableOp2l
4sequential_6/batch_normalization_46/ReadVariableOp_14sequential_6/batch_normalization_46/ReadVariableOp_12?
Csequential_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOp2?
Esequential_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_47/ReadVariableOp2sequential_6/batch_normalization_47/ReadVariableOp2l
4sequential_6/batch_normalization_47/ReadVariableOp_14sequential_6/batch_normalization_47/ReadVariableOp_12?
Csequential_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOpCsequential_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOp2?
Esequential_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1Esequential_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12h
2sequential_6/batch_normalization_48/ReadVariableOp2sequential_6/batch_normalization_48/ReadVariableOp2l
4sequential_6/batch_normalization_48/ReadVariableOp_14sequential_6/batch_normalization_48/ReadVariableOp_12^
-sequential_6/conv2d_42/BiasAdd/ReadVariableOp-sequential_6/conv2d_42/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_42/Conv2D/ReadVariableOp,sequential_6/conv2d_42/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_43/BiasAdd/ReadVariableOp-sequential_6/conv2d_43/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_43/Conv2D/ReadVariableOp,sequential_6/conv2d_43/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_44/BiasAdd/ReadVariableOp-sequential_6/conv2d_44/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_44/Conv2D/ReadVariableOp,sequential_6/conv2d_44/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_45/BiasAdd/ReadVariableOp-sequential_6/conv2d_45/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_45/Conv2D/ReadVariableOp,sequential_6/conv2d_45/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_46/BiasAdd/ReadVariableOp-sequential_6/conv2d_46/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_46/Conv2D/ReadVariableOp,sequential_6/conv2d_46/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_47/BiasAdd/ReadVariableOp-sequential_6/conv2d_47/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_47/Conv2D/ReadVariableOp,sequential_6/conv2d_47/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_48/BiasAdd/ReadVariableOp-sequential_6/conv2d_48/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_48/Conv2D/ReadVariableOp,sequential_6/conv2d_48/Conv2D/ReadVariableOp2r
7sequential_6/conv2d_transpose_18/BiasAdd/ReadVariableOp7sequential_6/conv2d_transpose_18/BiasAdd/ReadVariableOp2?
@sequential_6/conv2d_transpose_18/conv2d_transpose/ReadVariableOp@sequential_6/conv2d_transpose_18/conv2d_transpose/ReadVariableOp2r
7sequential_6/conv2d_transpose_19/BiasAdd/ReadVariableOp7sequential_6/conv2d_transpose_19/BiasAdd/ReadVariableOp2?
@sequential_6/conv2d_transpose_19/conv2d_transpose/ReadVariableOp@sequential_6/conv2d_transpose_19/conv2d_transpose/ReadVariableOp2r
7sequential_6/conv2d_transpose_20/BiasAdd/ReadVariableOp7sequential_6/conv2d_transpose_20/BiasAdd/ReadVariableOp2?
@sequential_6/conv2d_transpose_20/conv2d_transpose/ReadVariableOp@sequential_6/conv2d_transpose_20/conv2d_transpose/ReadVariableOp2Z
+sequential_6/decoded/BiasAdd/ReadVariableOp+sequential_6/decoded/BiasAdd/ReadVariableOp2l
4sequential_6/decoded/conv2d_transpose/ReadVariableOp4sequential_6/decoded/conv2d_transpose/ReadVariableOp:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_42_input
?
f
J__inference_activation_54_layer_call_and_return_conditional_losses_2598119

inputs
identityQ
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:???????????i
IdentityIdentityLeakyRelu:activations:0*
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
F__inference_conv2d_47_layer_call_and_return_conditional_losses_2596191

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
?
?
.__inference_sequential_6_layer_call_fn_2597457

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23:  

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: $

unknown_29: @

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@$

unknown_35:@ 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: $

unknown_41:@ 

unknown_42:@$

unknown_43: @

unknown_44: $

unknown_45: 

unknown_46:$

unknown_47:

unknown_48:
identity??StatefulPartitionedCall?
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
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_2596289y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2595570

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
?
f
J__inference_activation_62_layer_call_and_return_conditional_losses_2596281

inputs
identityQ
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:???????????i
IdentityIdentityLeakyRelu:activations:0*
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
?
?
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2598200

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
?
?
)__inference_decoded_layer_call_fn_2598830

inputs!
unknown:
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
GPU2 *0J 8? *M
fHRF
D__inference_decoded_layer_call_and_return_conditional_losses_2596007?
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
?
f
J__inference_activation_61_layer_call_and_return_conditional_losses_2596269

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@@ g
IdentityIdentityLeakyRelu:activations:0*
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
?
?
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2595826

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
f
J__inference_activation_56_layer_call_and_return_conditional_losses_2596115

inputs
identityQ
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:???????????i
IdentityIdentityLeakyRelu:activations:0*
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
?
K
/__inference_activation_57_layer_call_fn_2598387

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
GPU2 *0J 8? *S
fNRL
J__inference_activation_57_layer_call_and_return_conditional_losses_2596147h
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
?
f
J__inference_activation_55_layer_call_and_return_conditional_losses_2598210

inputs
identityQ
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:???????????i
IdentityIdentityLeakyRelu:activations:0*
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
?
?
5__inference_conv2d_transpose_19_layer_call_fn_2598726

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
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_2595918?
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
?	
?
8__inference_batch_normalization_45_layer_call_fn_2598333

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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2595603?
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
?
?
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2598291

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
?
K
/__inference_activation_58_layer_call_fn_2598478

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
GPU2 *0J 8? *S
fNRL
J__inference_activation_58_layer_call_and_return_conditional_losses_2596179h
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
?
?
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2595506

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
8__inference_batch_normalization_44_layer_call_fn_2598255

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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2595570?
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
?
E
)__inference_encoded_layer_call_fn_2598660

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
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
GPU2 *0J 8? *M
fHRF
D__inference_encoded_layer_call_and_return_conditional_losses_2596244h
IdentityIdentityPartitionedCall:output:0*
T0*/
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
?
?
+__inference_conv2d_43_layer_call_fn_2598128

inputs!
unknown:
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_43_layer_call_and_return_conditional_losses_2596063y
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
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
? 
?
P__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_2598811

inputsB
(conv2d_transpose_readvariableop_resource: -
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
: *
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
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
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
?
?
5__inference_conv2d_transpose_20_layer_call_fn_2598778

inputs!
unknown: 
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
P__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_2595962?
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
?
f
J__inference_activation_56_layer_call_and_return_conditional_losses_2598301

inputs
identityQ
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:???????????i
IdentityIdentityLeakyRelu:activations:0*
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
?
f
J__inference_activation_58_layer_call_and_return_conditional_losses_2596179

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@@ g
IdentityIdentityLeakyRelu:activations:0*
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
8__inference_batch_normalization_42_layer_call_fn_2598073

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
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2595442?
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
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2598455

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
?
?
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2598637

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
F__inference_conv2d_48_layer_call_and_return_conditional_losses_2596223

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
?
?
+__inference_conv2d_46_layer_call_fn_2598401

inputs!
unknown:  
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_46_layer_call_and_return_conditional_losses_2596159w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@ `
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
?
?
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2598364

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
?
?
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2595667

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
8__inference_batch_normalization_48_layer_call_fn_2598619

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
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2595826?
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
?
?
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2595442

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
?
?
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2595603

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
8__inference_batch_normalization_42_layer_call_fn_2598060

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
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2595411?
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
?
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_2595874

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
-:+???????????????????????????@y
IdentityIdentityBiasAdd:output:0^NoOp*
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
?
f
J__inference_activation_61_layer_call_and_return_conditional_losses_2598769

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@@ g
IdentityIdentityLeakyRelu:activations:0*
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
?
?
+__inference_conv2d_42_layer_call_fn_2598037

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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_42_layer_call_and_return_conditional_losses_2596031y
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
??
?X
#__inference__traced_restore_2599671
file_prefix;
!assignvariableop_conv2d_42_kernel:/
!assignvariableop_1_conv2d_42_bias:=
/assignvariableop_2_batch_normalization_42_gamma:<
.assignvariableop_3_batch_normalization_42_beta:C
5assignvariableop_4_batch_normalization_42_moving_mean:G
9assignvariableop_5_batch_normalization_42_moving_variance:=
#assignvariableop_6_conv2d_43_kernel:/
!assignvariableop_7_conv2d_43_bias:=
/assignvariableop_8_batch_normalization_43_gamma:<
.assignvariableop_9_batch_normalization_43_beta:D
6assignvariableop_10_batch_normalization_43_moving_mean:H
:assignvariableop_11_batch_normalization_43_moving_variance:>
$assignvariableop_12_conv2d_44_kernel:0
"assignvariableop_13_conv2d_44_bias:>
0assignvariableop_14_batch_normalization_44_gamma:=
/assignvariableop_15_batch_normalization_44_beta:D
6assignvariableop_16_batch_normalization_44_moving_mean:H
:assignvariableop_17_batch_normalization_44_moving_variance:>
$assignvariableop_18_conv2d_45_kernel: 0
"assignvariableop_19_conv2d_45_bias: >
0assignvariableop_20_batch_normalization_45_gamma: =
/assignvariableop_21_batch_normalization_45_beta: D
6assignvariableop_22_batch_normalization_45_moving_mean: H
:assignvariableop_23_batch_normalization_45_moving_variance: >
$assignvariableop_24_conv2d_46_kernel:  0
"assignvariableop_25_conv2d_46_bias: >
0assignvariableop_26_batch_normalization_46_gamma: =
/assignvariableop_27_batch_normalization_46_beta: D
6assignvariableop_28_batch_normalization_46_moving_mean: H
:assignvariableop_29_batch_normalization_46_moving_variance: >
$assignvariableop_30_conv2d_47_kernel: @0
"assignvariableop_31_conv2d_47_bias:@>
0assignvariableop_32_batch_normalization_47_gamma:@=
/assignvariableop_33_batch_normalization_47_beta:@D
6assignvariableop_34_batch_normalization_47_moving_mean:@H
:assignvariableop_35_batch_normalization_47_moving_variance:@>
$assignvariableop_36_conv2d_48_kernel:@ 0
"assignvariableop_37_conv2d_48_bias: >
0assignvariableop_38_batch_normalization_48_gamma: =
/assignvariableop_39_batch_normalization_48_beta: D
6assignvariableop_40_batch_normalization_48_moving_mean: H
:assignvariableop_41_batch_normalization_48_moving_variance: H
.assignvariableop_42_conv2d_transpose_18_kernel:@ :
,assignvariableop_43_conv2d_transpose_18_bias:@H
.assignvariableop_44_conv2d_transpose_19_kernel: @:
,assignvariableop_45_conv2d_transpose_19_bias: H
.assignvariableop_46_conv2d_transpose_20_kernel: :
,assignvariableop_47_conv2d_transpose_20_bias:<
"assignvariableop_48_decoded_kernel:.
 assignvariableop_49_decoded_bias:'
assignvariableop_50_adam_iter:	 )
assignvariableop_51_adam_beta_1: )
assignvariableop_52_adam_beta_2: (
assignvariableop_53_adam_decay: 0
&assignvariableop_54_adam_learning_rate: #
assignvariableop_55_total: #
assignvariableop_56_count: E
+assignvariableop_57_adam_conv2d_42_kernel_m:7
)assignvariableop_58_adam_conv2d_42_bias_m:E
7assignvariableop_59_adam_batch_normalization_42_gamma_m:D
6assignvariableop_60_adam_batch_normalization_42_beta_m:E
+assignvariableop_61_adam_conv2d_43_kernel_m:7
)assignvariableop_62_adam_conv2d_43_bias_m:E
7assignvariableop_63_adam_batch_normalization_43_gamma_m:D
6assignvariableop_64_adam_batch_normalization_43_beta_m:E
+assignvariableop_65_adam_conv2d_44_kernel_m:7
)assignvariableop_66_adam_conv2d_44_bias_m:E
7assignvariableop_67_adam_batch_normalization_44_gamma_m:D
6assignvariableop_68_adam_batch_normalization_44_beta_m:E
+assignvariableop_69_adam_conv2d_45_kernel_m: 7
)assignvariableop_70_adam_conv2d_45_bias_m: E
7assignvariableop_71_adam_batch_normalization_45_gamma_m: D
6assignvariableop_72_adam_batch_normalization_45_beta_m: E
+assignvariableop_73_adam_conv2d_46_kernel_m:  7
)assignvariableop_74_adam_conv2d_46_bias_m: E
7assignvariableop_75_adam_batch_normalization_46_gamma_m: D
6assignvariableop_76_adam_batch_normalization_46_beta_m: E
+assignvariableop_77_adam_conv2d_47_kernel_m: @7
)assignvariableop_78_adam_conv2d_47_bias_m:@E
7assignvariableop_79_adam_batch_normalization_47_gamma_m:@D
6assignvariableop_80_adam_batch_normalization_47_beta_m:@E
+assignvariableop_81_adam_conv2d_48_kernel_m:@ 7
)assignvariableop_82_adam_conv2d_48_bias_m: E
7assignvariableop_83_adam_batch_normalization_48_gamma_m: D
6assignvariableop_84_adam_batch_normalization_48_beta_m: O
5assignvariableop_85_adam_conv2d_transpose_18_kernel_m:@ A
3assignvariableop_86_adam_conv2d_transpose_18_bias_m:@O
5assignvariableop_87_adam_conv2d_transpose_19_kernel_m: @A
3assignvariableop_88_adam_conv2d_transpose_19_bias_m: O
5assignvariableop_89_adam_conv2d_transpose_20_kernel_m: A
3assignvariableop_90_adam_conv2d_transpose_20_bias_m:C
)assignvariableop_91_adam_decoded_kernel_m:5
'assignvariableop_92_adam_decoded_bias_m:E
+assignvariableop_93_adam_conv2d_42_kernel_v:7
)assignvariableop_94_adam_conv2d_42_bias_v:E
7assignvariableop_95_adam_batch_normalization_42_gamma_v:D
6assignvariableop_96_adam_batch_normalization_42_beta_v:E
+assignvariableop_97_adam_conv2d_43_kernel_v:7
)assignvariableop_98_adam_conv2d_43_bias_v:E
7assignvariableop_99_adam_batch_normalization_43_gamma_v:E
7assignvariableop_100_adam_batch_normalization_43_beta_v:F
,assignvariableop_101_adam_conv2d_44_kernel_v:8
*assignvariableop_102_adam_conv2d_44_bias_v:F
8assignvariableop_103_adam_batch_normalization_44_gamma_v:E
7assignvariableop_104_adam_batch_normalization_44_beta_v:F
,assignvariableop_105_adam_conv2d_45_kernel_v: 8
*assignvariableop_106_adam_conv2d_45_bias_v: F
8assignvariableop_107_adam_batch_normalization_45_gamma_v: E
7assignvariableop_108_adam_batch_normalization_45_beta_v: F
,assignvariableop_109_adam_conv2d_46_kernel_v:  8
*assignvariableop_110_adam_conv2d_46_bias_v: F
8assignvariableop_111_adam_batch_normalization_46_gamma_v: E
7assignvariableop_112_adam_batch_normalization_46_beta_v: F
,assignvariableop_113_adam_conv2d_47_kernel_v: @8
*assignvariableop_114_adam_conv2d_47_bias_v:@F
8assignvariableop_115_adam_batch_normalization_47_gamma_v:@E
7assignvariableop_116_adam_batch_normalization_47_beta_v:@F
,assignvariableop_117_adam_conv2d_48_kernel_v:@ 8
*assignvariableop_118_adam_conv2d_48_bias_v: F
8assignvariableop_119_adam_batch_normalization_48_gamma_v: E
7assignvariableop_120_adam_batch_normalization_48_beta_v: P
6assignvariableop_121_adam_conv2d_transpose_18_kernel_v:@ B
4assignvariableop_122_adam_conv2d_transpose_18_bias_v:@P
6assignvariableop_123_adam_conv2d_transpose_19_kernel_v: @B
4assignvariableop_124_adam_conv2d_transpose_19_bias_v: P
6assignvariableop_125_adam_conv2d_transpose_20_kernel_v: B
4assignvariableop_126_adam_conv2d_transpose_20_bias_v:D
*assignvariableop_127_adam_decoded_kernel_v:6
(assignvariableop_128_adam_decoded_bias_v:
identity_130??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_125?AssignVariableOp_126?AssignVariableOp_127?AssignVariableOp_128?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?I
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?H
value?HB?H?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_42_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_42_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_42_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_42_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_42_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_42_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_43_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_43_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_43_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_43_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_43_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_43_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_44_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_44_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_44_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_44_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_44_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_44_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_45_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_45_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_45_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_45_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_45_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_45_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_46_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_46_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_normalization_46_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_46_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp6assignvariableop_28_batch_normalization_46_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_normalization_46_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv2d_47_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv2d_47_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_47_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_47_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_47_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_47_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp$assignvariableop_36_conv2d_48_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp"assignvariableop_37_conv2d_48_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp0assignvariableop_38_batch_normalization_48_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp/assignvariableop_39_batch_normalization_48_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp6assignvariableop_40_batch_normalization_48_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp:assignvariableop_41_batch_normalization_48_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp.assignvariableop_42_conv2d_transpose_18_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp,assignvariableop_43_conv2d_transpose_18_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp.assignvariableop_44_conv2d_transpose_19_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp,assignvariableop_45_conv2d_transpose_19_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp.assignvariableop_46_conv2d_transpose_20_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp,assignvariableop_47_conv2d_transpose_20_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp"assignvariableop_48_decoded_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp assignvariableop_49_decoded_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_50AssignVariableOpassignvariableop_50_adam_iterIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOpassignvariableop_51_adam_beta_1Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOpassignvariableop_52_adam_beta_2Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOpassignvariableop_53_adam_decayIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp&assignvariableop_54_adam_learning_rateIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOpassignvariableop_55_totalIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOpassignvariableop_56_countIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv2d_42_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv2d_42_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp7assignvariableop_59_adam_batch_normalization_42_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adam_batch_normalization_42_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_conv2d_43_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_conv2d_43_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp7assignvariableop_63_adam_batch_normalization_43_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_batch_normalization_43_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv2d_44_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv2d_44_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp7assignvariableop_67_adam_batch_normalization_44_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adam_batch_normalization_44_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv2d_45_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv2d_45_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp7assignvariableop_71_adam_batch_normalization_45_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adam_batch_normalization_45_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_conv2d_46_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_conv2d_46_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp7assignvariableop_75_adam_batch_normalization_46_gamma_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOp6assignvariableop_76_adam_batch_normalization_46_beta_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_conv2d_47_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_conv2d_47_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp7assignvariableop_79_adam_batch_normalization_47_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOp6assignvariableop_80_adam_batch_normalization_47_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOp+assignvariableop_81_adam_conv2d_48_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_conv2d_48_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOp7assignvariableop_83_adam_batch_normalization_48_gamma_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOp6assignvariableop_84_adam_batch_normalization_48_beta_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOp5assignvariableop_85_adam_conv2d_transpose_18_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOp3assignvariableop_86_adam_conv2d_transpose_18_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_87AssignVariableOp5assignvariableop_87_adam_conv2d_transpose_19_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_88AssignVariableOp3assignvariableop_88_adam_conv2d_transpose_19_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_89AssignVariableOp5assignvariableop_89_adam_conv2d_transpose_20_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_90AssignVariableOp3assignvariableop_90_adam_conv2d_transpose_20_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_91AssignVariableOp)assignvariableop_91_adam_decoded_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_92AssignVariableOp'assignvariableop_92_adam_decoded_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_93AssignVariableOp+assignvariableop_93_adam_conv2d_42_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_94AssignVariableOp)assignvariableop_94_adam_conv2d_42_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_95AssignVariableOp7assignvariableop_95_adam_batch_normalization_42_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_96AssignVariableOp6assignvariableop_96_adam_batch_normalization_42_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_97AssignVariableOp+assignvariableop_97_adam_conv2d_43_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_98AssignVariableOp)assignvariableop_98_adam_conv2d_43_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_99AssignVariableOp7assignvariableop_99_adam_batch_normalization_43_gamma_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_100AssignVariableOp7assignvariableop_100_adam_batch_normalization_43_beta_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_101AssignVariableOp,assignvariableop_101_adam_conv2d_44_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_102AssignVariableOp*assignvariableop_102_adam_conv2d_44_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_103AssignVariableOp8assignvariableop_103_adam_batch_normalization_44_gamma_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_104AssignVariableOp7assignvariableop_104_adam_batch_normalization_44_beta_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_105AssignVariableOp,assignvariableop_105_adam_conv2d_45_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_106AssignVariableOp*assignvariableop_106_adam_conv2d_45_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_107AssignVariableOp8assignvariableop_107_adam_batch_normalization_45_gamma_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_108AssignVariableOp7assignvariableop_108_adam_batch_normalization_45_beta_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_109AssignVariableOp,assignvariableop_109_adam_conv2d_46_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_110AssignVariableOp*assignvariableop_110_adam_conv2d_46_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_111AssignVariableOp8assignvariableop_111_adam_batch_normalization_46_gamma_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_112AssignVariableOp7assignvariableop_112_adam_batch_normalization_46_beta_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_113AssignVariableOp,assignvariableop_113_adam_conv2d_47_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_114AssignVariableOp*assignvariableop_114_adam_conv2d_47_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_115AssignVariableOp8assignvariableop_115_adam_batch_normalization_47_gamma_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_116AssignVariableOp7assignvariableop_116_adam_batch_normalization_47_beta_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_117AssignVariableOp,assignvariableop_117_adam_conv2d_48_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_118AssignVariableOp*assignvariableop_118_adam_conv2d_48_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_119AssignVariableOp8assignvariableop_119_adam_batch_normalization_48_gamma_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_120AssignVariableOp7assignvariableop_120_adam_batch_normalization_48_beta_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_121AssignVariableOp6assignvariableop_121_adam_conv2d_transpose_18_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_122AssignVariableOp4assignvariableop_122_adam_conv2d_transpose_18_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_123AssignVariableOp6assignvariableop_123_adam_conv2d_transpose_19_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_124AssignVariableOp4assignvariableop_124_adam_conv2d_transpose_19_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_125AssignVariableOp6assignvariableop_125_adam_conv2d_transpose_20_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_126AssignVariableOp4assignvariableop_126_adam_conv2d_transpose_20_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_127AssignVariableOp*assignvariableop_127_adam_decoded_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_128AssignVariableOp(assignvariableop_128_adam_decoded_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_129Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_130IdentityIdentity_129:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_130Identity_130:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282*
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
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
.__inference_sequential_6_layer_call_fn_2596392
conv2d_42_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23:  

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: $

unknown_29: @

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@$

unknown_35:@ 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: $

unknown_41:@ 

unknown_42:@$

unknown_43: @

unknown_44: $

unknown_45: 

unknown_46:$

unknown_47:

unknown_48:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_42_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_2596289y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_42_input
?
?
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2595731

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
?
?
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2598473

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
? 
?
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_2595918

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
-:+??????????????????????????? y
IdentityIdentityBiasAdd:output:0^NoOp*
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
J__inference_activation_58_layer_call_and_return_conditional_losses_2598483

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@@ g
IdentityIdentityLeakyRelu:activations:0*
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
?
?
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2598564

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
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2595475

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
`
D__inference_encoded_layer_call_and_return_conditional_losses_2598665

inputs
identityX
	LeakyRelu	LeakyReluinputs*
T0*/
_output_shapes
:????????? g
IdentityIdentityLeakyRelu:activations:0*
T0*/
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
K
/__inference_activation_56_layer_call_fn_2598296

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
GPU2 *0J 8? *S
fNRL
J__inference_activation_56_layer_call_and_return_conditional_losses_2596115j
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
?

?
F__inference_conv2d_44_layer_call_and_return_conditional_losses_2598229

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
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
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_18_layer_call_fn_2598674

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
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_2595874?
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
?
`
D__inference_encoded_layer_call_and_return_conditional_losses_2596244

inputs
identityX
	LeakyRelu	LeakyReluinputs*
T0*/
_output_shapes
:????????? g
IdentityIdentityLeakyRelu:activations:0*
T0*/
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
?
?
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2598182

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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2598382

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
f
J__inference_activation_60_layer_call_and_return_conditional_losses_2598717

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????  @g
IdentityIdentityLeakyRelu:activations:0*
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
?
?
.__inference_sequential_6_layer_call_fn_2597562

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23:  

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: $

unknown_29: @

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@$

unknown_35:@ 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: $

unknown_41:@ 

unknown_42:@$

unknown_43: @

unknown_44: $

unknown_45: 

unknown_46:$

unknown_47:

unknown_48:
identity??StatefulPartitionedCall?
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
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*F
_read_only_resource_inputs(
&$	
 !"%&'(+,-./012*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_2596763y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
F__inference_conv2d_45_layer_call_and_return_conditional_losses_2598320

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
8__inference_batch_normalization_43_layer_call_fn_2598164

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
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2595506?
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
D__inference_decoded_layer_call_and_return_conditional_losses_2598864

inputsB
(conv2d_transpose_readvariableop_resource:-
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
:*
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
-:+???????????????????????????j
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????q
IdentityIdentityTanh:y:0^NoOp*
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
?

?
F__inference_conv2d_43_layer_call_and_return_conditional_losses_2596063

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
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
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
? 
?
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_2598759

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
-:+??????????????????????????? y
IdentityIdentityBiasAdd:output:0^NoOp*
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
8__inference_batch_normalization_47_layer_call_fn_2598528

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
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2595762?
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
?
K
/__inference_activation_54_layer_call_fn_2598114

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
GPU2 *0J 8? *S
fNRL
J__inference_activation_54_layer_call_and_return_conditional_losses_2596051j
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
??
?<
 __inference__traced_save_2599274
file_prefix/
+savev2_conv2d_42_kernel_read_readvariableop-
)savev2_conv2d_42_bias_read_readvariableop;
7savev2_batch_normalization_42_gamma_read_readvariableop:
6savev2_batch_normalization_42_beta_read_readvariableopA
=savev2_batch_normalization_42_moving_mean_read_readvariableopE
Asavev2_batch_normalization_42_moving_variance_read_readvariableop/
+savev2_conv2d_43_kernel_read_readvariableop-
)savev2_conv2d_43_bias_read_readvariableop;
7savev2_batch_normalization_43_gamma_read_readvariableop:
6savev2_batch_normalization_43_beta_read_readvariableopA
=savev2_batch_normalization_43_moving_mean_read_readvariableopE
Asavev2_batch_normalization_43_moving_variance_read_readvariableop/
+savev2_conv2d_44_kernel_read_readvariableop-
)savev2_conv2d_44_bias_read_readvariableop;
7savev2_batch_normalization_44_gamma_read_readvariableop:
6savev2_batch_normalization_44_beta_read_readvariableopA
=savev2_batch_normalization_44_moving_mean_read_readvariableopE
Asavev2_batch_normalization_44_moving_variance_read_readvariableop/
+savev2_conv2d_45_kernel_read_readvariableop-
)savev2_conv2d_45_bias_read_readvariableop;
7savev2_batch_normalization_45_gamma_read_readvariableop:
6savev2_batch_normalization_45_beta_read_readvariableopA
=savev2_batch_normalization_45_moving_mean_read_readvariableopE
Asavev2_batch_normalization_45_moving_variance_read_readvariableop/
+savev2_conv2d_46_kernel_read_readvariableop-
)savev2_conv2d_46_bias_read_readvariableop;
7savev2_batch_normalization_46_gamma_read_readvariableop:
6savev2_batch_normalization_46_beta_read_readvariableopA
=savev2_batch_normalization_46_moving_mean_read_readvariableopE
Asavev2_batch_normalization_46_moving_variance_read_readvariableop/
+savev2_conv2d_47_kernel_read_readvariableop-
)savev2_conv2d_47_bias_read_readvariableop;
7savev2_batch_normalization_47_gamma_read_readvariableop:
6savev2_batch_normalization_47_beta_read_readvariableopA
=savev2_batch_normalization_47_moving_mean_read_readvariableopE
Asavev2_batch_normalization_47_moving_variance_read_readvariableop/
+savev2_conv2d_48_kernel_read_readvariableop-
)savev2_conv2d_48_bias_read_readvariableop;
7savev2_batch_normalization_48_gamma_read_readvariableop:
6savev2_batch_normalization_48_beta_read_readvariableopA
=savev2_batch_normalization_48_moving_mean_read_readvariableopE
Asavev2_batch_normalization_48_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_18_kernel_read_readvariableop7
3savev2_conv2d_transpose_18_bias_read_readvariableop9
5savev2_conv2d_transpose_19_kernel_read_readvariableop7
3savev2_conv2d_transpose_19_bias_read_readvariableop9
5savev2_conv2d_transpose_20_kernel_read_readvariableop7
3savev2_conv2d_transpose_20_bias_read_readvariableop-
)savev2_decoded_kernel_read_readvariableop+
'savev2_decoded_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_42_kernel_m_read_readvariableop4
0savev2_adam_conv2d_42_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_42_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_42_beta_m_read_readvariableop6
2savev2_adam_conv2d_43_kernel_m_read_readvariableop4
0savev2_adam_conv2d_43_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_43_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_43_beta_m_read_readvariableop6
2savev2_adam_conv2d_44_kernel_m_read_readvariableop4
0savev2_adam_conv2d_44_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_44_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_44_beta_m_read_readvariableop6
2savev2_adam_conv2d_45_kernel_m_read_readvariableop4
0savev2_adam_conv2d_45_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_45_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_45_beta_m_read_readvariableop6
2savev2_adam_conv2d_46_kernel_m_read_readvariableop4
0savev2_adam_conv2d_46_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_46_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_46_beta_m_read_readvariableop6
2savev2_adam_conv2d_47_kernel_m_read_readvariableop4
0savev2_adam_conv2d_47_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_47_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_47_beta_m_read_readvariableop6
2savev2_adam_conv2d_48_kernel_m_read_readvariableop4
0savev2_adam_conv2d_48_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_48_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_48_beta_m_read_readvariableop@
<savev2_adam_conv2d_transpose_18_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_18_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_19_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_19_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_20_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_20_bias_m_read_readvariableop4
0savev2_adam_decoded_kernel_m_read_readvariableop2
.savev2_adam_decoded_bias_m_read_readvariableop6
2savev2_adam_conv2d_42_kernel_v_read_readvariableop4
0savev2_adam_conv2d_42_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_42_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_42_beta_v_read_readvariableop6
2savev2_adam_conv2d_43_kernel_v_read_readvariableop4
0savev2_adam_conv2d_43_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_43_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_43_beta_v_read_readvariableop6
2savev2_adam_conv2d_44_kernel_v_read_readvariableop4
0savev2_adam_conv2d_44_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_44_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_44_beta_v_read_readvariableop6
2savev2_adam_conv2d_45_kernel_v_read_readvariableop4
0savev2_adam_conv2d_45_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_45_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_45_beta_v_read_readvariableop6
2savev2_adam_conv2d_46_kernel_v_read_readvariableop4
0savev2_adam_conv2d_46_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_46_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_46_beta_v_read_readvariableop6
2savev2_adam_conv2d_47_kernel_v_read_readvariableop4
0savev2_adam_conv2d_47_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_47_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_47_beta_v_read_readvariableop6
2savev2_adam_conv2d_48_kernel_v_read_readvariableop4
0savev2_adam_conv2d_48_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_48_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_48_beta_v_read_readvariableop@
<savev2_adam_conv2d_transpose_18_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_18_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_19_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_19_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_20_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_20_bias_v_read_readvariableop4
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
: ?I
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?H
value?HB?H?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?:
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_42_kernel_read_readvariableop)savev2_conv2d_42_bias_read_readvariableop7savev2_batch_normalization_42_gamma_read_readvariableop6savev2_batch_normalization_42_beta_read_readvariableop=savev2_batch_normalization_42_moving_mean_read_readvariableopAsavev2_batch_normalization_42_moving_variance_read_readvariableop+savev2_conv2d_43_kernel_read_readvariableop)savev2_conv2d_43_bias_read_readvariableop7savev2_batch_normalization_43_gamma_read_readvariableop6savev2_batch_normalization_43_beta_read_readvariableop=savev2_batch_normalization_43_moving_mean_read_readvariableopAsavev2_batch_normalization_43_moving_variance_read_readvariableop+savev2_conv2d_44_kernel_read_readvariableop)savev2_conv2d_44_bias_read_readvariableop7savev2_batch_normalization_44_gamma_read_readvariableop6savev2_batch_normalization_44_beta_read_readvariableop=savev2_batch_normalization_44_moving_mean_read_readvariableopAsavev2_batch_normalization_44_moving_variance_read_readvariableop+savev2_conv2d_45_kernel_read_readvariableop)savev2_conv2d_45_bias_read_readvariableop7savev2_batch_normalization_45_gamma_read_readvariableop6savev2_batch_normalization_45_beta_read_readvariableop=savev2_batch_normalization_45_moving_mean_read_readvariableopAsavev2_batch_normalization_45_moving_variance_read_readvariableop+savev2_conv2d_46_kernel_read_readvariableop)savev2_conv2d_46_bias_read_readvariableop7savev2_batch_normalization_46_gamma_read_readvariableop6savev2_batch_normalization_46_beta_read_readvariableop=savev2_batch_normalization_46_moving_mean_read_readvariableopAsavev2_batch_normalization_46_moving_variance_read_readvariableop+savev2_conv2d_47_kernel_read_readvariableop)savev2_conv2d_47_bias_read_readvariableop7savev2_batch_normalization_47_gamma_read_readvariableop6savev2_batch_normalization_47_beta_read_readvariableop=savev2_batch_normalization_47_moving_mean_read_readvariableopAsavev2_batch_normalization_47_moving_variance_read_readvariableop+savev2_conv2d_48_kernel_read_readvariableop)savev2_conv2d_48_bias_read_readvariableop7savev2_batch_normalization_48_gamma_read_readvariableop6savev2_batch_normalization_48_beta_read_readvariableop=savev2_batch_normalization_48_moving_mean_read_readvariableopAsavev2_batch_normalization_48_moving_variance_read_readvariableop5savev2_conv2d_transpose_18_kernel_read_readvariableop3savev2_conv2d_transpose_18_bias_read_readvariableop5savev2_conv2d_transpose_19_kernel_read_readvariableop3savev2_conv2d_transpose_19_bias_read_readvariableop5savev2_conv2d_transpose_20_kernel_read_readvariableop3savev2_conv2d_transpose_20_bias_read_readvariableop)savev2_decoded_kernel_read_readvariableop'savev2_decoded_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_42_kernel_m_read_readvariableop0savev2_adam_conv2d_42_bias_m_read_readvariableop>savev2_adam_batch_normalization_42_gamma_m_read_readvariableop=savev2_adam_batch_normalization_42_beta_m_read_readvariableop2savev2_adam_conv2d_43_kernel_m_read_readvariableop0savev2_adam_conv2d_43_bias_m_read_readvariableop>savev2_adam_batch_normalization_43_gamma_m_read_readvariableop=savev2_adam_batch_normalization_43_beta_m_read_readvariableop2savev2_adam_conv2d_44_kernel_m_read_readvariableop0savev2_adam_conv2d_44_bias_m_read_readvariableop>savev2_adam_batch_normalization_44_gamma_m_read_readvariableop=savev2_adam_batch_normalization_44_beta_m_read_readvariableop2savev2_adam_conv2d_45_kernel_m_read_readvariableop0savev2_adam_conv2d_45_bias_m_read_readvariableop>savev2_adam_batch_normalization_45_gamma_m_read_readvariableop=savev2_adam_batch_normalization_45_beta_m_read_readvariableop2savev2_adam_conv2d_46_kernel_m_read_readvariableop0savev2_adam_conv2d_46_bias_m_read_readvariableop>savev2_adam_batch_normalization_46_gamma_m_read_readvariableop=savev2_adam_batch_normalization_46_beta_m_read_readvariableop2savev2_adam_conv2d_47_kernel_m_read_readvariableop0savev2_adam_conv2d_47_bias_m_read_readvariableop>savev2_adam_batch_normalization_47_gamma_m_read_readvariableop=savev2_adam_batch_normalization_47_beta_m_read_readvariableop2savev2_adam_conv2d_48_kernel_m_read_readvariableop0savev2_adam_conv2d_48_bias_m_read_readvariableop>savev2_adam_batch_normalization_48_gamma_m_read_readvariableop=savev2_adam_batch_normalization_48_beta_m_read_readvariableop<savev2_adam_conv2d_transpose_18_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_18_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_19_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_19_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_20_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_20_bias_m_read_readvariableop0savev2_adam_decoded_kernel_m_read_readvariableop.savev2_adam_decoded_bias_m_read_readvariableop2savev2_adam_conv2d_42_kernel_v_read_readvariableop0savev2_adam_conv2d_42_bias_v_read_readvariableop>savev2_adam_batch_normalization_42_gamma_v_read_readvariableop=savev2_adam_batch_normalization_42_beta_v_read_readvariableop2savev2_adam_conv2d_43_kernel_v_read_readvariableop0savev2_adam_conv2d_43_bias_v_read_readvariableop>savev2_adam_batch_normalization_43_gamma_v_read_readvariableop=savev2_adam_batch_normalization_43_beta_v_read_readvariableop2savev2_adam_conv2d_44_kernel_v_read_readvariableop0savev2_adam_conv2d_44_bias_v_read_readvariableop>savev2_adam_batch_normalization_44_gamma_v_read_readvariableop=savev2_adam_batch_normalization_44_beta_v_read_readvariableop2savev2_adam_conv2d_45_kernel_v_read_readvariableop0savev2_adam_conv2d_45_bias_v_read_readvariableop>savev2_adam_batch_normalization_45_gamma_v_read_readvariableop=savev2_adam_batch_normalization_45_beta_v_read_readvariableop2savev2_adam_conv2d_46_kernel_v_read_readvariableop0savev2_adam_conv2d_46_bias_v_read_readvariableop>savev2_adam_batch_normalization_46_gamma_v_read_readvariableop=savev2_adam_batch_normalization_46_beta_v_read_readvariableop2savev2_adam_conv2d_47_kernel_v_read_readvariableop0savev2_adam_conv2d_47_bias_v_read_readvariableop>savev2_adam_batch_normalization_47_gamma_v_read_readvariableop=savev2_adam_batch_normalization_47_beta_v_read_readvariableop2savev2_adam_conv2d_48_kernel_v_read_readvariableop0savev2_adam_conv2d_48_bias_v_read_readvariableop>savev2_adam_batch_normalization_48_gamma_v_read_readvariableop=savev2_adam_batch_normalization_48_beta_v_read_readvariableop<savev2_adam_conv2d_transpose_18_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_18_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_19_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_19_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_20_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_20_bias_v_read_readvariableop0savev2_adam_decoded_kernel_v_read_readvariableop.savev2_adam_decoded_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2?	?
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

identity_1Identity_1:output:0*?	
_input_shapes?
?: ::::::::::::::::::: : : : : : :  : : : : : : @:@:@:@:@:@:@ : : : : : :@ :@: @: : :::: : : : : : : ::::::::::::: : : : :  : : : : @:@:@:@:@ : : : :@ :@: @: : :::::::::::::::: : : : :  : : : : @:@:@:@:@ : : : :@ :@: @: : :::: 2(
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
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 
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
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @:  

_output_shapes
:@: !

_output_shapes
:@: "

_output_shapes
:@: #

_output_shapes
:@: $

_output_shapes
:@:,%(
&
_output_shapes
:@ : &

_output_shapes
: : '

_output_shapes
: : (

_output_shapes
: : )

_output_shapes
: : *

_output_shapes
: :,+(
&
_output_shapes
:@ : ,

_output_shapes
:@:,-(
&
_output_shapes
: @: .

_output_shapes
: :,/(
&
_output_shapes
: : 0

_output_shapes
::,1(
&
_output_shapes
:: 2

_output_shapes
::3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: :,:(
&
_output_shapes
:: ;

_output_shapes
:: <

_output_shapes
:: =

_output_shapes
::,>(
&
_output_shapes
:: ?

_output_shapes
:: @

_output_shapes
:: A

_output_shapes
::,B(
&
_output_shapes
:: C

_output_shapes
:: D

_output_shapes
:: E

_output_shapes
::,F(
&
_output_shapes
: : G

_output_shapes
: : H

_output_shapes
: : I

_output_shapes
: :,J(
&
_output_shapes
:  : K

_output_shapes
: : L

_output_shapes
: : M

_output_shapes
: :,N(
&
_output_shapes
: @: O

_output_shapes
:@: P

_output_shapes
:@: Q

_output_shapes
:@:,R(
&
_output_shapes
:@ : S

_output_shapes
: : T

_output_shapes
: : U

_output_shapes
: :,V(
&
_output_shapes
:@ : W

_output_shapes
:@:,X(
&
_output_shapes
: @: Y

_output_shapes
: :,Z(
&
_output_shapes
: : [

_output_shapes
::,\(
&
_output_shapes
:: ]

_output_shapes
::,^(
&
_output_shapes
:: _

_output_shapes
:: `

_output_shapes
:: a

_output_shapes
::,b(
&
_output_shapes
:: c

_output_shapes
:: d

_output_shapes
:: e

_output_shapes
::,f(
&
_output_shapes
:: g

_output_shapes
:: h

_output_shapes
:: i

_output_shapes
::,j(
&
_output_shapes
: : k

_output_shapes
: : l

_output_shapes
: : m

_output_shapes
: :,n(
&
_output_shapes
:  : o

_output_shapes
: : p

_output_shapes
: : q

_output_shapes
: :,r(
&
_output_shapes
: @: s

_output_shapes
:@: t

_output_shapes
:@: u

_output_shapes
:@:,v(
&
_output_shapes
:@ : w

_output_shapes
: : x

_output_shapes
: : y

_output_shapes
: :,z(
&
_output_shapes
:@ : {

_output_shapes
:@:,|(
&
_output_shapes
: @: }

_output_shapes
: :,~(
&
_output_shapes
: : 

_output_shapes
::-?(
&
_output_shapes
::!?

_output_shapes
::?

_output_shapes
: 
?	
?
8__inference_batch_normalization_46_layer_call_fn_2598437

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
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2595698?
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
?
+__inference_conv2d_47_layer_call_fn_2598492

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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_47_layer_call_and_return_conditional_losses_2596191w
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
?
f
J__inference_activation_60_layer_call_and_return_conditional_losses_2596257

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????  @g
IdentityIdentityLeakyRelu:activations:0*
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
?
?
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2595698

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
?
K
/__inference_activation_59_layer_call_fn_2598569

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
GPU2 *0J 8? *S
fNRL
J__inference_activation_59_layer_call_and_return_conditional_losses_2596211h
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
??
?1
I__inference_sequential_6_layer_call_and_return_conditional_losses_2598028

inputsB
(conv2d_42_conv2d_readvariableop_resource:7
)conv2d_42_biasadd_readvariableop_resource:<
.batch_normalization_42_readvariableop_resource:>
0batch_normalization_42_readvariableop_1_resource:M
?batch_normalization_42_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_42_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_43_conv2d_readvariableop_resource:7
)conv2d_43_biasadd_readvariableop_resource:<
.batch_normalization_43_readvariableop_resource:>
0batch_normalization_43_readvariableop_1_resource:M
?batch_normalization_43_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_44_conv2d_readvariableop_resource:7
)conv2d_44_biasadd_readvariableop_resource:<
.batch_normalization_44_readvariableop_resource:>
0batch_normalization_44_readvariableop_1_resource:M
?batch_normalization_44_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_45_conv2d_readvariableop_resource: 7
)conv2d_45_biasadd_readvariableop_resource: <
.batch_normalization_45_readvariableop_resource: >
0batch_normalization_45_readvariableop_1_resource: M
?batch_normalization_45_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_46_conv2d_readvariableop_resource:  7
)conv2d_46_biasadd_readvariableop_resource: <
.batch_normalization_46_readvariableop_resource: >
0batch_normalization_46_readvariableop_1_resource: M
?batch_normalization_46_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_47_conv2d_readvariableop_resource: @7
)conv2d_47_biasadd_readvariableop_resource:@<
.batch_normalization_47_readvariableop_resource:@>
0batch_normalization_47_readvariableop_1_resource:@M
?batch_normalization_47_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource:@B
(conv2d_48_conv2d_readvariableop_resource:@ 7
)conv2d_48_biasadd_readvariableop_resource: <
.batch_normalization_48_readvariableop_resource: >
0batch_normalization_48_readvariableop_1_resource: M
?batch_normalization_48_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_18_conv2d_transpose_readvariableop_resource:@ A
3conv2d_transpose_18_biasadd_readvariableop_resource:@V
<conv2d_transpose_19_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_19_biasadd_readvariableop_resource: V
<conv2d_transpose_20_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_20_biasadd_readvariableop_resource:J
0decoded_conv2d_transpose_readvariableop_resource:5
'decoded_biasadd_readvariableop_resource:
identity??%batch_normalization_42/AssignNewValue?'batch_normalization_42/AssignNewValue_1?6batch_normalization_42/FusedBatchNormV3/ReadVariableOp?8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_42/ReadVariableOp?'batch_normalization_42/ReadVariableOp_1?%batch_normalization_43/AssignNewValue?'batch_normalization_43/AssignNewValue_1?6batch_normalization_43/FusedBatchNormV3/ReadVariableOp?8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_43/ReadVariableOp?'batch_normalization_43/ReadVariableOp_1?%batch_normalization_44/AssignNewValue?'batch_normalization_44/AssignNewValue_1?6batch_normalization_44/FusedBatchNormV3/ReadVariableOp?8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_44/ReadVariableOp?'batch_normalization_44/ReadVariableOp_1?%batch_normalization_45/AssignNewValue?'batch_normalization_45/AssignNewValue_1?6batch_normalization_45/FusedBatchNormV3/ReadVariableOp?8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_45/ReadVariableOp?'batch_normalization_45/ReadVariableOp_1?%batch_normalization_46/AssignNewValue?'batch_normalization_46/AssignNewValue_1?6batch_normalization_46/FusedBatchNormV3/ReadVariableOp?8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_46/ReadVariableOp?'batch_normalization_46/ReadVariableOp_1?%batch_normalization_47/AssignNewValue?'batch_normalization_47/AssignNewValue_1?6batch_normalization_47/FusedBatchNormV3/ReadVariableOp?8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_47/ReadVariableOp?'batch_normalization_47/ReadVariableOp_1?%batch_normalization_48/AssignNewValue?'batch_normalization_48/AssignNewValue_1?6batch_normalization_48/FusedBatchNormV3/ReadVariableOp?8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_48/ReadVariableOp?'batch_normalization_48/ReadVariableOp_1? conv2d_42/BiasAdd/ReadVariableOp?conv2d_42/Conv2D/ReadVariableOp? conv2d_43/BiasAdd/ReadVariableOp?conv2d_43/Conv2D/ReadVariableOp? conv2d_44/BiasAdd/ReadVariableOp?conv2d_44/Conv2D/ReadVariableOp? conv2d_45/BiasAdd/ReadVariableOp?conv2d_45/Conv2D/ReadVariableOp? conv2d_46/BiasAdd/ReadVariableOp?conv2d_46/Conv2D/ReadVariableOp? conv2d_47/BiasAdd/ReadVariableOp?conv2d_47/Conv2D/ReadVariableOp? conv2d_48/BiasAdd/ReadVariableOp?conv2d_48/Conv2D/ReadVariableOp?*conv2d_transpose_18/BiasAdd/ReadVariableOp?3conv2d_transpose_18/conv2d_transpose/ReadVariableOp?*conv2d_transpose_19/BiasAdd/ReadVariableOp?3conv2d_transpose_19/conv2d_transpose/ReadVariableOp?*conv2d_transpose_20/BiasAdd/ReadVariableOp?3conv2d_transpose_20/conv2d_transpose/ReadVariableOp?decoded/BiasAdd/ReadVariableOp?'decoded/conv2d_transpose/ReadVariableOp?
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_42/Conv2DConv2Dinputs'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
%batch_normalization_42/ReadVariableOpReadVariableOp.batch_normalization_42_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_42/ReadVariableOp_1ReadVariableOp0batch_normalization_42_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_42/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_42_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_42_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_42/FusedBatchNormV3FusedBatchNormV3conv2d_42/BiasAdd:output:0-batch_normalization_42/ReadVariableOp:value:0/batch_normalization_42/ReadVariableOp_1:value:0>batch_normalization_42/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_42/AssignNewValueAssignVariableOp?batch_normalization_42_fusedbatchnormv3_readvariableop_resource4batch_normalization_42/FusedBatchNormV3:batch_mean:07^batch_normalization_42/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_42/AssignNewValue_1AssignVariableOpAbatch_normalization_42_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_42/FusedBatchNormV3:batch_variance:09^batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_54/LeakyRelu	LeakyRelu+batch_normalization_42/FusedBatchNormV3:y:0*1
_output_shapes
:????????????
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_43/Conv2DConv2D%activation_54/LeakyRelu:activations:0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
%batch_normalization_43/ReadVariableOpReadVariableOp.batch_normalization_43_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_43/ReadVariableOp_1ReadVariableOp0batch_normalization_43_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_43/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_43_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_43/FusedBatchNormV3FusedBatchNormV3conv2d_43/BiasAdd:output:0-batch_normalization_43/ReadVariableOp:value:0/batch_normalization_43/ReadVariableOp_1:value:0>batch_normalization_43/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_43/AssignNewValueAssignVariableOp?batch_normalization_43_fusedbatchnormv3_readvariableop_resource4batch_normalization_43/FusedBatchNormV3:batch_mean:07^batch_normalization_43/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_43/AssignNewValue_1AssignVariableOpAbatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_43/FusedBatchNormV3:batch_variance:09^batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_55/LeakyRelu	LeakyRelu+batch_normalization_43/FusedBatchNormV3:y:0*1
_output_shapes
:????????????
conv2d_44/Conv2D/ReadVariableOpReadVariableOp(conv2d_44_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_44/Conv2DConv2D%activation_55/LeakyRelu:activations:0'conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
 conv2d_44/BiasAdd/ReadVariableOpReadVariableOp)conv2d_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_44/BiasAddBiasAddconv2d_44/Conv2D:output:0(conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
%batch_normalization_44/ReadVariableOpReadVariableOp.batch_normalization_44_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_44/ReadVariableOp_1ReadVariableOp0batch_normalization_44_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_44/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_44_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_44/FusedBatchNormV3FusedBatchNormV3conv2d_44/BiasAdd:output:0-batch_normalization_44/ReadVariableOp:value:0/batch_normalization_44/ReadVariableOp_1:value:0>batch_normalization_44/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_44/AssignNewValueAssignVariableOp?batch_normalization_44_fusedbatchnormv3_readvariableop_resource4batch_normalization_44/FusedBatchNormV3:batch_mean:07^batch_normalization_44/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_44/AssignNewValue_1AssignVariableOpAbatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_44/FusedBatchNormV3:batch_variance:09^batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_56/LeakyRelu	LeakyRelu+batch_normalization_44/FusedBatchNormV3:y:0*1
_output_shapes
:????????????
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_45/Conv2DConv2D%activation_56/LeakyRelu:activations:0'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
%batch_normalization_45/ReadVariableOpReadVariableOp.batch_normalization_45_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_45/ReadVariableOp_1ReadVariableOp0batch_normalization_45_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_45/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_45_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_45/FusedBatchNormV3FusedBatchNormV3conv2d_45/BiasAdd:output:0-batch_normalization_45/ReadVariableOp:value:0/batch_normalization_45/ReadVariableOp_1:value:0>batch_normalization_45/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_45/AssignNewValueAssignVariableOp?batch_normalization_45_fusedbatchnormv3_readvariableop_resource4batch_normalization_45/FusedBatchNormV3:batch_mean:07^batch_normalization_45/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_45/AssignNewValue_1AssignVariableOpAbatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_45/FusedBatchNormV3:batch_variance:09^batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_57/LeakyRelu	LeakyRelu+batch_normalization_45/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@ ?
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_46/Conv2DConv2D%activation_57/LeakyRelu:activations:0'conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0(conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
%batch_normalization_46/ReadVariableOpReadVariableOp.batch_normalization_46_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_46/ReadVariableOp_1ReadVariableOp0batch_normalization_46_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_46/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_46_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_46/FusedBatchNormV3FusedBatchNormV3conv2d_46/BiasAdd:output:0-batch_normalization_46/ReadVariableOp:value:0/batch_normalization_46/ReadVariableOp_1:value:0>batch_normalization_46/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_46/AssignNewValueAssignVariableOp?batch_normalization_46_fusedbatchnormv3_readvariableop_resource4batch_normalization_46/FusedBatchNormV3:batch_mean:07^batch_normalization_46/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_46/AssignNewValue_1AssignVariableOpAbatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_46/FusedBatchNormV3:batch_variance:09^batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_58/LeakyRelu	LeakyRelu+batch_normalization_46/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@ ?
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_47/Conv2DConv2D%activation_58/LeakyRelu:activations:0'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
%batch_normalization_47/ReadVariableOpReadVariableOp.batch_normalization_47_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_47/ReadVariableOp_1ReadVariableOp0batch_normalization_47_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_47/FusedBatchNormV3FusedBatchNormV3conv2d_47/BiasAdd:output:0-batch_normalization_47/ReadVariableOp:value:0/batch_normalization_47/ReadVariableOp_1:value:0>batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_47/AssignNewValueAssignVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource4batch_normalization_47/FusedBatchNormV3:batch_mean:07^batch_normalization_47/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_47/AssignNewValue_1AssignVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_47/FusedBatchNormV3:batch_variance:09^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_59/LeakyRelu	LeakyRelu+batch_normalization_47/FusedBatchNormV3:y:0*/
_output_shapes
:?????????  @?
conv2d_48/Conv2D/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
conv2d_48/Conv2DConv2D%activation_59/LeakyRelu:activations:0'conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_48/BiasAdd/ReadVariableOpReadVariableOp)conv2d_48_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_48/BiasAddBiasAddconv2d_48/Conv2D:output:0(conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
%batch_normalization_48/ReadVariableOpReadVariableOp.batch_normalization_48_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_48/ReadVariableOp_1ReadVariableOp0batch_normalization_48_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_48/FusedBatchNormV3FusedBatchNormV3conv2d_48/BiasAdd:output:0-batch_normalization_48/ReadVariableOp:value:0/batch_normalization_48/ReadVariableOp_1:value:0>batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_48/AssignNewValueAssignVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource4batch_normalization_48/FusedBatchNormV3:batch_mean:07^batch_normalization_48/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_48/AssignNewValue_1AssignVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_48/FusedBatchNormV3:batch_variance:09^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
encoded/CastCast+batch_normalization_48/FusedBatchNormV3:y:0*

DstT0*

SrcT0*/
_output_shapes
:????????? j
encoded/LeakyRelu	LeakyReluencoded/Cast:y:0*
T0*/
_output_shapes
:????????? ?
conv2d_transpose_18/CastCastencoded/LeakyRelu:activations:0*

DstT0*

SrcT0*/
_output_shapes
:????????? e
conv2d_transpose_18/ShapeShapeconv2d_transpose_18/Cast:y:0*
T0*
_output_shapes
:q
'conv2d_transpose_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_18/strided_sliceStridedSlice"conv2d_transpose_18/Shape:output:00conv2d_transpose_18/strided_slice/stack:output:02conv2d_transpose_18/strided_slice/stack_1:output:02conv2d_transpose_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_18/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_18/stack/2Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_18/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose_18/stackPack*conv2d_transpose_18/strided_slice:output:0$conv2d_transpose_18/stack/1:output:0$conv2d_transpose_18/stack/2:output:0$conv2d_transpose_18/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_18/strided_slice_1StridedSlice"conv2d_transpose_18/stack:output:02conv2d_transpose_18/strided_slice_1/stack:output:04conv2d_transpose_18/strided_slice_1/stack_1:output:04conv2d_transpose_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_18/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_18_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
$conv2d_transpose_18/conv2d_transposeConv2DBackpropInput"conv2d_transpose_18/stack:output:0;conv2d_transpose_18/conv2d_transpose/ReadVariableOp:value:0conv2d_transpose_18/Cast:y:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
*conv2d_transpose_18/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_transpose_18/BiasAddBiasAdd-conv2d_transpose_18/conv2d_transpose:output:02conv2d_transpose_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @{
activation_60/LeakyRelu	LeakyRelu$conv2d_transpose_18/BiasAdd:output:0*/
_output_shapes
:?????????  @n
conv2d_transpose_19/ShapeShape%activation_60/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_19/strided_sliceStridedSlice"conv2d_transpose_19/Shape:output:00conv2d_transpose_19/strided_slice/stack:output:02conv2d_transpose_19/strided_slice/stack_1:output:02conv2d_transpose_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_19/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@]
conv2d_transpose_19/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@]
conv2d_transpose_19/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_19/stackPack*conv2d_transpose_19/strided_slice:output:0$conv2d_transpose_19/stack/1:output:0$conv2d_transpose_19/stack/2:output:0$conv2d_transpose_19/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_19/strided_slice_1StridedSlice"conv2d_transpose_19/stack:output:02conv2d_transpose_19/strided_slice_1/stack:output:04conv2d_transpose_19/strided_slice_1/stack_1:output:04conv2d_transpose_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_19/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_19_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
$conv2d_transpose_19/conv2d_transposeConv2DBackpropInput"conv2d_transpose_19/stack:output:0;conv2d_transpose_19/conv2d_transpose/ReadVariableOp:value:0%activation_60/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
*conv2d_transpose_19/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_19/BiasAddBiasAdd-conv2d_transpose_19/conv2d_transpose:output:02conv2d_transpose_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ {
activation_61/LeakyRelu	LeakyRelu$conv2d_transpose_19/BiasAdd:output:0*/
_output_shapes
:?????????@@ n
conv2d_transpose_20/ShapeShape%activation_61/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_20/strided_sliceStridedSlice"conv2d_transpose_20/Shape:output:00conv2d_transpose_20/strided_slice/stack:output:02conv2d_transpose_20/strided_slice/stack_1:output:02conv2d_transpose_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_20/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?^
conv2d_transpose_20/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?]
conv2d_transpose_20/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_20/stackPack*conv2d_transpose_20/strided_slice:output:0$conv2d_transpose_20/stack/1:output:0$conv2d_transpose_20/stack/2:output:0$conv2d_transpose_20/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_20/strided_slice_1StridedSlice"conv2d_transpose_20/stack:output:02conv2d_transpose_20/strided_slice_1/stack:output:04conv2d_transpose_20/strided_slice_1/stack_1:output:04conv2d_transpose_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_20/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_20_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
$conv2d_transpose_20/conv2d_transposeConv2DBackpropInput"conv2d_transpose_20/stack:output:0;conv2d_transpose_20/conv2d_transpose/ReadVariableOp:value:0%activation_61/LeakyRelu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
*conv2d_transpose_20/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_20/BiasAddBiasAdd-conv2d_transpose_20/conv2d_transpose:output:02conv2d_transpose_20/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????}
activation_62/LeakyRelu	LeakyRelu$conv2d_transpose_20/BiasAdd:output:0*1
_output_shapes
:???????????b
decoded/ShapeShape%activation_62/LeakyRelu:activations:0*
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
:*
dtype0?
decoded/conv2d_transposeConv2DBackpropInputdecoded/stack:output:0/decoded/conv2d_transpose/ReadVariableOp:value:0%activation_62/LeakyRelu:activations:0*
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
:???????????j
decoded/TanhTanhdecoded/BiasAdd:output:0*
T0*1
_output_shapes
:???????????i
IdentityIdentitydecoded/Tanh:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp&^batch_normalization_42/AssignNewValue(^batch_normalization_42/AssignNewValue_17^batch_normalization_42/FusedBatchNormV3/ReadVariableOp9^batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_42/ReadVariableOp(^batch_normalization_42/ReadVariableOp_1&^batch_normalization_43/AssignNewValue(^batch_normalization_43/AssignNewValue_17^batch_normalization_43/FusedBatchNormV3/ReadVariableOp9^batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_43/ReadVariableOp(^batch_normalization_43/ReadVariableOp_1&^batch_normalization_44/AssignNewValue(^batch_normalization_44/AssignNewValue_17^batch_normalization_44/FusedBatchNormV3/ReadVariableOp9^batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_44/ReadVariableOp(^batch_normalization_44/ReadVariableOp_1&^batch_normalization_45/AssignNewValue(^batch_normalization_45/AssignNewValue_17^batch_normalization_45/FusedBatchNormV3/ReadVariableOp9^batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_45/ReadVariableOp(^batch_normalization_45/ReadVariableOp_1&^batch_normalization_46/AssignNewValue(^batch_normalization_46/AssignNewValue_17^batch_normalization_46/FusedBatchNormV3/ReadVariableOp9^batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_46/ReadVariableOp(^batch_normalization_46/ReadVariableOp_1&^batch_normalization_47/AssignNewValue(^batch_normalization_47/AssignNewValue_17^batch_normalization_47/FusedBatchNormV3/ReadVariableOp9^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_47/ReadVariableOp(^batch_normalization_47/ReadVariableOp_1&^batch_normalization_48/AssignNewValue(^batch_normalization_48/AssignNewValue_17^batch_normalization_48/FusedBatchNormV3/ReadVariableOp9^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_48/ReadVariableOp(^batch_normalization_48/ReadVariableOp_1!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp!^conv2d_43/BiasAdd/ReadVariableOp ^conv2d_43/Conv2D/ReadVariableOp!^conv2d_44/BiasAdd/ReadVariableOp ^conv2d_44/Conv2D/ReadVariableOp!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp!^conv2d_46/BiasAdd/ReadVariableOp ^conv2d_46/Conv2D/ReadVariableOp!^conv2d_47/BiasAdd/ReadVariableOp ^conv2d_47/Conv2D/ReadVariableOp!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp+^conv2d_transpose_18/BiasAdd/ReadVariableOp4^conv2d_transpose_18/conv2d_transpose/ReadVariableOp+^conv2d_transpose_19/BiasAdd/ReadVariableOp4^conv2d_transpose_19/conv2d_transpose/ReadVariableOp+^conv2d_transpose_20/BiasAdd/ReadVariableOp4^conv2d_transpose_20/conv2d_transpose/ReadVariableOp^decoded/BiasAdd/ReadVariableOp(^decoded/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_42/AssignNewValue%batch_normalization_42/AssignNewValue2R
'batch_normalization_42/AssignNewValue_1'batch_normalization_42/AssignNewValue_12p
6batch_normalization_42/FusedBatchNormV3/ReadVariableOp6batch_normalization_42/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_18batch_normalization_42/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_42/ReadVariableOp%batch_normalization_42/ReadVariableOp2R
'batch_normalization_42/ReadVariableOp_1'batch_normalization_42/ReadVariableOp_12N
%batch_normalization_43/AssignNewValue%batch_normalization_43/AssignNewValue2R
'batch_normalization_43/AssignNewValue_1'batch_normalization_43/AssignNewValue_12p
6batch_normalization_43/FusedBatchNormV3/ReadVariableOp6batch_normalization_43/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_18batch_normalization_43/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_43/ReadVariableOp%batch_normalization_43/ReadVariableOp2R
'batch_normalization_43/ReadVariableOp_1'batch_normalization_43/ReadVariableOp_12N
%batch_normalization_44/AssignNewValue%batch_normalization_44/AssignNewValue2R
'batch_normalization_44/AssignNewValue_1'batch_normalization_44/AssignNewValue_12p
6batch_normalization_44/FusedBatchNormV3/ReadVariableOp6batch_normalization_44/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_18batch_normalization_44/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_44/ReadVariableOp%batch_normalization_44/ReadVariableOp2R
'batch_normalization_44/ReadVariableOp_1'batch_normalization_44/ReadVariableOp_12N
%batch_normalization_45/AssignNewValue%batch_normalization_45/AssignNewValue2R
'batch_normalization_45/AssignNewValue_1'batch_normalization_45/AssignNewValue_12p
6batch_normalization_45/FusedBatchNormV3/ReadVariableOp6batch_normalization_45/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_18batch_normalization_45/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_45/ReadVariableOp%batch_normalization_45/ReadVariableOp2R
'batch_normalization_45/ReadVariableOp_1'batch_normalization_45/ReadVariableOp_12N
%batch_normalization_46/AssignNewValue%batch_normalization_46/AssignNewValue2R
'batch_normalization_46/AssignNewValue_1'batch_normalization_46/AssignNewValue_12p
6batch_normalization_46/FusedBatchNormV3/ReadVariableOp6batch_normalization_46/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_18batch_normalization_46/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_46/ReadVariableOp%batch_normalization_46/ReadVariableOp2R
'batch_normalization_46/ReadVariableOp_1'batch_normalization_46/ReadVariableOp_12N
%batch_normalization_47/AssignNewValue%batch_normalization_47/AssignNewValue2R
'batch_normalization_47/AssignNewValue_1'batch_normalization_47/AssignNewValue_12p
6batch_normalization_47/FusedBatchNormV3/ReadVariableOp6batch_normalization_47/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_18batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_47/ReadVariableOp%batch_normalization_47/ReadVariableOp2R
'batch_normalization_47/ReadVariableOp_1'batch_normalization_47/ReadVariableOp_12N
%batch_normalization_48/AssignNewValue%batch_normalization_48/AssignNewValue2R
'batch_normalization_48/AssignNewValue_1'batch_normalization_48/AssignNewValue_12p
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp6batch_normalization_48/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_18batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_48/ReadVariableOp%batch_normalization_48/ReadVariableOp2R
'batch_normalization_48/ReadVariableOp_1'batch_normalization_48/ReadVariableOp_12D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp2D
 conv2d_43/BiasAdd/ReadVariableOp conv2d_43/BiasAdd/ReadVariableOp2B
conv2d_43/Conv2D/ReadVariableOpconv2d_43/Conv2D/ReadVariableOp2D
 conv2d_44/BiasAdd/ReadVariableOp conv2d_44/BiasAdd/ReadVariableOp2B
conv2d_44/Conv2D/ReadVariableOpconv2d_44/Conv2D/ReadVariableOp2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp2D
 conv2d_46/BiasAdd/ReadVariableOp conv2d_46/BiasAdd/ReadVariableOp2B
conv2d_46/Conv2D/ReadVariableOpconv2d_46/Conv2D/ReadVariableOp2D
 conv2d_47/BiasAdd/ReadVariableOp conv2d_47/BiasAdd/ReadVariableOp2B
conv2d_47/Conv2D/ReadVariableOpconv2d_47/Conv2D/ReadVariableOp2D
 conv2d_48/BiasAdd/ReadVariableOp conv2d_48/BiasAdd/ReadVariableOp2B
conv2d_48/Conv2D/ReadVariableOpconv2d_48/Conv2D/ReadVariableOp2X
*conv2d_transpose_18/BiasAdd/ReadVariableOp*conv2d_transpose_18/BiasAdd/ReadVariableOp2j
3conv2d_transpose_18/conv2d_transpose/ReadVariableOp3conv2d_transpose_18/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_19/BiasAdd/ReadVariableOp*conv2d_transpose_19/BiasAdd/ReadVariableOp2j
3conv2d_transpose_19/conv2d_transpose/ReadVariableOp3conv2d_transpose_19/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_20/BiasAdd/ReadVariableOp*conv2d_transpose_20/BiasAdd/ReadVariableOp2j
3conv2d_transpose_20/conv2d_transpose/ReadVariableOp3conv2d_transpose_20/conv2d_transpose/ReadVariableOp2@
decoded/BiasAdd/ReadVariableOpdecoded/BiasAdd/ReadVariableOp2R
'decoded/conv2d_transpose/ReadVariableOp'decoded/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
f
J__inference_activation_57_layer_call_and_return_conditional_losses_2598392

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@@ g
IdentityIdentityLeakyRelu:activations:0*
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
?
?
+__inference_conv2d_44_layer_call_fn_2598219

inputs!
unknown:
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_44_layer_call_and_return_conditional_losses_2596095y
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
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?!
?
D__inference_decoded_layer_call_and_return_conditional_losses_2596007

inputsB
(conv2d_transpose_readvariableop_resource:-
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
:*
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
-:+???????????????????????????j
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????q
IdentityIdentityTanh:y:0^NoOp*
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
?

?
F__inference_conv2d_47_layer_call_and_return_conditional_losses_2598502

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
?
?
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2598655

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
f
J__inference_activation_59_layer_call_and_return_conditional_losses_2598574

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????  @g
IdentityIdentityLeakyRelu:activations:0*
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
?
?
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2595411

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
f
J__inference_activation_59_layer_call_and_return_conditional_losses_2596211

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????  @g
IdentityIdentityLeakyRelu:activations:0*
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
8__inference_batch_normalization_46_layer_call_fn_2598424

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
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2595667?
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
?
K
/__inference_activation_61_layer_call_fn_2598764

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
GPU2 *0J 8? *S
fNRL
J__inference_activation_61_layer_call_and_return_conditional_losses_2596269h
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
?
?
.__inference_sequential_6_layer_call_fn_2596971
conv2d_42_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23:  

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: $

unknown_29: @

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@$

unknown_35:@ 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: $

unknown_41:@ 

unknown_42:@$

unknown_43: @

unknown_44: $

unknown_45: 

unknown_46:$

unknown_47:

unknown_48:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_42_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*F
_read_only_resource_inputs(
&$	
 !"%&'(+,-./012*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_2596763y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_42_input
?	
?
8__inference_batch_normalization_47_layer_call_fn_2598515

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
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2595731?
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

?
F__inference_conv2d_48_layer_call_and_return_conditional_losses_2598593

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
f
J__inference_activation_54_layer_call_and_return_conditional_losses_2596051

inputs
identityQ
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:???????????i
IdentityIdentityLeakyRelu:activations:0*
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
8__inference_batch_normalization_45_layer_call_fn_2598346

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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2595634?
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
F__inference_conv2d_44_layer_call_and_return_conditional_losses_2596095

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
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
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_48_layer_call_fn_2598606

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
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2595795?
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
8__inference_batch_normalization_43_layer_call_fn_2598151

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
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2595475?
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
?
?
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2598109

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
F__inference_conv2d_42_layer_call_and_return_conditional_losses_2598047

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
??
?
I__inference_sequential_6_layer_call_and_return_conditional_losses_2597239
conv2d_42_input+
conv2d_42_2597108:
conv2d_42_2597110:,
batch_normalization_42_2597113:,
batch_normalization_42_2597115:,
batch_normalization_42_2597117:,
batch_normalization_42_2597119:+
conv2d_43_2597123:
conv2d_43_2597125:,
batch_normalization_43_2597128:,
batch_normalization_43_2597130:,
batch_normalization_43_2597132:,
batch_normalization_43_2597134:+
conv2d_44_2597138:
conv2d_44_2597140:,
batch_normalization_44_2597143:,
batch_normalization_44_2597145:,
batch_normalization_44_2597147:,
batch_normalization_44_2597149:+
conv2d_45_2597153: 
conv2d_45_2597155: ,
batch_normalization_45_2597158: ,
batch_normalization_45_2597160: ,
batch_normalization_45_2597162: ,
batch_normalization_45_2597164: +
conv2d_46_2597168:  
conv2d_46_2597170: ,
batch_normalization_46_2597173: ,
batch_normalization_46_2597175: ,
batch_normalization_46_2597177: ,
batch_normalization_46_2597179: +
conv2d_47_2597183: @
conv2d_47_2597185:@,
batch_normalization_47_2597188:@,
batch_normalization_47_2597190:@,
batch_normalization_47_2597192:@,
batch_normalization_47_2597194:@+
conv2d_48_2597198:@ 
conv2d_48_2597200: ,
batch_normalization_48_2597203: ,
batch_normalization_48_2597205: ,
batch_normalization_48_2597207: ,
batch_normalization_48_2597209: 5
conv2d_transpose_18_2597215:@ )
conv2d_transpose_18_2597217:@5
conv2d_transpose_19_2597221: @)
conv2d_transpose_19_2597223: 5
conv2d_transpose_20_2597227: )
conv2d_transpose_20_2597229:)
decoded_2597233:
decoded_2597235:
identity??.batch_normalization_42/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_44/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?.batch_normalization_46/StatefulPartitionedCall?.batch_normalization_47/StatefulPartitionedCall?.batch_normalization_48/StatefulPartitionedCall?!conv2d_42/StatefulPartitionedCall?!conv2d_43/StatefulPartitionedCall?!conv2d_44/StatefulPartitionedCall?!conv2d_45/StatefulPartitionedCall?!conv2d_46/StatefulPartitionedCall?!conv2d_47/StatefulPartitionedCall?!conv2d_48/StatefulPartitionedCall?+conv2d_transpose_18/StatefulPartitionedCall?+conv2d_transpose_19/StatefulPartitionedCall?+conv2d_transpose_20/StatefulPartitionedCall?decoded/StatefulPartitionedCall?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCallconv2d_42_inputconv2d_42_2597108conv2d_42_2597110*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_42_layer_call_and_return_conditional_losses_2596031?
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0batch_normalization_42_2597113batch_normalization_42_2597115batch_normalization_42_2597117batch_normalization_42_2597119*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2595442?
activation_54/PartitionedCallPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_54_layer_call_and_return_conditional_losses_2596051?
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall&activation_54/PartitionedCall:output:0conv2d_43_2597123conv2d_43_2597125*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_43_layer_call_and_return_conditional_losses_2596063?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0batch_normalization_43_2597128batch_normalization_43_2597130batch_normalization_43_2597132batch_normalization_43_2597134*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2595506?
activation_55/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_55_layer_call_and_return_conditional_losses_2596083?
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCall&activation_55/PartitionedCall:output:0conv2d_44_2597138conv2d_44_2597140*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_44_layer_call_and_return_conditional_losses_2596095?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0batch_normalization_44_2597143batch_normalization_44_2597145batch_normalization_44_2597147batch_normalization_44_2597149*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2595570?
activation_56/PartitionedCallPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_56_layer_call_and_return_conditional_losses_2596115?
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCall&activation_56/PartitionedCall:output:0conv2d_45_2597153conv2d_45_2597155*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_45_layer_call_and_return_conditional_losses_2596127?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0batch_normalization_45_2597158batch_normalization_45_2597160batch_normalization_45_2597162batch_normalization_45_2597164*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2595634?
activation_57/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_57_layer_call_and_return_conditional_losses_2596147?
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCall&activation_57/PartitionedCall:output:0conv2d_46_2597168conv2d_46_2597170*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_46_layer_call_and_return_conditional_losses_2596159?
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0batch_normalization_46_2597173batch_normalization_46_2597175batch_normalization_46_2597177batch_normalization_46_2597179*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2595698?
activation_58/PartitionedCallPartitionedCall7batch_normalization_46/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_58_layer_call_and_return_conditional_losses_2596179?
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall&activation_58/PartitionedCall:output:0conv2d_47_2597183conv2d_47_2597185*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_47_layer_call_and_return_conditional_losses_2596191?
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0batch_normalization_47_2597188batch_normalization_47_2597190batch_normalization_47_2597192batch_normalization_47_2597194*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2595762?
activation_59/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_59_layer_call_and_return_conditional_losses_2596211?
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCall&activation_59/PartitionedCall:output:0conv2d_48_2597198conv2d_48_2597200*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_48_layer_call_and_return_conditional_losses_2596223?
.batch_normalization_48/StatefulPartitionedCallStatefulPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0batch_normalization_48_2597203batch_normalization_48_2597205batch_normalization_48_2597207batch_normalization_48_2597209*
Tin	
2*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2595826?
encoded/CastCast7batch_normalization_48/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
encoded/PartitionedCallPartitionedCallencoded/Cast:y:0*
Tin
2*
Tout
2*
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
GPU2 *0J 8? *M
fHRF
D__inference_encoded_layer_call_and_return_conditional_losses_2596244?
conv2d_transpose_18/CastCast encoded/PartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
+conv2d_transpose_18/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_18/Cast:y:0conv2d_transpose_18_2597215conv2d_transpose_18_2597217*
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
GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_2595874?
activation_60/PartitionedCallPartitionedCall4conv2d_transpose_18/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_60_layer_call_and_return_conditional_losses_2596257?
+conv2d_transpose_19/StatefulPartitionedCallStatefulPartitionedCall&activation_60/PartitionedCall:output:0conv2d_transpose_19_2597221conv2d_transpose_19_2597223*
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
GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_2595918?
activation_61/PartitionedCallPartitionedCall4conv2d_transpose_19/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_61_layer_call_and_return_conditional_losses_2596269?
+conv2d_transpose_20/StatefulPartitionedCallStatefulPartitionedCall&activation_61/PartitionedCall:output:0conv2d_transpose_20_2597227conv2d_transpose_20_2597229*
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
GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_2595962?
activation_62/PartitionedCallPartitionedCall4conv2d_transpose_20/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_62_layer_call_and_return_conditional_losses_2596281?
decoded/StatefulPartitionedCallStatefulPartitionedCall&activation_62/PartitionedCall:output:0decoded_2597233decoded_2597235*
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
GPU2 *0J 8? *M
fHRF
D__inference_decoded_layer_call_and_return_conditional_losses_2596007?
IdentityIdentity(decoded/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_46/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_48/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall"^conv2d_44/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall,^conv2d_transpose_18/StatefulPartitionedCall,^conv2d_transpose_19/StatefulPartitionedCall,^conv2d_transpose_20/StatefulPartitionedCall ^decoded/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_46/StatefulPartitionedCall.batch_normalization_46/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_48/StatefulPartitionedCall.batch_normalization_48/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2Z
+conv2d_transpose_18/StatefulPartitionedCall+conv2d_transpose_18/StatefulPartitionedCall2Z
+conv2d_transpose_19/StatefulPartitionedCall+conv2d_transpose_19/StatefulPartitionedCall2Z
+conv2d_transpose_20/StatefulPartitionedCall+conv2d_transpose_20/StatefulPartitionedCall2B
decoded/StatefulPartitionedCalldecoded/StatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_42_input
?
f
J__inference_activation_57_layer_call_and_return_conditional_losses_2596147

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@@ g
IdentityIdentityLeakyRelu:activations:0*
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
?
?
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2598546

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
? 
?
P__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_2595962

inputsB
(conv2d_transpose_readvariableop_resource: -
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
: *
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
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
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
? 
?
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_2598707

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
-:+???????????????????????????@y
IdentityIdentityBiasAdd:output:0^NoOp*
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
?
?
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2595762

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
??
?
I__inference_sequential_6_layer_call_and_return_conditional_losses_2596289

inputs+
conv2d_42_2596032:
conv2d_42_2596034:,
batch_normalization_42_2596037:,
batch_normalization_42_2596039:,
batch_normalization_42_2596041:,
batch_normalization_42_2596043:+
conv2d_43_2596064:
conv2d_43_2596066:,
batch_normalization_43_2596069:,
batch_normalization_43_2596071:,
batch_normalization_43_2596073:,
batch_normalization_43_2596075:+
conv2d_44_2596096:
conv2d_44_2596098:,
batch_normalization_44_2596101:,
batch_normalization_44_2596103:,
batch_normalization_44_2596105:,
batch_normalization_44_2596107:+
conv2d_45_2596128: 
conv2d_45_2596130: ,
batch_normalization_45_2596133: ,
batch_normalization_45_2596135: ,
batch_normalization_45_2596137: ,
batch_normalization_45_2596139: +
conv2d_46_2596160:  
conv2d_46_2596162: ,
batch_normalization_46_2596165: ,
batch_normalization_46_2596167: ,
batch_normalization_46_2596169: ,
batch_normalization_46_2596171: +
conv2d_47_2596192: @
conv2d_47_2596194:@,
batch_normalization_47_2596197:@,
batch_normalization_47_2596199:@,
batch_normalization_47_2596201:@,
batch_normalization_47_2596203:@+
conv2d_48_2596224:@ 
conv2d_48_2596226: ,
batch_normalization_48_2596229: ,
batch_normalization_48_2596231: ,
batch_normalization_48_2596233: ,
batch_normalization_48_2596235: 5
conv2d_transpose_18_2596247:@ )
conv2d_transpose_18_2596249:@5
conv2d_transpose_19_2596259: @)
conv2d_transpose_19_2596261: 5
conv2d_transpose_20_2596271: )
conv2d_transpose_20_2596273:)
decoded_2596283:
decoded_2596285:
identity??.batch_normalization_42/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_44/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?.batch_normalization_46/StatefulPartitionedCall?.batch_normalization_47/StatefulPartitionedCall?.batch_normalization_48/StatefulPartitionedCall?!conv2d_42/StatefulPartitionedCall?!conv2d_43/StatefulPartitionedCall?!conv2d_44/StatefulPartitionedCall?!conv2d_45/StatefulPartitionedCall?!conv2d_46/StatefulPartitionedCall?!conv2d_47/StatefulPartitionedCall?!conv2d_48/StatefulPartitionedCall?+conv2d_transpose_18/StatefulPartitionedCall?+conv2d_transpose_19/StatefulPartitionedCall?+conv2d_transpose_20/StatefulPartitionedCall?decoded/StatefulPartitionedCall?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_42_2596032conv2d_42_2596034*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_42_layer_call_and_return_conditional_losses_2596031?
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0batch_normalization_42_2596037batch_normalization_42_2596039batch_normalization_42_2596041batch_normalization_42_2596043*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2595411?
activation_54/PartitionedCallPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_54_layer_call_and_return_conditional_losses_2596051?
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall&activation_54/PartitionedCall:output:0conv2d_43_2596064conv2d_43_2596066*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_43_layer_call_and_return_conditional_losses_2596063?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0batch_normalization_43_2596069batch_normalization_43_2596071batch_normalization_43_2596073batch_normalization_43_2596075*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2595475?
activation_55/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_55_layer_call_and_return_conditional_losses_2596083?
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCall&activation_55/PartitionedCall:output:0conv2d_44_2596096conv2d_44_2596098*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_44_layer_call_and_return_conditional_losses_2596095?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0batch_normalization_44_2596101batch_normalization_44_2596103batch_normalization_44_2596105batch_normalization_44_2596107*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2595539?
activation_56/PartitionedCallPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_56_layer_call_and_return_conditional_losses_2596115?
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCall&activation_56/PartitionedCall:output:0conv2d_45_2596128conv2d_45_2596130*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_45_layer_call_and_return_conditional_losses_2596127?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0batch_normalization_45_2596133batch_normalization_45_2596135batch_normalization_45_2596137batch_normalization_45_2596139*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2595603?
activation_57/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_57_layer_call_and_return_conditional_losses_2596147?
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCall&activation_57/PartitionedCall:output:0conv2d_46_2596160conv2d_46_2596162*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_46_layer_call_and_return_conditional_losses_2596159?
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0batch_normalization_46_2596165batch_normalization_46_2596167batch_normalization_46_2596169batch_normalization_46_2596171*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2595667?
activation_58/PartitionedCallPartitionedCall7batch_normalization_46/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_58_layer_call_and_return_conditional_losses_2596179?
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall&activation_58/PartitionedCall:output:0conv2d_47_2596192conv2d_47_2596194*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_47_layer_call_and_return_conditional_losses_2596191?
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0batch_normalization_47_2596197batch_normalization_47_2596199batch_normalization_47_2596201batch_normalization_47_2596203*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2595731?
activation_59/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_59_layer_call_and_return_conditional_losses_2596211?
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCall&activation_59/PartitionedCall:output:0conv2d_48_2596224conv2d_48_2596226*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_48_layer_call_and_return_conditional_losses_2596223?
.batch_normalization_48/StatefulPartitionedCallStatefulPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0batch_normalization_48_2596229batch_normalization_48_2596231batch_normalization_48_2596233batch_normalization_48_2596235*
Tin	
2*
Tout
2*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2595795?
encoded/CastCast7batch_normalization_48/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
encoded/PartitionedCallPartitionedCallencoded/Cast:y:0*
Tin
2*
Tout
2*
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
GPU2 *0J 8? *M
fHRF
D__inference_encoded_layer_call_and_return_conditional_losses_2596244?
conv2d_transpose_18/CastCast encoded/PartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
+conv2d_transpose_18/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_18/Cast:y:0conv2d_transpose_18_2596247conv2d_transpose_18_2596249*
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
GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_2595874?
activation_60/PartitionedCallPartitionedCall4conv2d_transpose_18/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_60_layer_call_and_return_conditional_losses_2596257?
+conv2d_transpose_19/StatefulPartitionedCallStatefulPartitionedCall&activation_60/PartitionedCall:output:0conv2d_transpose_19_2596259conv2d_transpose_19_2596261*
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
GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_2595918?
activation_61/PartitionedCallPartitionedCall4conv2d_transpose_19/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_61_layer_call_and_return_conditional_losses_2596269?
+conv2d_transpose_20/StatefulPartitionedCallStatefulPartitionedCall&activation_61/PartitionedCall:output:0conv2d_transpose_20_2596271conv2d_transpose_20_2596273*
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
GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_2595962?
activation_62/PartitionedCallPartitionedCall4conv2d_transpose_20/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_62_layer_call_and_return_conditional_losses_2596281?
decoded/StatefulPartitionedCallStatefulPartitionedCall&activation_62/PartitionedCall:output:0decoded_2596283decoded_2596285*
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
GPU2 *0J 8? *M
fHRF
D__inference_decoded_layer_call_and_return_conditional_losses_2596007?
IdentityIdentity(decoded/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_46/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_48/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall"^conv2d_44/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall,^conv2d_transpose_18/StatefulPartitionedCall,^conv2d_transpose_19/StatefulPartitionedCall,^conv2d_transpose_20/StatefulPartitionedCall ^decoded/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_46/StatefulPartitionedCall.batch_normalization_46/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_48/StatefulPartitionedCall.batch_normalization_48/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2Z
+conv2d_transpose_18/StatefulPartitionedCall+conv2d_transpose_18/StatefulPartitionedCall2Z
+conv2d_transpose_19/StatefulPartitionedCall+conv2d_transpose_19/StatefulPartitionedCall2Z
+conv2d_transpose_20/StatefulPartitionedCall+conv2d_transpose_20/StatefulPartitionedCall2B
decoded/StatefulPartitionedCalldecoded/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2595795

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
?
?
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2598091

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
?
?
%__inference_signature_wrapper_2597352
conv2d_42_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23:  

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: $

unknown_29: @

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@$

unknown_35:@ 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: $

unknown_41:@ 

unknown_42:@$

unknown_43: @

unknown_44: $

unknown_45: 

unknown_46:$

unknown_47:

unknown_48:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_42_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__wrapped_model_2595389y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_42_input
?	
?
8__inference_batch_normalization_44_layer_call_fn_2598242

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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2595539?
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
?
?
+__inference_conv2d_48_layer_call_fn_2598583

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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_48_layer_call_and_return_conditional_losses_2596223w
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
?

?
F__inference_conv2d_46_layer_call_and_return_conditional_losses_2596159

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
r
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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2598273

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
?
K
/__inference_activation_60_layer_call_fn_2598712

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
GPU2 *0J 8? *S
fNRL
J__inference_activation_60_layer_call_and_return_conditional_losses_2596257h
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
??
?
I__inference_sequential_6_layer_call_and_return_conditional_losses_2596763

inputs+
conv2d_42_2596632:
conv2d_42_2596634:,
batch_normalization_42_2596637:,
batch_normalization_42_2596639:,
batch_normalization_42_2596641:,
batch_normalization_42_2596643:+
conv2d_43_2596647:
conv2d_43_2596649:,
batch_normalization_43_2596652:,
batch_normalization_43_2596654:,
batch_normalization_43_2596656:,
batch_normalization_43_2596658:+
conv2d_44_2596662:
conv2d_44_2596664:,
batch_normalization_44_2596667:,
batch_normalization_44_2596669:,
batch_normalization_44_2596671:,
batch_normalization_44_2596673:+
conv2d_45_2596677: 
conv2d_45_2596679: ,
batch_normalization_45_2596682: ,
batch_normalization_45_2596684: ,
batch_normalization_45_2596686: ,
batch_normalization_45_2596688: +
conv2d_46_2596692:  
conv2d_46_2596694: ,
batch_normalization_46_2596697: ,
batch_normalization_46_2596699: ,
batch_normalization_46_2596701: ,
batch_normalization_46_2596703: +
conv2d_47_2596707: @
conv2d_47_2596709:@,
batch_normalization_47_2596712:@,
batch_normalization_47_2596714:@,
batch_normalization_47_2596716:@,
batch_normalization_47_2596718:@+
conv2d_48_2596722:@ 
conv2d_48_2596724: ,
batch_normalization_48_2596727: ,
batch_normalization_48_2596729: ,
batch_normalization_48_2596731: ,
batch_normalization_48_2596733: 5
conv2d_transpose_18_2596739:@ )
conv2d_transpose_18_2596741:@5
conv2d_transpose_19_2596745: @)
conv2d_transpose_19_2596747: 5
conv2d_transpose_20_2596751: )
conv2d_transpose_20_2596753:)
decoded_2596757:
decoded_2596759:
identity??.batch_normalization_42/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_44/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?.batch_normalization_46/StatefulPartitionedCall?.batch_normalization_47/StatefulPartitionedCall?.batch_normalization_48/StatefulPartitionedCall?!conv2d_42/StatefulPartitionedCall?!conv2d_43/StatefulPartitionedCall?!conv2d_44/StatefulPartitionedCall?!conv2d_45/StatefulPartitionedCall?!conv2d_46/StatefulPartitionedCall?!conv2d_47/StatefulPartitionedCall?!conv2d_48/StatefulPartitionedCall?+conv2d_transpose_18/StatefulPartitionedCall?+conv2d_transpose_19/StatefulPartitionedCall?+conv2d_transpose_20/StatefulPartitionedCall?decoded/StatefulPartitionedCall?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_42_2596632conv2d_42_2596634*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_42_layer_call_and_return_conditional_losses_2596031?
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0batch_normalization_42_2596637batch_normalization_42_2596639batch_normalization_42_2596641batch_normalization_42_2596643*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2595442?
activation_54/PartitionedCallPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_54_layer_call_and_return_conditional_losses_2596051?
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall&activation_54/PartitionedCall:output:0conv2d_43_2596647conv2d_43_2596649*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_43_layer_call_and_return_conditional_losses_2596063?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0batch_normalization_43_2596652batch_normalization_43_2596654batch_normalization_43_2596656batch_normalization_43_2596658*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2595506?
activation_55/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_55_layer_call_and_return_conditional_losses_2596083?
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCall&activation_55/PartitionedCall:output:0conv2d_44_2596662conv2d_44_2596664*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_44_layer_call_and_return_conditional_losses_2596095?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0batch_normalization_44_2596667batch_normalization_44_2596669batch_normalization_44_2596671batch_normalization_44_2596673*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2595570?
activation_56/PartitionedCallPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_56_layer_call_and_return_conditional_losses_2596115?
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCall&activation_56/PartitionedCall:output:0conv2d_45_2596677conv2d_45_2596679*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_45_layer_call_and_return_conditional_losses_2596127?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0batch_normalization_45_2596682batch_normalization_45_2596684batch_normalization_45_2596686batch_normalization_45_2596688*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2595634?
activation_57/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_57_layer_call_and_return_conditional_losses_2596147?
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCall&activation_57/PartitionedCall:output:0conv2d_46_2596692conv2d_46_2596694*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_46_layer_call_and_return_conditional_losses_2596159?
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0batch_normalization_46_2596697batch_normalization_46_2596699batch_normalization_46_2596701batch_normalization_46_2596703*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2595698?
activation_58/PartitionedCallPartitionedCall7batch_normalization_46/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_58_layer_call_and_return_conditional_losses_2596179?
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall&activation_58/PartitionedCall:output:0conv2d_47_2596707conv2d_47_2596709*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_47_layer_call_and_return_conditional_losses_2596191?
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0batch_normalization_47_2596712batch_normalization_47_2596714batch_normalization_47_2596716batch_normalization_47_2596718*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2595762?
activation_59/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_59_layer_call_and_return_conditional_losses_2596211?
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCall&activation_59/PartitionedCall:output:0conv2d_48_2596722conv2d_48_2596724*
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
GPU2 *0J 8? *O
fJRH
F__inference_conv2d_48_layer_call_and_return_conditional_losses_2596223?
.batch_normalization_48/StatefulPartitionedCallStatefulPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0batch_normalization_48_2596727batch_normalization_48_2596729batch_normalization_48_2596731batch_normalization_48_2596733*
Tin	
2*
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
GPU2 *0J 8? *\
fWRU
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2595826?
encoded/CastCast7batch_normalization_48/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
encoded/PartitionedCallPartitionedCallencoded/Cast:y:0*
Tin
2*
Tout
2*
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
GPU2 *0J 8? *M
fHRF
D__inference_encoded_layer_call_and_return_conditional_losses_2596244?
conv2d_transpose_18/CastCast encoded/PartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
+conv2d_transpose_18/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_18/Cast:y:0conv2d_transpose_18_2596739conv2d_transpose_18_2596741*
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
GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_2595874?
activation_60/PartitionedCallPartitionedCall4conv2d_transpose_18/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_60_layer_call_and_return_conditional_losses_2596257?
+conv2d_transpose_19/StatefulPartitionedCallStatefulPartitionedCall&activation_60/PartitionedCall:output:0conv2d_transpose_19_2596745conv2d_transpose_19_2596747*
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
GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_2595918?
activation_61/PartitionedCallPartitionedCall4conv2d_transpose_19/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_61_layer_call_and_return_conditional_losses_2596269?
+conv2d_transpose_20/StatefulPartitionedCallStatefulPartitionedCall&activation_61/PartitionedCall:output:0conv2d_transpose_20_2596751conv2d_transpose_20_2596753*
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
GPU2 *0J 8? *Y
fTRR
P__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_2595962?
activation_62/PartitionedCallPartitionedCall4conv2d_transpose_20/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8? *S
fNRL
J__inference_activation_62_layer_call_and_return_conditional_losses_2596281?
decoded/StatefulPartitionedCallStatefulPartitionedCall&activation_62/PartitionedCall:output:0decoded_2596757decoded_2596759*
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
GPU2 *0J 8? *M
fHRF
D__inference_decoded_layer_call_and_return_conditional_losses_2596007?
IdentityIdentity(decoded/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_46/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_48/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall"^conv2d_44/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall,^conv2d_transpose_18/StatefulPartitionedCall,^conv2d_transpose_19/StatefulPartitionedCall,^conv2d_transpose_20/StatefulPartitionedCall ^decoded/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_46/StatefulPartitionedCall.batch_normalization_46/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_48/StatefulPartitionedCall.batch_normalization_48/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2Z
+conv2d_transpose_18/StatefulPartitionedCall+conv2d_transpose_18/StatefulPartitionedCall2Z
+conv2d_transpose_19/StatefulPartitionedCall+conv2d_transpose_19/StatefulPartitionedCall2Z
+conv2d_transpose_20/StatefulPartitionedCall+conv2d_transpose_20/StatefulPartitionedCall2B
decoded/StatefulPartitionedCalldecoded/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2595539

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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2595634

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
?

?
F__inference_conv2d_45_layer_call_and_return_conditional_losses_2596127

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
F__inference_conv2d_42_layer_call_and_return_conditional_losses_2596031

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
?
K
/__inference_activation_62_layer_call_fn_2598816

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
GPU2 *0J 8? *S
fNRL
J__inference_activation_62_layer_call_and_return_conditional_losses_2596281j
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
?
f
J__inference_activation_55_layer_call_and_return_conditional_losses_2596083

inputs
identityQ
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:???????????i
IdentityIdentityLeakyRelu:activations:0*
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
ܧ
?-
I__inference_sequential_6_layer_call_and_return_conditional_losses_2597795

inputsB
(conv2d_42_conv2d_readvariableop_resource:7
)conv2d_42_biasadd_readvariableop_resource:<
.batch_normalization_42_readvariableop_resource:>
0batch_normalization_42_readvariableop_1_resource:M
?batch_normalization_42_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_42_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_43_conv2d_readvariableop_resource:7
)conv2d_43_biasadd_readvariableop_resource:<
.batch_normalization_43_readvariableop_resource:>
0batch_normalization_43_readvariableop_1_resource:M
?batch_normalization_43_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_44_conv2d_readvariableop_resource:7
)conv2d_44_biasadd_readvariableop_resource:<
.batch_normalization_44_readvariableop_resource:>
0batch_normalization_44_readvariableop_1_resource:M
?batch_normalization_44_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_45_conv2d_readvariableop_resource: 7
)conv2d_45_biasadd_readvariableop_resource: <
.batch_normalization_45_readvariableop_resource: >
0batch_normalization_45_readvariableop_1_resource: M
?batch_normalization_45_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_46_conv2d_readvariableop_resource:  7
)conv2d_46_biasadd_readvariableop_resource: <
.batch_normalization_46_readvariableop_resource: >
0batch_normalization_46_readvariableop_1_resource: M
?batch_normalization_46_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_47_conv2d_readvariableop_resource: @7
)conv2d_47_biasadd_readvariableop_resource:@<
.batch_normalization_47_readvariableop_resource:@>
0batch_normalization_47_readvariableop_1_resource:@M
?batch_normalization_47_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource:@B
(conv2d_48_conv2d_readvariableop_resource:@ 7
)conv2d_48_biasadd_readvariableop_resource: <
.batch_normalization_48_readvariableop_resource: >
0batch_normalization_48_readvariableop_1_resource: M
?batch_normalization_48_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_18_conv2d_transpose_readvariableop_resource:@ A
3conv2d_transpose_18_biasadd_readvariableop_resource:@V
<conv2d_transpose_19_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_19_biasadd_readvariableop_resource: V
<conv2d_transpose_20_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_20_biasadd_readvariableop_resource:J
0decoded_conv2d_transpose_readvariableop_resource:5
'decoded_biasadd_readvariableop_resource:
identity??6batch_normalization_42/FusedBatchNormV3/ReadVariableOp?8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_42/ReadVariableOp?'batch_normalization_42/ReadVariableOp_1?6batch_normalization_43/FusedBatchNormV3/ReadVariableOp?8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_43/ReadVariableOp?'batch_normalization_43/ReadVariableOp_1?6batch_normalization_44/FusedBatchNormV3/ReadVariableOp?8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_44/ReadVariableOp?'batch_normalization_44/ReadVariableOp_1?6batch_normalization_45/FusedBatchNormV3/ReadVariableOp?8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_45/ReadVariableOp?'batch_normalization_45/ReadVariableOp_1?6batch_normalization_46/FusedBatchNormV3/ReadVariableOp?8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_46/ReadVariableOp?'batch_normalization_46/ReadVariableOp_1?6batch_normalization_47/FusedBatchNormV3/ReadVariableOp?8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_47/ReadVariableOp?'batch_normalization_47/ReadVariableOp_1?6batch_normalization_48/FusedBatchNormV3/ReadVariableOp?8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_48/ReadVariableOp?'batch_normalization_48/ReadVariableOp_1? conv2d_42/BiasAdd/ReadVariableOp?conv2d_42/Conv2D/ReadVariableOp? conv2d_43/BiasAdd/ReadVariableOp?conv2d_43/Conv2D/ReadVariableOp? conv2d_44/BiasAdd/ReadVariableOp?conv2d_44/Conv2D/ReadVariableOp? conv2d_45/BiasAdd/ReadVariableOp?conv2d_45/Conv2D/ReadVariableOp? conv2d_46/BiasAdd/ReadVariableOp?conv2d_46/Conv2D/ReadVariableOp? conv2d_47/BiasAdd/ReadVariableOp?conv2d_47/Conv2D/ReadVariableOp? conv2d_48/BiasAdd/ReadVariableOp?conv2d_48/Conv2D/ReadVariableOp?*conv2d_transpose_18/BiasAdd/ReadVariableOp?3conv2d_transpose_18/conv2d_transpose/ReadVariableOp?*conv2d_transpose_19/BiasAdd/ReadVariableOp?3conv2d_transpose_19/conv2d_transpose/ReadVariableOp?*conv2d_transpose_20/BiasAdd/ReadVariableOp?3conv2d_transpose_20/conv2d_transpose/ReadVariableOp?decoded/BiasAdd/ReadVariableOp?'decoded/conv2d_transpose/ReadVariableOp?
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_42/Conv2DConv2Dinputs'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
%batch_normalization_42/ReadVariableOpReadVariableOp.batch_normalization_42_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_42/ReadVariableOp_1ReadVariableOp0batch_normalization_42_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_42/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_42_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_42_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_42/FusedBatchNormV3FusedBatchNormV3conv2d_42/BiasAdd:output:0-batch_normalization_42/ReadVariableOp:value:0/batch_normalization_42/ReadVariableOp_1:value:0>batch_normalization_42/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
activation_54/LeakyRelu	LeakyRelu+batch_normalization_42/FusedBatchNormV3:y:0*1
_output_shapes
:????????????
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_43/Conv2DConv2D%activation_54/LeakyRelu:activations:0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
%batch_normalization_43/ReadVariableOpReadVariableOp.batch_normalization_43_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_43/ReadVariableOp_1ReadVariableOp0batch_normalization_43_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_43/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_43_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_43/FusedBatchNormV3FusedBatchNormV3conv2d_43/BiasAdd:output:0-batch_normalization_43/ReadVariableOp:value:0/batch_normalization_43/ReadVariableOp_1:value:0>batch_normalization_43/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
activation_55/LeakyRelu	LeakyRelu+batch_normalization_43/FusedBatchNormV3:y:0*1
_output_shapes
:????????????
conv2d_44/Conv2D/ReadVariableOpReadVariableOp(conv2d_44_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_44/Conv2DConv2D%activation_55/LeakyRelu:activations:0'conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
 conv2d_44/BiasAdd/ReadVariableOpReadVariableOp)conv2d_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_44/BiasAddBiasAddconv2d_44/Conv2D:output:0(conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
%batch_normalization_44/ReadVariableOpReadVariableOp.batch_normalization_44_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_44/ReadVariableOp_1ReadVariableOp0batch_normalization_44_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_44/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_44_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_44/FusedBatchNormV3FusedBatchNormV3conv2d_44/BiasAdd:output:0-batch_normalization_44/ReadVariableOp:value:0/batch_normalization_44/ReadVariableOp_1:value:0>batch_normalization_44/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
activation_56/LeakyRelu	LeakyRelu+batch_normalization_44/FusedBatchNormV3:y:0*1
_output_shapes
:????????????
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_45/Conv2DConv2D%activation_56/LeakyRelu:activations:0'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
%batch_normalization_45/ReadVariableOpReadVariableOp.batch_normalization_45_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_45/ReadVariableOp_1ReadVariableOp0batch_normalization_45_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_45/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_45_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_45/FusedBatchNormV3FusedBatchNormV3conv2d_45/BiasAdd:output:0-batch_normalization_45/ReadVariableOp:value:0/batch_normalization_45/ReadVariableOp_1:value:0>batch_normalization_45/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( ?
activation_57/LeakyRelu	LeakyRelu+batch_normalization_45/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@ ?
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_46/Conv2DConv2D%activation_57/LeakyRelu:activations:0'conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0(conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
%batch_normalization_46/ReadVariableOpReadVariableOp.batch_normalization_46_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_46/ReadVariableOp_1ReadVariableOp0batch_normalization_46_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_46/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_46_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_46/FusedBatchNormV3FusedBatchNormV3conv2d_46/BiasAdd:output:0-batch_normalization_46/ReadVariableOp:value:0/batch_normalization_46/ReadVariableOp_1:value:0>batch_normalization_46/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( ?
activation_58/LeakyRelu	LeakyRelu+batch_normalization_46/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@ ?
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_47/Conv2DConv2D%activation_58/LeakyRelu:activations:0'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
%batch_normalization_47/ReadVariableOpReadVariableOp.batch_normalization_47_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_47/ReadVariableOp_1ReadVariableOp0batch_normalization_47_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_47/FusedBatchNormV3FusedBatchNormV3conv2d_47/BiasAdd:output:0-batch_normalization_47/ReadVariableOp:value:0/batch_normalization_47/ReadVariableOp_1:value:0>batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( ?
activation_59/LeakyRelu	LeakyRelu+batch_normalization_47/FusedBatchNormV3:y:0*/
_output_shapes
:?????????  @?
conv2d_48/Conv2D/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
conv2d_48/Conv2DConv2D%activation_59/LeakyRelu:activations:0'conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_48/BiasAdd/ReadVariableOpReadVariableOp)conv2d_48_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_48/BiasAddBiasAddconv2d_48/Conv2D:output:0(conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
%batch_normalization_48/ReadVariableOpReadVariableOp.batch_normalization_48_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_48/ReadVariableOp_1ReadVariableOp0batch_normalization_48_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_48/FusedBatchNormV3FusedBatchNormV3conv2d_48/BiasAdd:output:0-batch_normalization_48/ReadVariableOp:value:0/batch_normalization_48/ReadVariableOp_1:value:0>batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
encoded/CastCast+batch_normalization_48/FusedBatchNormV3:y:0*

DstT0*

SrcT0*/
_output_shapes
:????????? j
encoded/LeakyRelu	LeakyReluencoded/Cast:y:0*
T0*/
_output_shapes
:????????? ?
conv2d_transpose_18/CastCastencoded/LeakyRelu:activations:0*

DstT0*

SrcT0*/
_output_shapes
:????????? e
conv2d_transpose_18/ShapeShapeconv2d_transpose_18/Cast:y:0*
T0*
_output_shapes
:q
'conv2d_transpose_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_18/strided_sliceStridedSlice"conv2d_transpose_18/Shape:output:00conv2d_transpose_18/strided_slice/stack:output:02conv2d_transpose_18/strided_slice/stack_1:output:02conv2d_transpose_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_18/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_18/stack/2Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_18/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose_18/stackPack*conv2d_transpose_18/strided_slice:output:0$conv2d_transpose_18/stack/1:output:0$conv2d_transpose_18/stack/2:output:0$conv2d_transpose_18/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_18/strided_slice_1StridedSlice"conv2d_transpose_18/stack:output:02conv2d_transpose_18/strided_slice_1/stack:output:04conv2d_transpose_18/strided_slice_1/stack_1:output:04conv2d_transpose_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_18/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_18_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
$conv2d_transpose_18/conv2d_transposeConv2DBackpropInput"conv2d_transpose_18/stack:output:0;conv2d_transpose_18/conv2d_transpose/ReadVariableOp:value:0conv2d_transpose_18/Cast:y:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
*conv2d_transpose_18/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_transpose_18/BiasAddBiasAdd-conv2d_transpose_18/conv2d_transpose:output:02conv2d_transpose_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @{
activation_60/LeakyRelu	LeakyRelu$conv2d_transpose_18/BiasAdd:output:0*/
_output_shapes
:?????????  @n
conv2d_transpose_19/ShapeShape%activation_60/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_19/strided_sliceStridedSlice"conv2d_transpose_19/Shape:output:00conv2d_transpose_19/strided_slice/stack:output:02conv2d_transpose_19/strided_slice/stack_1:output:02conv2d_transpose_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_19/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@]
conv2d_transpose_19/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@]
conv2d_transpose_19/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_19/stackPack*conv2d_transpose_19/strided_slice:output:0$conv2d_transpose_19/stack/1:output:0$conv2d_transpose_19/stack/2:output:0$conv2d_transpose_19/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_19/strided_slice_1StridedSlice"conv2d_transpose_19/stack:output:02conv2d_transpose_19/strided_slice_1/stack:output:04conv2d_transpose_19/strided_slice_1/stack_1:output:04conv2d_transpose_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_19/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_19_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
$conv2d_transpose_19/conv2d_transposeConv2DBackpropInput"conv2d_transpose_19/stack:output:0;conv2d_transpose_19/conv2d_transpose/ReadVariableOp:value:0%activation_60/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
*conv2d_transpose_19/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_19/BiasAddBiasAdd-conv2d_transpose_19/conv2d_transpose:output:02conv2d_transpose_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ {
activation_61/LeakyRelu	LeakyRelu$conv2d_transpose_19/BiasAdd:output:0*/
_output_shapes
:?????????@@ n
conv2d_transpose_20/ShapeShape%activation_61/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_20/strided_sliceStridedSlice"conv2d_transpose_20/Shape:output:00conv2d_transpose_20/strided_slice/stack:output:02conv2d_transpose_20/strided_slice/stack_1:output:02conv2d_transpose_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_20/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?^
conv2d_transpose_20/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?]
conv2d_transpose_20/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_20/stackPack*conv2d_transpose_20/strided_slice:output:0$conv2d_transpose_20/stack/1:output:0$conv2d_transpose_20/stack/2:output:0$conv2d_transpose_20/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_20/strided_slice_1StridedSlice"conv2d_transpose_20/stack:output:02conv2d_transpose_20/strided_slice_1/stack:output:04conv2d_transpose_20/strided_slice_1/stack_1:output:04conv2d_transpose_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_20/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_20_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
$conv2d_transpose_20/conv2d_transposeConv2DBackpropInput"conv2d_transpose_20/stack:output:0;conv2d_transpose_20/conv2d_transpose/ReadVariableOp:value:0%activation_61/LeakyRelu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
*conv2d_transpose_20/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_20/BiasAddBiasAdd-conv2d_transpose_20/conv2d_transpose:output:02conv2d_transpose_20/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????}
activation_62/LeakyRelu	LeakyRelu$conv2d_transpose_20/BiasAdd:output:0*1
_output_shapes
:???????????b
decoded/ShapeShape%activation_62/LeakyRelu:activations:0*
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
:*
dtype0?
decoded/conv2d_transposeConv2DBackpropInputdecoded/stack:output:0/decoded/conv2d_transpose/ReadVariableOp:value:0%activation_62/LeakyRelu:activations:0*
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
:???????????j
decoded/TanhTanhdecoded/BiasAdd:output:0*
T0*1
_output_shapes
:???????????i
IdentityIdentitydecoded/Tanh:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp7^batch_normalization_42/FusedBatchNormV3/ReadVariableOp9^batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_42/ReadVariableOp(^batch_normalization_42/ReadVariableOp_17^batch_normalization_43/FusedBatchNormV3/ReadVariableOp9^batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_43/ReadVariableOp(^batch_normalization_43/ReadVariableOp_17^batch_normalization_44/FusedBatchNormV3/ReadVariableOp9^batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_44/ReadVariableOp(^batch_normalization_44/ReadVariableOp_17^batch_normalization_45/FusedBatchNormV3/ReadVariableOp9^batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_45/ReadVariableOp(^batch_normalization_45/ReadVariableOp_17^batch_normalization_46/FusedBatchNormV3/ReadVariableOp9^batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_46/ReadVariableOp(^batch_normalization_46/ReadVariableOp_17^batch_normalization_47/FusedBatchNormV3/ReadVariableOp9^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_47/ReadVariableOp(^batch_normalization_47/ReadVariableOp_17^batch_normalization_48/FusedBatchNormV3/ReadVariableOp9^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_48/ReadVariableOp(^batch_normalization_48/ReadVariableOp_1!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp!^conv2d_43/BiasAdd/ReadVariableOp ^conv2d_43/Conv2D/ReadVariableOp!^conv2d_44/BiasAdd/ReadVariableOp ^conv2d_44/Conv2D/ReadVariableOp!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp!^conv2d_46/BiasAdd/ReadVariableOp ^conv2d_46/Conv2D/ReadVariableOp!^conv2d_47/BiasAdd/ReadVariableOp ^conv2d_47/Conv2D/ReadVariableOp!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp+^conv2d_transpose_18/BiasAdd/ReadVariableOp4^conv2d_transpose_18/conv2d_transpose/ReadVariableOp+^conv2d_transpose_19/BiasAdd/ReadVariableOp4^conv2d_transpose_19/conv2d_transpose/ReadVariableOp+^conv2d_transpose_20/BiasAdd/ReadVariableOp4^conv2d_transpose_20/conv2d_transpose/ReadVariableOp^decoded/BiasAdd/ReadVariableOp(^decoded/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_42/FusedBatchNormV3/ReadVariableOp6batch_normalization_42/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_18batch_normalization_42/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_42/ReadVariableOp%batch_normalization_42/ReadVariableOp2R
'batch_normalization_42/ReadVariableOp_1'batch_normalization_42/ReadVariableOp_12p
6batch_normalization_43/FusedBatchNormV3/ReadVariableOp6batch_normalization_43/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_18batch_normalization_43/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_43/ReadVariableOp%batch_normalization_43/ReadVariableOp2R
'batch_normalization_43/ReadVariableOp_1'batch_normalization_43/ReadVariableOp_12p
6batch_normalization_44/FusedBatchNormV3/ReadVariableOp6batch_normalization_44/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_18batch_normalization_44/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_44/ReadVariableOp%batch_normalization_44/ReadVariableOp2R
'batch_normalization_44/ReadVariableOp_1'batch_normalization_44/ReadVariableOp_12p
6batch_normalization_45/FusedBatchNormV3/ReadVariableOp6batch_normalization_45/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_18batch_normalization_45/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_45/ReadVariableOp%batch_normalization_45/ReadVariableOp2R
'batch_normalization_45/ReadVariableOp_1'batch_normalization_45/ReadVariableOp_12p
6batch_normalization_46/FusedBatchNormV3/ReadVariableOp6batch_normalization_46/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_18batch_normalization_46/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_46/ReadVariableOp%batch_normalization_46/ReadVariableOp2R
'batch_normalization_46/ReadVariableOp_1'batch_normalization_46/ReadVariableOp_12p
6batch_normalization_47/FusedBatchNormV3/ReadVariableOp6batch_normalization_47/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_18batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_47/ReadVariableOp%batch_normalization_47/ReadVariableOp2R
'batch_normalization_47/ReadVariableOp_1'batch_normalization_47/ReadVariableOp_12p
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp6batch_normalization_48/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_18batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_48/ReadVariableOp%batch_normalization_48/ReadVariableOp2R
'batch_normalization_48/ReadVariableOp_1'batch_normalization_48/ReadVariableOp_12D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp2D
 conv2d_43/BiasAdd/ReadVariableOp conv2d_43/BiasAdd/ReadVariableOp2B
conv2d_43/Conv2D/ReadVariableOpconv2d_43/Conv2D/ReadVariableOp2D
 conv2d_44/BiasAdd/ReadVariableOp conv2d_44/BiasAdd/ReadVariableOp2B
conv2d_44/Conv2D/ReadVariableOpconv2d_44/Conv2D/ReadVariableOp2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp2D
 conv2d_46/BiasAdd/ReadVariableOp conv2d_46/BiasAdd/ReadVariableOp2B
conv2d_46/Conv2D/ReadVariableOpconv2d_46/Conv2D/ReadVariableOp2D
 conv2d_47/BiasAdd/ReadVariableOp conv2d_47/BiasAdd/ReadVariableOp2B
conv2d_47/Conv2D/ReadVariableOpconv2d_47/Conv2D/ReadVariableOp2D
 conv2d_48/BiasAdd/ReadVariableOp conv2d_48/BiasAdd/ReadVariableOp2B
conv2d_48/Conv2D/ReadVariableOpconv2d_48/Conv2D/ReadVariableOp2X
*conv2d_transpose_18/BiasAdd/ReadVariableOp*conv2d_transpose_18/BiasAdd/ReadVariableOp2j
3conv2d_transpose_18/conv2d_transpose/ReadVariableOp3conv2d_transpose_18/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_19/BiasAdd/ReadVariableOp*conv2d_transpose_19/BiasAdd/ReadVariableOp2j
3conv2d_transpose_19/conv2d_transpose/ReadVariableOp3conv2d_transpose_19/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_20/BiasAdd/ReadVariableOp*conv2d_transpose_20/BiasAdd/ReadVariableOp2j
3conv2d_transpose_20/conv2d_transpose/ReadVariableOp3conv2d_transpose_20/conv2d_transpose/ReadVariableOp2@
decoded/BiasAdd/ReadVariableOpdecoded/BiasAdd/ReadVariableOp2R
'decoded/conv2d_transpose/ReadVariableOp'decoded/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs"?	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
U
conv2d_42_inputB
!serving_default_conv2d_42_input:0???????????E
decoded:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer-17
layer_with_weights-12
layer-18
layer_with_weights-13
layer-19
layer-20
layer_with_weights-14
layer-21
layer-22
layer_with_weights-15
layer-23
layer-24
layer_with_weights-16
layer-25
layer-26
layer_with_weights-17
layer-27
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_default_save_signature
$	optimizer
%
signatures"
_tf_keras_sequential
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias
 ._jit_compiled_convolution_op"
_tf_keras_layer
?
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5axis
	6gamma
7beta
8moving_mean
9moving_variance"
_tf_keras_layer
?
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias
 H_jit_compiled_convolution_op"
_tf_keras_layer
?
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance"
_tf_keras_layer
?
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias
 b_jit_compiled_convolution_op"
_tf_keras_layer
?
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses
iaxis
	jgamma
kbeta
lmoving_mean
mmoving_variance"
_tf_keras_layer
?
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
?
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

zkernel
{bias
 |_jit_compiled_convolution_op"
_tf_keras_layer
?
}	variables
~trainable_variables
regularization_losses
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
+?&call_and_return_all_conditional_losses"
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
+?&call_and_return_all_conditional_losses"
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
+?&call_and_return_all_conditional_losses"
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
+?&call_and_return_all_conditional_losses"
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
+?&call_and_return_all_conditional_losses"
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
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
,0
-1
62
73
84
95
F6
G7
P8
Q9
R10
S11
`12
a13
j14
k15
l16
m17
z18
{19
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
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49"
trackable_list_wrapper
?
,0
-1
62
73
F4
G5
P6
Q7
`8
a9
j10
k11
z12
{13
?14
?15
?16
?17
?18
?19
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
?31
?32
?33
?34
?35"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
#_default_save_signature
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
.__inference_sequential_6_layer_call_fn_2596392
.__inference_sequential_6_layer_call_fn_2597457
.__inference_sequential_6_layer_call_fn_2597562
.__inference_sequential_6_layer_call_fn_2596971?
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
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
I__inference_sequential_6_layer_call_and_return_conditional_losses_2597795
I__inference_sequential_6_layer_call_and_return_conditional_losses_2598028
I__inference_sequential_6_layer_call_and_return_conditional_losses_2597105
I__inference_sequential_6_layer_call_and_return_conditional_losses_2597239?
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
 z?trace_0z?trace_1z?trace_2z?trace_3
?B?
"__inference__wrapped_model_2595389conv2d_42_input"?
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
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate,m?-m?6m?7m?Fm?Gm?Pm?Qm?`m?am?jm?km?zm?{m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?,v?-v?6v?7v?Fv?Gv?Pv?Qv?`v?av?jv?kv?zv?{v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
-
?serving_default"
signature_map
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_conv2d_42_layer_call_fn_2598037?
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
F__inference_conv2d_42_layer_call_and_return_conditional_losses_2598047?
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
*:(2conv2d_42/kernel
:2conv2d_42/bias
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
<
60
71
82
93"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
8__inference_batch_normalization_42_layer_call_fn_2598060
8__inference_batch_normalization_42_layer_call_fn_2598073?
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
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2598091
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2598109?
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
*:(2batch_normalization_42/gamma
):'2batch_normalization_42/beta
2:0 (2"batch_normalization_42/moving_mean
6:4 (2&batch_normalization_42/moving_variance
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
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_activation_54_layer_call_fn_2598114?
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
J__inference_activation_54_layer_call_and_return_conditional_losses_2598119?
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
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_conv2d_43_layer_call_fn_2598128?
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
F__inference_conv2d_43_layer_call_and_return_conditional_losses_2598138?
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
*:(2conv2d_43/kernel
:2conv2d_43/bias
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
<
P0
Q1
R2
S3"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
8__inference_batch_normalization_43_layer_call_fn_2598151
8__inference_batch_normalization_43_layer_call_fn_2598164?
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
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2598182
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2598200?
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
*:(2batch_normalization_43/gamma
):'2batch_normalization_43/beta
2:0 (2"batch_normalization_43/moving_mean
6:4 (2&batch_normalization_43/moving_variance
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
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_activation_55_layer_call_fn_2598205?
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
J__inference_activation_55_layer_call_and_return_conditional_losses_2598210?
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
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_conv2d_44_layer_call_fn_2598219?
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
F__inference_conv2d_44_layer_call_and_return_conditional_losses_2598229?
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
*:(2conv2d_44/kernel
:2conv2d_44/bias
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
<
j0
k1
l2
m3"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
8__inference_batch_normalization_44_layer_call_fn_2598242
8__inference_batch_normalization_44_layer_call_fn_2598255?
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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2598273
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2598291?
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
*:(2batch_normalization_44/gamma
):'2batch_normalization_44/beta
2:0 (2"batch_normalization_44/moving_mean
6:4 (2&batch_normalization_44/moving_variance
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
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_activation_56_layer_call_fn_2598296?
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
J__inference_activation_56_layer_call_and_return_conditional_losses_2598301?
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
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_conv2d_45_layer_call_fn_2598310?
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
F__inference_conv2d_45_layer_call_and_return_conditional_losses_2598320?
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
*:( 2conv2d_45/kernel
: 2conv2d_45/bias
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
}	variables
~trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
8__inference_batch_normalization_45_layer_call_fn_2598333
8__inference_batch_normalization_45_layer_call_fn_2598346?
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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2598364
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2598382?
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
*:( 2batch_normalization_45/gamma
):' 2batch_normalization_45/beta
2:0  (2"batch_normalization_45/moving_mean
6:4  (2&batch_normalization_45/moving_variance
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
/__inference_activation_57_layer_call_fn_2598387?
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
?trace_02?
J__inference_activation_57_layer_call_and_return_conditional_losses_2598392?
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
 z?trace_0
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_conv2d_46_layer_call_fn_2598401?
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
 z?trace_0
?
?trace_02?
F__inference_conv2d_46_layer_call_and_return_conditional_losses_2598411?
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
 z?trace_0
*:(  2conv2d_46/kernel
: 2conv2d_46/bias
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
8__inference_batch_normalization_46_layer_call_fn_2598424
8__inference_batch_normalization_46_layer_call_fn_2598437?
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
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2598455
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2598473?
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
 z?trace_0z?trace_1
 "
trackable_list_wrapper
*:( 2batch_normalization_46/gamma
):' 2batch_normalization_46/beta
2:0  (2"batch_normalization_46/moving_mean
6:4  (2&batch_normalization_46/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_activation_58_layer_call_fn_2598478?
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
 z?trace_0
?
?trace_02?
J__inference_activation_58_layer_call_and_return_conditional_losses_2598483?
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
 z?trace_0
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_conv2d_47_layer_call_fn_2598492?
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
 z?trace_0
?
?trace_02?
F__inference_conv2d_47_layer_call_and_return_conditional_losses_2598502?
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
 z?trace_0
*:( @2conv2d_47/kernel
:@2conv2d_47/bias
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
8__inference_batch_normalization_47_layer_call_fn_2598515
8__inference_batch_normalization_47_layer_call_fn_2598528?
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
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2598546
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2598564?
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
 z?trace_0z?trace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_47/gamma
):'@2batch_normalization_47/beta
2:0@ (2"batch_normalization_47/moving_mean
6:4@ (2&batch_normalization_47/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_activation_59_layer_call_fn_2598569?
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
 z?trace_0
?
?trace_02?
J__inference_activation_59_layer_call_and_return_conditional_losses_2598574?
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
 z?trace_0
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_conv2d_48_layer_call_fn_2598583?
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
 z?trace_0
?
?trace_02?
F__inference_conv2d_48_layer_call_and_return_conditional_losses_2598593?
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
 z?trace_0
*:(@ 2conv2d_48/kernel
: 2conv2d_48/bias
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
8__inference_batch_normalization_48_layer_call_fn_2598606
8__inference_batch_normalization_48_layer_call_fn_2598619?
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
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2598637
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2598655?
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
 z?trace_0z?trace_1
 "
trackable_list_wrapper
*:( 2batch_normalization_48/gamma
):' 2batch_normalization_48/beta
2:0  (2"batch_normalization_48/moving_mean
6:4  (2&batch_normalization_48/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_encoded_layer_call_fn_2598660?
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
 z?trace_0
?
?trace_02?
D__inference_encoded_layer_call_and_return_conditional_losses_2598665?
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
 z?trace_0
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
5__inference_conv2d_transpose_18_layer_call_fn_2598674?
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
 z?trace_0
?
?trace_02?
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_2598707?
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
 z?trace_0
4:2@ 2conv2d_transpose_18/kernel
&:$@2conv2d_transpose_18/bias
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_activation_60_layer_call_fn_2598712?
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
 z?trace_0
?
?trace_02?
J__inference_activation_60_layer_call_and_return_conditional_losses_2598717?
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
 z?trace_0
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
5__inference_conv2d_transpose_19_layer_call_fn_2598726?
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
 z?trace_0
?
?trace_02?
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_2598759?
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
 z?trace_0
4:2 @2conv2d_transpose_19/kernel
&:$ 2conv2d_transpose_19/bias
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_activation_61_layer_call_fn_2598764?
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
 z?trace_0
?
?trace_02?
J__inference_activation_61_layer_call_and_return_conditional_losses_2598769?
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
 z?trace_0
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
5__inference_conv2d_transpose_20_layer_call_fn_2598778?
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
 z?trace_0
?
?trace_02?
P__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_2598811?
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
 z?trace_0
4:2 2conv2d_transpose_20/kernel
&:$2conv2d_transpose_20/bias
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_activation_62_layer_call_fn_2598816?
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
 z?trace_0
?
?trace_02?
J__inference_activation_62_layer_call_and_return_conditional_losses_2598821?
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
 z?trace_0
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_decoded_layer_call_fn_2598830?
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
 z?trace_0
?
?trace_02?
D__inference_decoded_layer_call_and_return_conditional_losses_2598864?
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
 z?trace_0
(:&2decoded/kernel
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
?
80
91
R2
S3
l4
m5
?6
?7
?8
?9
?10
?11
?12
?13"
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
17
18
19
20
21
22
23
24
25
26
27"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
.__inference_sequential_6_layer_call_fn_2596392conv2d_42_input"?
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
.__inference_sequential_6_layer_call_fn_2597457inputs"?
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
.__inference_sequential_6_layer_call_fn_2597562inputs"?
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
.__inference_sequential_6_layer_call_fn_2596971conv2d_42_input"?
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
I__inference_sequential_6_layer_call_and_return_conditional_losses_2597795inputs"?
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
I__inference_sequential_6_layer_call_and_return_conditional_losses_2598028inputs"?
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
I__inference_sequential_6_layer_call_and_return_conditional_losses_2597105conv2d_42_input"?
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
I__inference_sequential_6_layer_call_and_return_conditional_losses_2597239conv2d_42_input"?
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
%__inference_signature_wrapper_2597352conv2d_42_input"?
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
+__inference_conv2d_42_layer_call_fn_2598037inputs"?
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
F__inference_conv2d_42_layer_call_and_return_conditional_losses_2598047inputs"?
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
80
91"
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
8__inference_batch_normalization_42_layer_call_fn_2598060inputs"?
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
8__inference_batch_normalization_42_layer_call_fn_2598073inputs"?
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
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2598091inputs"?
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
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2598109inputs"?
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
/__inference_activation_54_layer_call_fn_2598114inputs"?
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
J__inference_activation_54_layer_call_and_return_conditional_losses_2598119inputs"?
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
+__inference_conv2d_43_layer_call_fn_2598128inputs"?
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
F__inference_conv2d_43_layer_call_and_return_conditional_losses_2598138inputs"?
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
R0
S1"
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
8__inference_batch_normalization_43_layer_call_fn_2598151inputs"?
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
8__inference_batch_normalization_43_layer_call_fn_2598164inputs"?
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
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2598182inputs"?
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
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2598200inputs"?
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
/__inference_activation_55_layer_call_fn_2598205inputs"?
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
J__inference_activation_55_layer_call_and_return_conditional_losses_2598210inputs"?
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
+__inference_conv2d_44_layer_call_fn_2598219inputs"?
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
F__inference_conv2d_44_layer_call_and_return_conditional_losses_2598229inputs"?
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
l0
m1"
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
8__inference_batch_normalization_44_layer_call_fn_2598242inputs"?
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
8__inference_batch_normalization_44_layer_call_fn_2598255inputs"?
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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2598273inputs"?
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
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2598291inputs"?
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
/__inference_activation_56_layer_call_fn_2598296inputs"?
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
J__inference_activation_56_layer_call_and_return_conditional_losses_2598301inputs"?
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
+__inference_conv2d_45_layer_call_fn_2598310inputs"?
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
F__inference_conv2d_45_layer_call_and_return_conditional_losses_2598320inputs"?
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
8__inference_batch_normalization_45_layer_call_fn_2598333inputs"?
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
8__inference_batch_normalization_45_layer_call_fn_2598346inputs"?
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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2598364inputs"?
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
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2598382inputs"?
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
/__inference_activation_57_layer_call_fn_2598387inputs"?
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
J__inference_activation_57_layer_call_and_return_conditional_losses_2598392inputs"?
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
+__inference_conv2d_46_layer_call_fn_2598401inputs"?
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
F__inference_conv2d_46_layer_call_and_return_conditional_losses_2598411inputs"?
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
8__inference_batch_normalization_46_layer_call_fn_2598424inputs"?
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
8__inference_batch_normalization_46_layer_call_fn_2598437inputs"?
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
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2598455inputs"?
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
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2598473inputs"?
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
/__inference_activation_58_layer_call_fn_2598478inputs"?
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
J__inference_activation_58_layer_call_and_return_conditional_losses_2598483inputs"?
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
+__inference_conv2d_47_layer_call_fn_2598492inputs"?
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
F__inference_conv2d_47_layer_call_and_return_conditional_losses_2598502inputs"?
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
8__inference_batch_normalization_47_layer_call_fn_2598515inputs"?
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
8__inference_batch_normalization_47_layer_call_fn_2598528inputs"?
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
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2598546inputs"?
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
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2598564inputs"?
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
/__inference_activation_59_layer_call_fn_2598569inputs"?
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
J__inference_activation_59_layer_call_and_return_conditional_losses_2598574inputs"?
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
+__inference_conv2d_48_layer_call_fn_2598583inputs"?
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
F__inference_conv2d_48_layer_call_and_return_conditional_losses_2598593inputs"?
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
8__inference_batch_normalization_48_layer_call_fn_2598606inputs"?
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
8__inference_batch_normalization_48_layer_call_fn_2598619inputs"?
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
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2598637inputs"?
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
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2598655inputs"?
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
)__inference_encoded_layer_call_fn_2598660inputs"?
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
D__inference_encoded_layer_call_and_return_conditional_losses_2598665inputs"?
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
5__inference_conv2d_transpose_18_layer_call_fn_2598674inputs"?
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
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_2598707inputs"?
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
/__inference_activation_60_layer_call_fn_2598712inputs"?
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
J__inference_activation_60_layer_call_and_return_conditional_losses_2598717inputs"?
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
5__inference_conv2d_transpose_19_layer_call_fn_2598726inputs"?
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
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_2598759inputs"?
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
/__inference_activation_61_layer_call_fn_2598764inputs"?
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
J__inference_activation_61_layer_call_and_return_conditional_losses_2598769inputs"?
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
5__inference_conv2d_transpose_20_layer_call_fn_2598778inputs"?
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
P__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_2598811inputs"?
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
/__inference_activation_62_layer_call_fn_2598816inputs"?
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
J__inference_activation_62_layer_call_and_return_conditional_losses_2598821inputs"?
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
)__inference_decoded_layer_call_fn_2598830inputs"?
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
D__inference_decoded_layer_call_and_return_conditional_losses_2598864inputs"?
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
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
/:-2Adam/conv2d_42/kernel/m
!:2Adam/conv2d_42/bias/m
/:-2#Adam/batch_normalization_42/gamma/m
.:,2"Adam/batch_normalization_42/beta/m
/:-2Adam/conv2d_43/kernel/m
!:2Adam/conv2d_43/bias/m
/:-2#Adam/batch_normalization_43/gamma/m
.:,2"Adam/batch_normalization_43/beta/m
/:-2Adam/conv2d_44/kernel/m
!:2Adam/conv2d_44/bias/m
/:-2#Adam/batch_normalization_44/gamma/m
.:,2"Adam/batch_normalization_44/beta/m
/:- 2Adam/conv2d_45/kernel/m
!: 2Adam/conv2d_45/bias/m
/:- 2#Adam/batch_normalization_45/gamma/m
.:, 2"Adam/batch_normalization_45/beta/m
/:-  2Adam/conv2d_46/kernel/m
!: 2Adam/conv2d_46/bias/m
/:- 2#Adam/batch_normalization_46/gamma/m
.:, 2"Adam/batch_normalization_46/beta/m
/:- @2Adam/conv2d_47/kernel/m
!:@2Adam/conv2d_47/bias/m
/:-@2#Adam/batch_normalization_47/gamma/m
.:,@2"Adam/batch_normalization_47/beta/m
/:-@ 2Adam/conv2d_48/kernel/m
!: 2Adam/conv2d_48/bias/m
/:- 2#Adam/batch_normalization_48/gamma/m
.:, 2"Adam/batch_normalization_48/beta/m
9:7@ 2!Adam/conv2d_transpose_18/kernel/m
+:)@2Adam/conv2d_transpose_18/bias/m
9:7 @2!Adam/conv2d_transpose_19/kernel/m
+:) 2Adam/conv2d_transpose_19/bias/m
9:7 2!Adam/conv2d_transpose_20/kernel/m
+:)2Adam/conv2d_transpose_20/bias/m
-:+2Adam/decoded/kernel/m
:2Adam/decoded/bias/m
/:-2Adam/conv2d_42/kernel/v
!:2Adam/conv2d_42/bias/v
/:-2#Adam/batch_normalization_42/gamma/v
.:,2"Adam/batch_normalization_42/beta/v
/:-2Adam/conv2d_43/kernel/v
!:2Adam/conv2d_43/bias/v
/:-2#Adam/batch_normalization_43/gamma/v
.:,2"Adam/batch_normalization_43/beta/v
/:-2Adam/conv2d_44/kernel/v
!:2Adam/conv2d_44/bias/v
/:-2#Adam/batch_normalization_44/gamma/v
.:,2"Adam/batch_normalization_44/beta/v
/:- 2Adam/conv2d_45/kernel/v
!: 2Adam/conv2d_45/bias/v
/:- 2#Adam/batch_normalization_45/gamma/v
.:, 2"Adam/batch_normalization_45/beta/v
/:-  2Adam/conv2d_46/kernel/v
!: 2Adam/conv2d_46/bias/v
/:- 2#Adam/batch_normalization_46/gamma/v
.:, 2"Adam/batch_normalization_46/beta/v
/:- @2Adam/conv2d_47/kernel/v
!:@2Adam/conv2d_47/bias/v
/:-@2#Adam/batch_normalization_47/gamma/v
.:,@2"Adam/batch_normalization_47/beta/v
/:-@ 2Adam/conv2d_48/kernel/v
!: 2Adam/conv2d_48/bias/v
/:- 2#Adam/batch_normalization_48/gamma/v
.:, 2"Adam/batch_normalization_48/beta/v
9:7@ 2!Adam/conv2d_transpose_18/kernel/v
+:)@2Adam/conv2d_transpose_18/bias/v
9:7 @2!Adam/conv2d_transpose_19/kernel/v
+:) 2Adam/conv2d_transpose_19/bias/v
9:7 2!Adam/conv2d_transpose_20/kernel/v
+:)2Adam/conv2d_transpose_20/bias/v
-:+2Adam/decoded/kernel/v
:2Adam/decoded/bias/v?
"__inference__wrapped_model_2595389?P,-6789FGPQRS`ajklmz{??????????????????????????????B??
8?5
3?0
conv2d_42_input???????????
? ";?8
6
decoded+?(
decoded????????????
J__inference_activation_54_layer_call_and_return_conditional_losses_2598119l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
/__inference_activation_54_layer_call_fn_2598114_9?6
/?,
*?'
inputs???????????
? ""?????????????
J__inference_activation_55_layer_call_and_return_conditional_losses_2598210l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
/__inference_activation_55_layer_call_fn_2598205_9?6
/?,
*?'
inputs???????????
? ""?????????????
J__inference_activation_56_layer_call_and_return_conditional_losses_2598301l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
/__inference_activation_56_layer_call_fn_2598296_9?6
/?,
*?'
inputs???????????
? ""?????????????
J__inference_activation_57_layer_call_and_return_conditional_losses_2598392h7?4
-?*
(?%
inputs?????????@@ 
? "-?*
#? 
0?????????@@ 
? ?
/__inference_activation_57_layer_call_fn_2598387[7?4
-?*
(?%
inputs?????????@@ 
? " ??????????@@ ?
J__inference_activation_58_layer_call_and_return_conditional_losses_2598483h7?4
-?*
(?%
inputs?????????@@ 
? "-?*
#? 
0?????????@@ 
? ?
/__inference_activation_58_layer_call_fn_2598478[7?4
-?*
(?%
inputs?????????@@ 
? " ??????????@@ ?
J__inference_activation_59_layer_call_and_return_conditional_losses_2598574h7?4
-?*
(?%
inputs?????????  @
? "-?*
#? 
0?????????  @
? ?
/__inference_activation_59_layer_call_fn_2598569[7?4
-?*
(?%
inputs?????????  @
? " ??????????  @?
J__inference_activation_60_layer_call_and_return_conditional_losses_2598717h7?4
-?*
(?%
inputs?????????  @
? "-?*
#? 
0?????????  @
? ?
/__inference_activation_60_layer_call_fn_2598712[7?4
-?*
(?%
inputs?????????  @
? " ??????????  @?
J__inference_activation_61_layer_call_and_return_conditional_losses_2598769h7?4
-?*
(?%
inputs?????????@@ 
? "-?*
#? 
0?????????@@ 
? ?
/__inference_activation_61_layer_call_fn_2598764[7?4
-?*
(?%
inputs?????????@@ 
? " ??????????@@ ?
J__inference_activation_62_layer_call_and_return_conditional_losses_2598821l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
/__inference_activation_62_layer_call_fn_2598816_9?6
/?,
*?'
inputs???????????
? ""?????????????
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2598091?6789M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_42_layer_call_and_return_conditional_losses_2598109?6789M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_42_layer_call_fn_2598060?6789M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_42_layer_call_fn_2598073?6789M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2598182?PQRSM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_43_layer_call_and_return_conditional_losses_2598200?PQRSM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_43_layer_call_fn_2598151?PQRSM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_43_layer_call_fn_2598164?PQRSM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2598273?jklmM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_44_layer_call_and_return_conditional_losses_2598291?jklmM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_44_layer_call_fn_2598242?jklmM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_44_layer_call_fn_2598255?jklmM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2598364?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
S__inference_batch_normalization_45_layer_call_and_return_conditional_losses_2598382?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
8__inference_batch_normalization_45_layer_call_fn_2598333?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
8__inference_batch_normalization_45_layer_call_fn_2598346?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2598455?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_2598473?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
8__inference_batch_normalization_46_layer_call_fn_2598424?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
8__inference_batch_normalization_46_layer_call_fn_2598437?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2598546?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_2598564?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
8__inference_batch_normalization_47_layer_call_fn_2598515?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
8__inference_batch_normalization_47_layer_call_fn_2598528?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2598637?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_2598655?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
8__inference_batch_normalization_48_layer_call_fn_2598606?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
8__inference_batch_normalization_48_layer_call_fn_2598619?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
F__inference_conv2d_42_layer_call_and_return_conditional_losses_2598047p,-9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
+__inference_conv2d_42_layer_call_fn_2598037c,-9?6
/?,
*?'
inputs???????????
? ""?????????????
F__inference_conv2d_43_layer_call_and_return_conditional_losses_2598138pFG9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
+__inference_conv2d_43_layer_call_fn_2598128cFG9?6
/?,
*?'
inputs???????????
? ""?????????????
F__inference_conv2d_44_layer_call_and_return_conditional_losses_2598229p`a9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
+__inference_conv2d_44_layer_call_fn_2598219c`a9?6
/?,
*?'
inputs???????????
? ""?????????????
F__inference_conv2d_45_layer_call_and_return_conditional_losses_2598320nz{9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????@@ 
? ?
+__inference_conv2d_45_layer_call_fn_2598310az{9?6
/?,
*?'
inputs???????????
? " ??????????@@ ?
F__inference_conv2d_46_layer_call_and_return_conditional_losses_2598411n??7?4
-?*
(?%
inputs?????????@@ 
? "-?*
#? 
0?????????@@ 
? ?
+__inference_conv2d_46_layer_call_fn_2598401a??7?4
-?*
(?%
inputs?????????@@ 
? " ??????????@@ ?
F__inference_conv2d_47_layer_call_and_return_conditional_losses_2598502n??7?4
-?*
(?%
inputs?????????@@ 
? "-?*
#? 
0?????????  @
? ?
+__inference_conv2d_47_layer_call_fn_2598492a??7?4
-?*
(?%
inputs?????????@@ 
? " ??????????  @?
F__inference_conv2d_48_layer_call_and_return_conditional_losses_2598593n??7?4
-?*
(?%
inputs?????????  @
? "-?*
#? 
0????????? 
? ?
+__inference_conv2d_48_layer_call_fn_2598583a??7?4
-?*
(?%
inputs?????????  @
? " ?????????? ?
P__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_2598707???I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????@
? ?
5__inference_conv2d_transpose_18_layer_call_fn_2598674???I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+???????????????????????????@?
P__inference_conv2d_transpose_19_layer_call_and_return_conditional_losses_2598759???I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
5__inference_conv2d_transpose_19_layer_call_fn_2598726???I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
P__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_2598811???I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
5__inference_conv2d_transpose_20_layer_call_fn_2598778???I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
D__inference_decoded_layer_call_and_return_conditional_losses_2598864???I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
)__inference_decoded_layer_call_fn_2598830???I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
D__inference_encoded_layer_call_and_return_conditional_losses_2598665h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
)__inference_encoded_layer_call_fn_2598660[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
I__inference_sequential_6_layer_call_and_return_conditional_losses_2597105?P,-6789FGPQRS`ajklmz{??????????????????????????????J?G
@?=
3?0
conv2d_42_input???????????
p 

 
? "/?,
%?"
0???????????
? ?
I__inference_sequential_6_layer_call_and_return_conditional_losses_2597239?P,-6789FGPQRS`ajklmz{??????????????????????????????J?G
@?=
3?0
conv2d_42_input???????????
p

 
? "/?,
%?"
0???????????
? ?
I__inference_sequential_6_layer_call_and_return_conditional_losses_2597795?P,-6789FGPQRS`ajklmz{??????????????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
I__inference_sequential_6_layer_call_and_return_conditional_losses_2598028?P,-6789FGPQRS`ajklmz{??????????????????????????????A?>
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
.__inference_sequential_6_layer_call_fn_2596392?P,-6789FGPQRS`ajklmz{??????????????????????????????J?G
@?=
3?0
conv2d_42_input???????????
p 

 
? ""?????????????
.__inference_sequential_6_layer_call_fn_2596971?P,-6789FGPQRS`ajklmz{??????????????????????????????J?G
@?=
3?0
conv2d_42_input???????????
p

 
? ""?????????????
.__inference_sequential_6_layer_call_fn_2597457?P,-6789FGPQRS`ajklmz{??????????????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? ""?????????????
.__inference_sequential_6_layer_call_fn_2597562?P,-6789FGPQRS`ajklmz{??????????????????????????????A?>
7?4
*?'
inputs???????????
p

 
? ""?????????????
%__inference_signature_wrapper_2597352?P,-6789FGPQRS`ajklmz{??????????????????????????????U?R
? 
K?H
F
conv2d_42_input3?0
conv2d_42_input???????????";?8
6
decoded+?(
decoded???????????