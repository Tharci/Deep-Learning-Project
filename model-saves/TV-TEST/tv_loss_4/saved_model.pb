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
Adam/conv2d_transpose_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_29/bias/v
?
3Adam/conv2d_transpose_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_29/bias/v*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_29/kernel/v
?
5Adam/conv2d_transpose_29/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_29/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_28/bias/v
?
3Adam/conv2d_transpose_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_28/bias/v*
_output_shapes
: *
dtype0
?
!Adam/conv2d_transpose_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/conv2d_transpose_28/kernel/v
?
5Adam/conv2d_transpose_28/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_28/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_transpose_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/conv2d_transpose_27/bias/v
?
3Adam/conv2d_transpose_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_27/bias/v*
_output_shapes
:@*
dtype0
?
!Adam/conv2d_transpose_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *2
shared_name#!Adam/conv2d_transpose_27/kernel/v
?
5Adam/conv2d_transpose_27/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_27/kernel/v*&
_output_shapes
:@ *
dtype0
?
"Adam/batch_normalization_69/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_69/beta/v
?
6Adam/batch_normalization_69/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_69/beta/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_69/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_69/gamma/v
?
7Adam/batch_normalization_69/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_69/gamma/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_69/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_69/bias/v
{
)Adam/conv2d_69/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_69/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_69/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *(
shared_nameAdam/conv2d_69/kernel/v
?
+Adam/conv2d_69/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_69/kernel/v*&
_output_shapes
:@ *
dtype0
?
"Adam/batch_normalization_68/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_68/beta/v
?
6Adam/batch_normalization_68/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_68/beta/v*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_68/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_68/gamma/v
?
7Adam/batch_normalization_68/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_68/gamma/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_68/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_68/bias/v
{
)Adam/conv2d_68/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_68/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_68/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_68/kernel/v
?
+Adam/conv2d_68/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_68/kernel/v*&
_output_shapes
: @*
dtype0
?
"Adam/batch_normalization_67/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_67/beta/v
?
6Adam/batch_normalization_67/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_67/beta/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_67/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_67/gamma/v
?
7Adam/batch_normalization_67/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_67/gamma/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_67/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_67/bias/v
{
)Adam/conv2d_67/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_67/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_67/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_67/kernel/v
?
+Adam/conv2d_67/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_67/kernel/v*&
_output_shapes
:  *
dtype0
?
"Adam/batch_normalization_66/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_66/beta/v
?
6Adam/batch_normalization_66/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_66/beta/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_66/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_66/gamma/v
?
7Adam/batch_normalization_66/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_66/gamma/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_66/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_66/bias/v
{
)Adam/conv2d_66/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_66/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_66/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_66/kernel/v
?
+Adam/conv2d_66/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_66/kernel/v*&
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_65/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_65/beta/v
?
6Adam/batch_normalization_65/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_65/beta/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_65/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_65/gamma/v
?
7Adam/batch_normalization_65/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_65/gamma/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_65/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_65/bias/v
{
)Adam/conv2d_65/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_65/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_65/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_65/kernel/v
?
+Adam/conv2d_65/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_65/kernel/v*&
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_64/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_64/beta/v
?
6Adam/batch_normalization_64/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_64/beta/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_64/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_64/gamma/v
?
7Adam/batch_normalization_64/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_64/gamma/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_64/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_64/bias/v
{
)Adam/conv2d_64/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_64/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_64/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_64/kernel/v
?
+Adam/conv2d_64/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_64/kernel/v*&
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_63/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_63/beta/v
?
6Adam/batch_normalization_63/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_63/beta/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_63/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_63/gamma/v
?
7Adam/batch_normalization_63/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_63/gamma/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_63/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_63/bias/v
{
)Adam/conv2d_63/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_63/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_63/kernel/v
?
+Adam/conv2d_63/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/kernel/v*&
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
Adam/conv2d_transpose_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_29/bias/m
?
3Adam/conv2d_transpose_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_29/bias/m*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_29/kernel/m
?
5Adam/conv2d_transpose_29/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_29/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_28/bias/m
?
3Adam/conv2d_transpose_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_28/bias/m*
_output_shapes
: *
dtype0
?
!Adam/conv2d_transpose_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/conv2d_transpose_28/kernel/m
?
5Adam/conv2d_transpose_28/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_28/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_transpose_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/conv2d_transpose_27/bias/m
?
3Adam/conv2d_transpose_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_27/bias/m*
_output_shapes
:@*
dtype0
?
!Adam/conv2d_transpose_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *2
shared_name#!Adam/conv2d_transpose_27/kernel/m
?
5Adam/conv2d_transpose_27/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_27/kernel/m*&
_output_shapes
:@ *
dtype0
?
"Adam/batch_normalization_69/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_69/beta/m
?
6Adam/batch_normalization_69/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_69/beta/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_69/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_69/gamma/m
?
7Adam/batch_normalization_69/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_69/gamma/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_69/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_69/bias/m
{
)Adam/conv2d_69/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_69/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_69/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *(
shared_nameAdam/conv2d_69/kernel/m
?
+Adam/conv2d_69/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_69/kernel/m*&
_output_shapes
:@ *
dtype0
?
"Adam/batch_normalization_68/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_68/beta/m
?
6Adam/batch_normalization_68/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_68/beta/m*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_68/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_68/gamma/m
?
7Adam/batch_normalization_68/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_68/gamma/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_68/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_68/bias/m
{
)Adam/conv2d_68/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_68/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_68/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_68/kernel/m
?
+Adam/conv2d_68/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_68/kernel/m*&
_output_shapes
: @*
dtype0
?
"Adam/batch_normalization_67/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_67/beta/m
?
6Adam/batch_normalization_67/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_67/beta/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_67/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_67/gamma/m
?
7Adam/batch_normalization_67/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_67/gamma/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_67/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_67/bias/m
{
)Adam/conv2d_67/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_67/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_67/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_67/kernel/m
?
+Adam/conv2d_67/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_67/kernel/m*&
_output_shapes
:  *
dtype0
?
"Adam/batch_normalization_66/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_66/beta/m
?
6Adam/batch_normalization_66/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_66/beta/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_66/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_66/gamma/m
?
7Adam/batch_normalization_66/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_66/gamma/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_66/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_66/bias/m
{
)Adam/conv2d_66/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_66/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_66/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_66/kernel/m
?
+Adam/conv2d_66/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_66/kernel/m*&
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_65/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_65/beta/m
?
6Adam/batch_normalization_65/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_65/beta/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_65/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_65/gamma/m
?
7Adam/batch_normalization_65/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_65/gamma/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_65/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_65/bias/m
{
)Adam/conv2d_65/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_65/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_65/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_65/kernel/m
?
+Adam/conv2d_65/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_65/kernel/m*&
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_64/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_64/beta/m
?
6Adam/batch_normalization_64/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_64/beta/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_64/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_64/gamma/m
?
7Adam/batch_normalization_64/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_64/gamma/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_64/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_64/bias/m
{
)Adam/conv2d_64/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_64/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_64/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_64/kernel/m
?
+Adam/conv2d_64/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_64/kernel/m*&
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_63/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_63/beta/m
?
6Adam/batch_normalization_63/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_63/beta/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_63/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_63/gamma/m
?
7Adam/batch_normalization_63/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_63/gamma/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_63/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_63/bias/m
{
)Adam/conv2d_63/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_63/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_63/kernel/m
?
+Adam/conv2d_63/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/kernel/m*&
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
conv2d_transpose_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_29/bias
?
,conv2d_transpose_29/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_29/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_29/kernel
?
.conv2d_transpose_29/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_29/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_28/bias
?
,conv2d_transpose_28/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_28/bias*
_output_shapes
: *
dtype0
?
conv2d_transpose_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameconv2d_transpose_28/kernel
?
.conv2d_transpose_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_28/kernel*&
_output_shapes
: @*
dtype0
?
conv2d_transpose_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv2d_transpose_27/bias
?
,conv2d_transpose_27/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_27/bias*
_output_shapes
:@*
dtype0
?
conv2d_transpose_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *+
shared_nameconv2d_transpose_27/kernel
?
.conv2d_transpose_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_27/kernel*&
_output_shapes
:@ *
dtype0
?
&batch_normalization_69/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_69/moving_variance
?
:batch_normalization_69/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_69/moving_variance*
_output_shapes
: *
dtype0
?
"batch_normalization_69/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_69/moving_mean
?
6batch_normalization_69/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_69/moving_mean*
_output_shapes
: *
dtype0
?
batch_normalization_69/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_69/beta
?
/batch_normalization_69/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_69/beta*
_output_shapes
: *
dtype0
?
batch_normalization_69/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_69/gamma
?
0batch_normalization_69/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_69/gamma*
_output_shapes
: *
dtype0
t
conv2d_69/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_69/bias
m
"conv2d_69/bias/Read/ReadVariableOpReadVariableOpconv2d_69/bias*
_output_shapes
: *
dtype0
?
conv2d_69/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *!
shared_nameconv2d_69/kernel
}
$conv2d_69/kernel/Read/ReadVariableOpReadVariableOpconv2d_69/kernel*&
_output_shapes
:@ *
dtype0
?
&batch_normalization_68/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_68/moving_variance
?
:batch_normalization_68/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_68/moving_variance*
_output_shapes
:@*
dtype0
?
"batch_normalization_68/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_68/moving_mean
?
6batch_normalization_68/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_68/moving_mean*
_output_shapes
:@*
dtype0
?
batch_normalization_68/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_68/beta
?
/batch_normalization_68/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_68/beta*
_output_shapes
:@*
dtype0
?
batch_normalization_68/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_68/gamma
?
0batch_normalization_68/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_68/gamma*
_output_shapes
:@*
dtype0
t
conv2d_68/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_68/bias
m
"conv2d_68/bias/Read/ReadVariableOpReadVariableOpconv2d_68/bias*
_output_shapes
:@*
dtype0
?
conv2d_68/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_68/kernel
}
$conv2d_68/kernel/Read/ReadVariableOpReadVariableOpconv2d_68/kernel*&
_output_shapes
: @*
dtype0
?
&batch_normalization_67/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_67/moving_variance
?
:batch_normalization_67/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_67/moving_variance*
_output_shapes
: *
dtype0
?
"batch_normalization_67/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_67/moving_mean
?
6batch_normalization_67/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_67/moving_mean*
_output_shapes
: *
dtype0
?
batch_normalization_67/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_67/beta
?
/batch_normalization_67/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_67/beta*
_output_shapes
: *
dtype0
?
batch_normalization_67/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_67/gamma
?
0batch_normalization_67/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_67/gamma*
_output_shapes
: *
dtype0
t
conv2d_67/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_67/bias
m
"conv2d_67/bias/Read/ReadVariableOpReadVariableOpconv2d_67/bias*
_output_shapes
: *
dtype0
?
conv2d_67/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_67/kernel
}
$conv2d_67/kernel/Read/ReadVariableOpReadVariableOpconv2d_67/kernel*&
_output_shapes
:  *
dtype0
?
&batch_normalization_66/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_66/moving_variance
?
:batch_normalization_66/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_66/moving_variance*
_output_shapes
: *
dtype0
?
"batch_normalization_66/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_66/moving_mean
?
6batch_normalization_66/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_66/moving_mean*
_output_shapes
: *
dtype0
?
batch_normalization_66/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_66/beta
?
/batch_normalization_66/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_66/beta*
_output_shapes
: *
dtype0
?
batch_normalization_66/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_66/gamma
?
0batch_normalization_66/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_66/gamma*
_output_shapes
: *
dtype0
t
conv2d_66/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_66/bias
m
"conv2d_66/bias/Read/ReadVariableOpReadVariableOpconv2d_66/bias*
_output_shapes
: *
dtype0
?
conv2d_66/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_66/kernel
}
$conv2d_66/kernel/Read/ReadVariableOpReadVariableOpconv2d_66/kernel*&
_output_shapes
: *
dtype0
?
&batch_normalization_65/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_65/moving_variance
?
:batch_normalization_65/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_65/moving_variance*
_output_shapes
:*
dtype0
?
"batch_normalization_65/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_65/moving_mean
?
6batch_normalization_65/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_65/moving_mean*
_output_shapes
:*
dtype0
?
batch_normalization_65/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_65/beta
?
/batch_normalization_65/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_65/beta*
_output_shapes
:*
dtype0
?
batch_normalization_65/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_65/gamma
?
0batch_normalization_65/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_65/gamma*
_output_shapes
:*
dtype0
t
conv2d_65/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_65/bias
m
"conv2d_65/bias/Read/ReadVariableOpReadVariableOpconv2d_65/bias*
_output_shapes
:*
dtype0
?
conv2d_65/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_65/kernel
}
$conv2d_65/kernel/Read/ReadVariableOpReadVariableOpconv2d_65/kernel*&
_output_shapes
:*
dtype0
?
&batch_normalization_64/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_64/moving_variance
?
:batch_normalization_64/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_64/moving_variance*
_output_shapes
:*
dtype0
?
"batch_normalization_64/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_64/moving_mean
?
6batch_normalization_64/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_64/moving_mean*
_output_shapes
:*
dtype0
?
batch_normalization_64/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_64/beta
?
/batch_normalization_64/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_64/beta*
_output_shapes
:*
dtype0
?
batch_normalization_64/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_64/gamma
?
0batch_normalization_64/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_64/gamma*
_output_shapes
:*
dtype0
t
conv2d_64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_64/bias
m
"conv2d_64/bias/Read/ReadVariableOpReadVariableOpconv2d_64/bias*
_output_shapes
:*
dtype0
?
conv2d_64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_64/kernel
}
$conv2d_64/kernel/Read/ReadVariableOpReadVariableOpconv2d_64/kernel*&
_output_shapes
:*
dtype0
?
&batch_normalization_63/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_63/moving_variance
?
:batch_normalization_63/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_63/moving_variance*
_output_shapes
:*
dtype0
?
"batch_normalization_63/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_63/moving_mean
?
6batch_normalization_63/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_63/moving_mean*
_output_shapes
:*
dtype0
?
batch_normalization_63/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_63/beta
?
/batch_normalization_63/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_63/beta*
_output_shapes
:*
dtype0
?
batch_normalization_63/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_63/gamma
?
0batch_normalization_63/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_63/gamma*
_output_shapes
:*
dtype0
t
conv2d_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_63/bias
m
"conv2d_63/bias/Read/ReadVariableOpReadVariableOpconv2d_63/bias*
_output_shapes
:*
dtype0
?
conv2d_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_63/kernel
}
$conv2d_63/kernel/Read/ReadVariableOpReadVariableOpconv2d_63/kernel*&
_output_shapes
:*
dtype0
?
serving_default_conv2d_63_inputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_63_inputconv2d_63/kernelconv2d_63/biasbatch_normalization_63/gammabatch_normalization_63/beta"batch_normalization_63/moving_mean&batch_normalization_63/moving_varianceconv2d_64/kernelconv2d_64/biasbatch_normalization_64/gammabatch_normalization_64/beta"batch_normalization_64/moving_mean&batch_normalization_64/moving_varianceconv2d_65/kernelconv2d_65/biasbatch_normalization_65/gammabatch_normalization_65/beta"batch_normalization_65/moving_mean&batch_normalization_65/moving_varianceconv2d_66/kernelconv2d_66/biasbatch_normalization_66/gammabatch_normalization_66/beta"batch_normalization_66/moving_mean&batch_normalization_66/moving_varianceconv2d_67/kernelconv2d_67/biasbatch_normalization_67/gammabatch_normalization_67/beta"batch_normalization_67/moving_mean&batch_normalization_67/moving_varianceconv2d_68/kernelconv2d_68/biasbatch_normalization_68/gammabatch_normalization_68/beta"batch_normalization_68/moving_mean&batch_normalization_68/moving_varianceconv2d_69/kernelconv2d_69/biasbatch_normalization_69/gammabatch_normalization_69/beta"batch_normalization_69/moving_mean&batch_normalization_69/moving_varianceconv2d_transpose_27/kernelconv2d_transpose_27/biasconv2d_transpose_28/kernelconv2d_transpose_28/biasconv2d_transpose_29/kernelconv2d_transpose_29/biasdecoded/kerneldecoded/bias*>
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
%__inference_signature_wrapper_3373296

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
VARIABLE_VALUEconv2d_63/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_63/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_63/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_63/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_63/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_63/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_64/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_64/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_64/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_64/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_64/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_64/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_65/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_65/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_65/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_65/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_65/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_65/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_66/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_66/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_66/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_66/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_66/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_66/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_67/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_67/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_67/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_67/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_67/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_67/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_68/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_68/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_68/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_68/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_68/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE&batch_normalization_68/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_69/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_69/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_69/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_69/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_69/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE&batch_normalization_69/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_transpose_27/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEconv2d_transpose_27/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_transpose_28/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEconv2d_transpose_28/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_transpose_29/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEconv2d_transpose_29/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/conv2d_63/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_63/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_63/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_63/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_64/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_64/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_64/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_64/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_65/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_65/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_65/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_65/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_66/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_66/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_66/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_66/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_67/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_67/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_67/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_67/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_68/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_68/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_68/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_68/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_69/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_69/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_69/gamma/mRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_69/beta/mQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_27/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_27/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_28/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_28/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_29/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_29/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/decoded/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/decoded/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_63/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_63/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_63/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_63/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_64/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_64/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_64/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_64/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_65/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_65/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_65/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_65/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_66/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_66/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_66/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_66/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_67/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_67/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_67/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_67/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_68/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_68/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_68/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_68/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_69/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_69/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_69/gamma/vRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_69/beta/vQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_27/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_27/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_28/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_28/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_29/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_29/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
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
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_63/kernel/Read/ReadVariableOp"conv2d_63/bias/Read/ReadVariableOp0batch_normalization_63/gamma/Read/ReadVariableOp/batch_normalization_63/beta/Read/ReadVariableOp6batch_normalization_63/moving_mean/Read/ReadVariableOp:batch_normalization_63/moving_variance/Read/ReadVariableOp$conv2d_64/kernel/Read/ReadVariableOp"conv2d_64/bias/Read/ReadVariableOp0batch_normalization_64/gamma/Read/ReadVariableOp/batch_normalization_64/beta/Read/ReadVariableOp6batch_normalization_64/moving_mean/Read/ReadVariableOp:batch_normalization_64/moving_variance/Read/ReadVariableOp$conv2d_65/kernel/Read/ReadVariableOp"conv2d_65/bias/Read/ReadVariableOp0batch_normalization_65/gamma/Read/ReadVariableOp/batch_normalization_65/beta/Read/ReadVariableOp6batch_normalization_65/moving_mean/Read/ReadVariableOp:batch_normalization_65/moving_variance/Read/ReadVariableOp$conv2d_66/kernel/Read/ReadVariableOp"conv2d_66/bias/Read/ReadVariableOp0batch_normalization_66/gamma/Read/ReadVariableOp/batch_normalization_66/beta/Read/ReadVariableOp6batch_normalization_66/moving_mean/Read/ReadVariableOp:batch_normalization_66/moving_variance/Read/ReadVariableOp$conv2d_67/kernel/Read/ReadVariableOp"conv2d_67/bias/Read/ReadVariableOp0batch_normalization_67/gamma/Read/ReadVariableOp/batch_normalization_67/beta/Read/ReadVariableOp6batch_normalization_67/moving_mean/Read/ReadVariableOp:batch_normalization_67/moving_variance/Read/ReadVariableOp$conv2d_68/kernel/Read/ReadVariableOp"conv2d_68/bias/Read/ReadVariableOp0batch_normalization_68/gamma/Read/ReadVariableOp/batch_normalization_68/beta/Read/ReadVariableOp6batch_normalization_68/moving_mean/Read/ReadVariableOp:batch_normalization_68/moving_variance/Read/ReadVariableOp$conv2d_69/kernel/Read/ReadVariableOp"conv2d_69/bias/Read/ReadVariableOp0batch_normalization_69/gamma/Read/ReadVariableOp/batch_normalization_69/beta/Read/ReadVariableOp6batch_normalization_69/moving_mean/Read/ReadVariableOp:batch_normalization_69/moving_variance/Read/ReadVariableOp.conv2d_transpose_27/kernel/Read/ReadVariableOp,conv2d_transpose_27/bias/Read/ReadVariableOp.conv2d_transpose_28/kernel/Read/ReadVariableOp,conv2d_transpose_28/bias/Read/ReadVariableOp.conv2d_transpose_29/kernel/Read/ReadVariableOp,conv2d_transpose_29/bias/Read/ReadVariableOp"decoded/kernel/Read/ReadVariableOp decoded/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_63/kernel/m/Read/ReadVariableOp)Adam/conv2d_63/bias/m/Read/ReadVariableOp7Adam/batch_normalization_63/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_63/beta/m/Read/ReadVariableOp+Adam/conv2d_64/kernel/m/Read/ReadVariableOp)Adam/conv2d_64/bias/m/Read/ReadVariableOp7Adam/batch_normalization_64/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_64/beta/m/Read/ReadVariableOp+Adam/conv2d_65/kernel/m/Read/ReadVariableOp)Adam/conv2d_65/bias/m/Read/ReadVariableOp7Adam/batch_normalization_65/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_65/beta/m/Read/ReadVariableOp+Adam/conv2d_66/kernel/m/Read/ReadVariableOp)Adam/conv2d_66/bias/m/Read/ReadVariableOp7Adam/batch_normalization_66/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_66/beta/m/Read/ReadVariableOp+Adam/conv2d_67/kernel/m/Read/ReadVariableOp)Adam/conv2d_67/bias/m/Read/ReadVariableOp7Adam/batch_normalization_67/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_67/beta/m/Read/ReadVariableOp+Adam/conv2d_68/kernel/m/Read/ReadVariableOp)Adam/conv2d_68/bias/m/Read/ReadVariableOp7Adam/batch_normalization_68/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_68/beta/m/Read/ReadVariableOp+Adam/conv2d_69/kernel/m/Read/ReadVariableOp)Adam/conv2d_69/bias/m/Read/ReadVariableOp7Adam/batch_normalization_69/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_69/beta/m/Read/ReadVariableOp5Adam/conv2d_transpose_27/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_27/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_28/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_28/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_29/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_29/bias/m/Read/ReadVariableOp)Adam/decoded/kernel/m/Read/ReadVariableOp'Adam/decoded/bias/m/Read/ReadVariableOp+Adam/conv2d_63/kernel/v/Read/ReadVariableOp)Adam/conv2d_63/bias/v/Read/ReadVariableOp7Adam/batch_normalization_63/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_63/beta/v/Read/ReadVariableOp+Adam/conv2d_64/kernel/v/Read/ReadVariableOp)Adam/conv2d_64/bias/v/Read/ReadVariableOp7Adam/batch_normalization_64/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_64/beta/v/Read/ReadVariableOp+Adam/conv2d_65/kernel/v/Read/ReadVariableOp)Adam/conv2d_65/bias/v/Read/ReadVariableOp7Adam/batch_normalization_65/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_65/beta/v/Read/ReadVariableOp+Adam/conv2d_66/kernel/v/Read/ReadVariableOp)Adam/conv2d_66/bias/v/Read/ReadVariableOp7Adam/batch_normalization_66/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_66/beta/v/Read/ReadVariableOp+Adam/conv2d_67/kernel/v/Read/ReadVariableOp)Adam/conv2d_67/bias/v/Read/ReadVariableOp7Adam/batch_normalization_67/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_67/beta/v/Read/ReadVariableOp+Adam/conv2d_68/kernel/v/Read/ReadVariableOp)Adam/conv2d_68/bias/v/Read/ReadVariableOp7Adam/batch_normalization_68/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_68/beta/v/Read/ReadVariableOp+Adam/conv2d_69/kernel/v/Read/ReadVariableOp)Adam/conv2d_69/bias/v/Read/ReadVariableOp7Adam/batch_normalization_69/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_69/beta/v/Read/ReadVariableOp5Adam/conv2d_transpose_27/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_27/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_28/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_28/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_29/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_29/bias/v/Read/ReadVariableOp)Adam/decoded/kernel/v/Read/ReadVariableOp'Adam/decoded/bias/v/Read/ReadVariableOpConst*?
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
 __inference__traced_save_3375218
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_63/kernelconv2d_63/biasbatch_normalization_63/gammabatch_normalization_63/beta"batch_normalization_63/moving_mean&batch_normalization_63/moving_varianceconv2d_64/kernelconv2d_64/biasbatch_normalization_64/gammabatch_normalization_64/beta"batch_normalization_64/moving_mean&batch_normalization_64/moving_varianceconv2d_65/kernelconv2d_65/biasbatch_normalization_65/gammabatch_normalization_65/beta"batch_normalization_65/moving_mean&batch_normalization_65/moving_varianceconv2d_66/kernelconv2d_66/biasbatch_normalization_66/gammabatch_normalization_66/beta"batch_normalization_66/moving_mean&batch_normalization_66/moving_varianceconv2d_67/kernelconv2d_67/biasbatch_normalization_67/gammabatch_normalization_67/beta"batch_normalization_67/moving_mean&batch_normalization_67/moving_varianceconv2d_68/kernelconv2d_68/biasbatch_normalization_68/gammabatch_normalization_68/beta"batch_normalization_68/moving_mean&batch_normalization_68/moving_varianceconv2d_69/kernelconv2d_69/biasbatch_normalization_69/gammabatch_normalization_69/beta"batch_normalization_69/moving_mean&batch_normalization_69/moving_varianceconv2d_transpose_27/kernelconv2d_transpose_27/biasconv2d_transpose_28/kernelconv2d_transpose_28/biasconv2d_transpose_29/kernelconv2d_transpose_29/biasdecoded/kerneldecoded/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_63/kernel/mAdam/conv2d_63/bias/m#Adam/batch_normalization_63/gamma/m"Adam/batch_normalization_63/beta/mAdam/conv2d_64/kernel/mAdam/conv2d_64/bias/m#Adam/batch_normalization_64/gamma/m"Adam/batch_normalization_64/beta/mAdam/conv2d_65/kernel/mAdam/conv2d_65/bias/m#Adam/batch_normalization_65/gamma/m"Adam/batch_normalization_65/beta/mAdam/conv2d_66/kernel/mAdam/conv2d_66/bias/m#Adam/batch_normalization_66/gamma/m"Adam/batch_normalization_66/beta/mAdam/conv2d_67/kernel/mAdam/conv2d_67/bias/m#Adam/batch_normalization_67/gamma/m"Adam/batch_normalization_67/beta/mAdam/conv2d_68/kernel/mAdam/conv2d_68/bias/m#Adam/batch_normalization_68/gamma/m"Adam/batch_normalization_68/beta/mAdam/conv2d_69/kernel/mAdam/conv2d_69/bias/m#Adam/batch_normalization_69/gamma/m"Adam/batch_normalization_69/beta/m!Adam/conv2d_transpose_27/kernel/mAdam/conv2d_transpose_27/bias/m!Adam/conv2d_transpose_28/kernel/mAdam/conv2d_transpose_28/bias/m!Adam/conv2d_transpose_29/kernel/mAdam/conv2d_transpose_29/bias/mAdam/decoded/kernel/mAdam/decoded/bias/mAdam/conv2d_63/kernel/vAdam/conv2d_63/bias/v#Adam/batch_normalization_63/gamma/v"Adam/batch_normalization_63/beta/vAdam/conv2d_64/kernel/vAdam/conv2d_64/bias/v#Adam/batch_normalization_64/gamma/v"Adam/batch_normalization_64/beta/vAdam/conv2d_65/kernel/vAdam/conv2d_65/bias/v#Adam/batch_normalization_65/gamma/v"Adam/batch_normalization_65/beta/vAdam/conv2d_66/kernel/vAdam/conv2d_66/bias/v#Adam/batch_normalization_66/gamma/v"Adam/batch_normalization_66/beta/vAdam/conv2d_67/kernel/vAdam/conv2d_67/bias/v#Adam/batch_normalization_67/gamma/v"Adam/batch_normalization_67/beta/vAdam/conv2d_68/kernel/vAdam/conv2d_68/bias/v#Adam/batch_normalization_68/gamma/v"Adam/batch_normalization_68/beta/vAdam/conv2d_69/kernel/vAdam/conv2d_69/bias/v#Adam/batch_normalization_69/gamma/v"Adam/batch_normalization_69/beta/v!Adam/conv2d_transpose_27/kernel/vAdam/conv2d_transpose_27/bias/v!Adam/conv2d_transpose_28/kernel/vAdam/conv2d_transpose_28/bias/v!Adam/conv2d_transpose_29/kernel/vAdam/conv2d_transpose_29/bias/vAdam/decoded/kernel/vAdam/decoded/bias/v*?
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
#__inference__traced_restore_3375615??
?
?
.__inference_sequential_9_layer_call_fn_3373506

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
I__inference_sequential_9_layer_call_and_return_conditional_losses_3372707y
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
?
?
+__inference_conv2d_68_layer_call_fn_3374436

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
F__inference_conv2d_68_layer_call_and_return_conditional_losses_3372135w
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
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_3371770

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
F__inference_conv2d_66_layer_call_and_return_conditional_losses_3372071

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
?
f
J__inference_activation_89_layer_call_and_return_conditional_losses_3374765

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
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3374053

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
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3371450

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
F__inference_conv2d_65_layer_call_and_return_conditional_losses_3374173

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
)__inference_decoded_layer_call_fn_3374774

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
D__inference_decoded_layer_call_and_return_conditional_losses_3371951?
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
?	
?
8__inference_batch_normalization_68_layer_call_fn_3374472

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
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_3371706?
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
?
?
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_3374599

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
/__inference_activation_88_layer_call_fn_3374708

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
J__inference_activation_88_layer_call_and_return_conditional_losses_3372213h
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
ܧ
?-
I__inference_sequential_9_layer_call_and_return_conditional_losses_3373739

inputsB
(conv2d_63_conv2d_readvariableop_resource:7
)conv2d_63_biasadd_readvariableop_resource:<
.batch_normalization_63_readvariableop_resource:>
0batch_normalization_63_readvariableop_1_resource:M
?batch_normalization_63_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_63_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_64_conv2d_readvariableop_resource:7
)conv2d_64_biasadd_readvariableop_resource:<
.batch_normalization_64_readvariableop_resource:>
0batch_normalization_64_readvariableop_1_resource:M
?batch_normalization_64_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_64_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_65_conv2d_readvariableop_resource:7
)conv2d_65_biasadd_readvariableop_resource:<
.batch_normalization_65_readvariableop_resource:>
0batch_normalization_65_readvariableop_1_resource:M
?batch_normalization_65_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_65_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_66_conv2d_readvariableop_resource: 7
)conv2d_66_biasadd_readvariableop_resource: <
.batch_normalization_66_readvariableop_resource: >
0batch_normalization_66_readvariableop_1_resource: M
?batch_normalization_66_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_66_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_67_conv2d_readvariableop_resource:  7
)conv2d_67_biasadd_readvariableop_resource: <
.batch_normalization_67_readvariableop_resource: >
0batch_normalization_67_readvariableop_1_resource: M
?batch_normalization_67_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_67_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_68_conv2d_readvariableop_resource: @7
)conv2d_68_biasadd_readvariableop_resource:@<
.batch_normalization_68_readvariableop_resource:@>
0batch_normalization_68_readvariableop_1_resource:@M
?batch_normalization_68_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_68_fusedbatchnormv3_readvariableop_1_resource:@B
(conv2d_69_conv2d_readvariableop_resource:@ 7
)conv2d_69_biasadd_readvariableop_resource: <
.batch_normalization_69_readvariableop_resource: >
0batch_normalization_69_readvariableop_1_resource: M
?batch_normalization_69_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_69_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_27_conv2d_transpose_readvariableop_resource:@ A
3conv2d_transpose_27_biasadd_readvariableop_resource:@V
<conv2d_transpose_28_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_28_biasadd_readvariableop_resource: V
<conv2d_transpose_29_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_29_biasadd_readvariableop_resource:J
0decoded_conv2d_transpose_readvariableop_resource:5
'decoded_biasadd_readvariableop_resource:
identity??6batch_normalization_63/FusedBatchNormV3/ReadVariableOp?8batch_normalization_63/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_63/ReadVariableOp?'batch_normalization_63/ReadVariableOp_1?6batch_normalization_64/FusedBatchNormV3/ReadVariableOp?8batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_64/ReadVariableOp?'batch_normalization_64/ReadVariableOp_1?6batch_normalization_65/FusedBatchNormV3/ReadVariableOp?8batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_65/ReadVariableOp?'batch_normalization_65/ReadVariableOp_1?6batch_normalization_66/FusedBatchNormV3/ReadVariableOp?8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_66/ReadVariableOp?'batch_normalization_66/ReadVariableOp_1?6batch_normalization_67/FusedBatchNormV3/ReadVariableOp?8batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_67/ReadVariableOp?'batch_normalization_67/ReadVariableOp_1?6batch_normalization_68/FusedBatchNormV3/ReadVariableOp?8batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_68/ReadVariableOp?'batch_normalization_68/ReadVariableOp_1?6batch_normalization_69/FusedBatchNormV3/ReadVariableOp?8batch_normalization_69/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_69/ReadVariableOp?'batch_normalization_69/ReadVariableOp_1? conv2d_63/BiasAdd/ReadVariableOp?conv2d_63/Conv2D/ReadVariableOp? conv2d_64/BiasAdd/ReadVariableOp?conv2d_64/Conv2D/ReadVariableOp? conv2d_65/BiasAdd/ReadVariableOp?conv2d_65/Conv2D/ReadVariableOp? conv2d_66/BiasAdd/ReadVariableOp?conv2d_66/Conv2D/ReadVariableOp? conv2d_67/BiasAdd/ReadVariableOp?conv2d_67/Conv2D/ReadVariableOp? conv2d_68/BiasAdd/ReadVariableOp?conv2d_68/Conv2D/ReadVariableOp? conv2d_69/BiasAdd/ReadVariableOp?conv2d_69/Conv2D/ReadVariableOp?*conv2d_transpose_27/BiasAdd/ReadVariableOp?3conv2d_transpose_27/conv2d_transpose/ReadVariableOp?*conv2d_transpose_28/BiasAdd/ReadVariableOp?3conv2d_transpose_28/conv2d_transpose/ReadVariableOp?*conv2d_transpose_29/BiasAdd/ReadVariableOp?3conv2d_transpose_29/conv2d_transpose/ReadVariableOp?decoded/BiasAdd/ReadVariableOp?'decoded/conv2d_transpose/ReadVariableOp?
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_63/Conv2DConv2Dinputs'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
%batch_normalization_63/ReadVariableOpReadVariableOp.batch_normalization_63_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_63/ReadVariableOp_1ReadVariableOp0batch_normalization_63_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_63/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_63_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_63/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_63_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_63/FusedBatchNormV3FusedBatchNormV3conv2d_63/BiasAdd:output:0-batch_normalization_63/ReadVariableOp:value:0/batch_normalization_63/ReadVariableOp_1:value:0>batch_normalization_63/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_63/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
activation_81/LeakyRelu	LeakyRelu+batch_normalization_63/FusedBatchNormV3:y:0*1
_output_shapes
:????????????
conv2d_64/Conv2D/ReadVariableOpReadVariableOp(conv2d_64_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_64/Conv2DConv2D%activation_81/LeakyRelu:activations:0'conv2d_64/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
 conv2d_64/BiasAdd/ReadVariableOpReadVariableOp)conv2d_64_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_64/BiasAddBiasAddconv2d_64/Conv2D:output:0(conv2d_64/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
%batch_normalization_64/ReadVariableOpReadVariableOp.batch_normalization_64_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_64/ReadVariableOp_1ReadVariableOp0batch_normalization_64_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_64/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_64_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_64_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_64/FusedBatchNormV3FusedBatchNormV3conv2d_64/BiasAdd:output:0-batch_normalization_64/ReadVariableOp:value:0/batch_normalization_64/ReadVariableOp_1:value:0>batch_normalization_64/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
activation_82/LeakyRelu	LeakyRelu+batch_normalization_64/FusedBatchNormV3:y:0*1
_output_shapes
:????????????
conv2d_65/Conv2D/ReadVariableOpReadVariableOp(conv2d_65_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_65/Conv2DConv2D%activation_82/LeakyRelu:activations:0'conv2d_65/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
 conv2d_65/BiasAdd/ReadVariableOpReadVariableOp)conv2d_65_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_65/BiasAddBiasAddconv2d_65/Conv2D:output:0(conv2d_65/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
%batch_normalization_65/ReadVariableOpReadVariableOp.batch_normalization_65_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_65/ReadVariableOp_1ReadVariableOp0batch_normalization_65_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_65/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_65_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_65_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_65/FusedBatchNormV3FusedBatchNormV3conv2d_65/BiasAdd:output:0-batch_normalization_65/ReadVariableOp:value:0/batch_normalization_65/ReadVariableOp_1:value:0>batch_normalization_65/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
activation_83/LeakyRelu	LeakyRelu+batch_normalization_65/FusedBatchNormV3:y:0*1
_output_shapes
:????????????
conv2d_66/Conv2D/ReadVariableOpReadVariableOp(conv2d_66_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_66/Conv2DConv2D%activation_83/LeakyRelu:activations:0'conv2d_66/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
 conv2d_66/BiasAdd/ReadVariableOpReadVariableOp)conv2d_66_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_66/BiasAddBiasAddconv2d_66/Conv2D:output:0(conv2d_66/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
%batch_normalization_66/ReadVariableOpReadVariableOp.batch_normalization_66_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_66/ReadVariableOp_1ReadVariableOp0batch_normalization_66_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_66/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_66_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_66_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_66/FusedBatchNormV3FusedBatchNormV3conv2d_66/BiasAdd:output:0-batch_normalization_66/ReadVariableOp:value:0/batch_normalization_66/ReadVariableOp_1:value:0>batch_normalization_66/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( ?
activation_84/LeakyRelu	LeakyRelu+batch_normalization_66/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@ ?
conv2d_67/Conv2D/ReadVariableOpReadVariableOp(conv2d_67_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_67/Conv2DConv2D%activation_84/LeakyRelu:activations:0'conv2d_67/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
 conv2d_67/BiasAdd/ReadVariableOpReadVariableOp)conv2d_67_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_67/BiasAddBiasAddconv2d_67/Conv2D:output:0(conv2d_67/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
%batch_normalization_67/ReadVariableOpReadVariableOp.batch_normalization_67_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_67/ReadVariableOp_1ReadVariableOp0batch_normalization_67_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_67/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_67_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_67_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_67/FusedBatchNormV3FusedBatchNormV3conv2d_67/BiasAdd:output:0-batch_normalization_67/ReadVariableOp:value:0/batch_normalization_67/ReadVariableOp_1:value:0>batch_normalization_67/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( ?
activation_85/LeakyRelu	LeakyRelu+batch_normalization_67/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@ ?
conv2d_68/Conv2D/ReadVariableOpReadVariableOp(conv2d_68_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_68/Conv2DConv2D%activation_85/LeakyRelu:activations:0'conv2d_68/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
 conv2d_68/BiasAdd/ReadVariableOpReadVariableOp)conv2d_68_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_68/BiasAddBiasAddconv2d_68/Conv2D:output:0(conv2d_68/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
%batch_normalization_68/ReadVariableOpReadVariableOp.batch_normalization_68_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_68/ReadVariableOp_1ReadVariableOp0batch_normalization_68_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_68/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_68_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_68_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_68/FusedBatchNormV3FusedBatchNormV3conv2d_68/BiasAdd:output:0-batch_normalization_68/ReadVariableOp:value:0/batch_normalization_68/ReadVariableOp_1:value:0>batch_normalization_68/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( ?
activation_86/LeakyRelu	LeakyRelu+batch_normalization_68/FusedBatchNormV3:y:0*/
_output_shapes
:?????????  @?
conv2d_69/Conv2D/ReadVariableOpReadVariableOp(conv2d_69_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
conv2d_69/Conv2DConv2D%activation_86/LeakyRelu:activations:0'conv2d_69/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_69/BiasAdd/ReadVariableOpReadVariableOp)conv2d_69_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_69/BiasAddBiasAddconv2d_69/Conv2D:output:0(conv2d_69/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
%batch_normalization_69/ReadVariableOpReadVariableOp.batch_normalization_69_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_69/ReadVariableOp_1ReadVariableOp0batch_normalization_69_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_69/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_69_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_69/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_69_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_69/FusedBatchNormV3FusedBatchNormV3conv2d_69/BiasAdd:output:0-batch_normalization_69/ReadVariableOp:value:0/batch_normalization_69/ReadVariableOp_1:value:0>batch_normalization_69/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_69/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
encoded/CastCast+batch_normalization_69/FusedBatchNormV3:y:0*

DstT0*

SrcT0*/
_output_shapes
:????????? j
encoded/LeakyRelu	LeakyReluencoded/Cast:y:0*
T0*/
_output_shapes
:????????? ?
conv2d_transpose_27/CastCastencoded/LeakyRelu:activations:0*

DstT0*

SrcT0*/
_output_shapes
:????????? e
conv2d_transpose_27/ShapeShapeconv2d_transpose_27/Cast:y:0*
T0*
_output_shapes
:q
'conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_27/strided_sliceStridedSlice"conv2d_transpose_27/Shape:output:00conv2d_transpose_27/strided_slice/stack:output:02conv2d_transpose_27/strided_slice/stack_1:output:02conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_27/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_27/stack/2Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose_27/stackPack*conv2d_transpose_27/strided_slice:output:0$conv2d_transpose_27/stack/1:output:0$conv2d_transpose_27/stack/2:output:0$conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_27/strided_slice_1StridedSlice"conv2d_transpose_27/stack:output:02conv2d_transpose_27/strided_slice_1/stack:output:04conv2d_transpose_27/strided_slice_1/stack_1:output:04conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_27_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
$conv2d_transpose_27/conv2d_transposeConv2DBackpropInput"conv2d_transpose_27/stack:output:0;conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0conv2d_transpose_27/Cast:y:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
*conv2d_transpose_27/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_transpose_27/BiasAddBiasAdd-conv2d_transpose_27/conv2d_transpose:output:02conv2d_transpose_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @{
activation_87/LeakyRelu	LeakyRelu$conv2d_transpose_27/BiasAdd:output:0*/
_output_shapes
:?????????  @n
conv2d_transpose_28/ShapeShape%activation_87/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_28/strided_sliceStridedSlice"conv2d_transpose_28/Shape:output:00conv2d_transpose_28/strided_slice/stack:output:02conv2d_transpose_28/strided_slice/stack_1:output:02conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_28/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@]
conv2d_transpose_28/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@]
conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_28/stackPack*conv2d_transpose_28/strided_slice:output:0$conv2d_transpose_28/stack/1:output:0$conv2d_transpose_28/stack/2:output:0$conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_28/strided_slice_1StridedSlice"conv2d_transpose_28/stack:output:02conv2d_transpose_28/strided_slice_1/stack:output:04conv2d_transpose_28/strided_slice_1/stack_1:output:04conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_28_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
$conv2d_transpose_28/conv2d_transposeConv2DBackpropInput"conv2d_transpose_28/stack:output:0;conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:0%activation_87/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
*conv2d_transpose_28/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_28/BiasAddBiasAdd-conv2d_transpose_28/conv2d_transpose:output:02conv2d_transpose_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ {
activation_88/LeakyRelu	LeakyRelu$conv2d_transpose_28/BiasAdd:output:0*/
_output_shapes
:?????????@@ n
conv2d_transpose_29/ShapeShape%activation_88/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_29/strided_sliceStridedSlice"conv2d_transpose_29/Shape:output:00conv2d_transpose_29/strided_slice/stack:output:02conv2d_transpose_29/strided_slice/stack_1:output:02conv2d_transpose_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_29/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?^
conv2d_transpose_29/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?]
conv2d_transpose_29/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_29/stackPack*conv2d_transpose_29/strided_slice:output:0$conv2d_transpose_29/stack/1:output:0$conv2d_transpose_29/stack/2:output:0$conv2d_transpose_29/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_29/strided_slice_1StridedSlice"conv2d_transpose_29/stack:output:02conv2d_transpose_29/strided_slice_1/stack:output:04conv2d_transpose_29/strided_slice_1/stack_1:output:04conv2d_transpose_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_29/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_29_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
$conv2d_transpose_29/conv2d_transposeConv2DBackpropInput"conv2d_transpose_29/stack:output:0;conv2d_transpose_29/conv2d_transpose/ReadVariableOp:value:0%activation_88/LeakyRelu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
*conv2d_transpose_29/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_29/BiasAddBiasAdd-conv2d_transpose_29/conv2d_transpose:output:02conv2d_transpose_29/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????}
activation_89/LeakyRelu	LeakyRelu$conv2d_transpose_29/BiasAdd:output:0*1
_output_shapes
:???????????b
decoded/ShapeShape%activation_89/LeakyRelu:activations:0*
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
decoded/conv2d_transposeConv2DBackpropInputdecoded/stack:output:0/decoded/conv2d_transpose/ReadVariableOp:value:0%activation_89/LeakyRelu:activations:0*
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
NoOpNoOp7^batch_normalization_63/FusedBatchNormV3/ReadVariableOp9^batch_normalization_63/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_63/ReadVariableOp(^batch_normalization_63/ReadVariableOp_17^batch_normalization_64/FusedBatchNormV3/ReadVariableOp9^batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_64/ReadVariableOp(^batch_normalization_64/ReadVariableOp_17^batch_normalization_65/FusedBatchNormV3/ReadVariableOp9^batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_65/ReadVariableOp(^batch_normalization_65/ReadVariableOp_17^batch_normalization_66/FusedBatchNormV3/ReadVariableOp9^batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_66/ReadVariableOp(^batch_normalization_66/ReadVariableOp_17^batch_normalization_67/FusedBatchNormV3/ReadVariableOp9^batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_67/ReadVariableOp(^batch_normalization_67/ReadVariableOp_17^batch_normalization_68/FusedBatchNormV3/ReadVariableOp9^batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_68/ReadVariableOp(^batch_normalization_68/ReadVariableOp_17^batch_normalization_69/FusedBatchNormV3/ReadVariableOp9^batch_normalization_69/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_69/ReadVariableOp(^batch_normalization_69/ReadVariableOp_1!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp!^conv2d_64/BiasAdd/ReadVariableOp ^conv2d_64/Conv2D/ReadVariableOp!^conv2d_65/BiasAdd/ReadVariableOp ^conv2d_65/Conv2D/ReadVariableOp!^conv2d_66/BiasAdd/ReadVariableOp ^conv2d_66/Conv2D/ReadVariableOp!^conv2d_67/BiasAdd/ReadVariableOp ^conv2d_67/Conv2D/ReadVariableOp!^conv2d_68/BiasAdd/ReadVariableOp ^conv2d_68/Conv2D/ReadVariableOp!^conv2d_69/BiasAdd/ReadVariableOp ^conv2d_69/Conv2D/ReadVariableOp+^conv2d_transpose_27/BiasAdd/ReadVariableOp4^conv2d_transpose_27/conv2d_transpose/ReadVariableOp+^conv2d_transpose_28/BiasAdd/ReadVariableOp4^conv2d_transpose_28/conv2d_transpose/ReadVariableOp+^conv2d_transpose_29/BiasAdd/ReadVariableOp4^conv2d_transpose_29/conv2d_transpose/ReadVariableOp^decoded/BiasAdd/ReadVariableOp(^decoded/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_63/FusedBatchNormV3/ReadVariableOp6batch_normalization_63/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_63/FusedBatchNormV3/ReadVariableOp_18batch_normalization_63/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_63/ReadVariableOp%batch_normalization_63/ReadVariableOp2R
'batch_normalization_63/ReadVariableOp_1'batch_normalization_63/ReadVariableOp_12p
6batch_normalization_64/FusedBatchNormV3/ReadVariableOp6batch_normalization_64/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_64/FusedBatchNormV3/ReadVariableOp_18batch_normalization_64/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_64/ReadVariableOp%batch_normalization_64/ReadVariableOp2R
'batch_normalization_64/ReadVariableOp_1'batch_normalization_64/ReadVariableOp_12p
6batch_normalization_65/FusedBatchNormV3/ReadVariableOp6batch_normalization_65/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_65/FusedBatchNormV3/ReadVariableOp_18batch_normalization_65/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_65/ReadVariableOp%batch_normalization_65/ReadVariableOp2R
'batch_normalization_65/ReadVariableOp_1'batch_normalization_65/ReadVariableOp_12p
6batch_normalization_66/FusedBatchNormV3/ReadVariableOp6batch_normalization_66/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_18batch_normalization_66/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_66/ReadVariableOp%batch_normalization_66/ReadVariableOp2R
'batch_normalization_66/ReadVariableOp_1'batch_normalization_66/ReadVariableOp_12p
6batch_normalization_67/FusedBatchNormV3/ReadVariableOp6batch_normalization_67/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_67/FusedBatchNormV3/ReadVariableOp_18batch_normalization_67/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_67/ReadVariableOp%batch_normalization_67/ReadVariableOp2R
'batch_normalization_67/ReadVariableOp_1'batch_normalization_67/ReadVariableOp_12p
6batch_normalization_68/FusedBatchNormV3/ReadVariableOp6batch_normalization_68/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_68/FusedBatchNormV3/ReadVariableOp_18batch_normalization_68/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_68/ReadVariableOp%batch_normalization_68/ReadVariableOp2R
'batch_normalization_68/ReadVariableOp_1'batch_normalization_68/ReadVariableOp_12p
6batch_normalization_69/FusedBatchNormV3/ReadVariableOp6batch_normalization_69/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_69/FusedBatchNormV3/ReadVariableOp_18batch_normalization_69/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_69/ReadVariableOp%batch_normalization_69/ReadVariableOp2R
'batch_normalization_69/ReadVariableOp_1'batch_normalization_69/ReadVariableOp_12D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp2D
 conv2d_64/BiasAdd/ReadVariableOp conv2d_64/BiasAdd/ReadVariableOp2B
conv2d_64/Conv2D/ReadVariableOpconv2d_64/Conv2D/ReadVariableOp2D
 conv2d_65/BiasAdd/ReadVariableOp conv2d_65/BiasAdd/ReadVariableOp2B
conv2d_65/Conv2D/ReadVariableOpconv2d_65/Conv2D/ReadVariableOp2D
 conv2d_66/BiasAdd/ReadVariableOp conv2d_66/BiasAdd/ReadVariableOp2B
conv2d_66/Conv2D/ReadVariableOpconv2d_66/Conv2D/ReadVariableOp2D
 conv2d_67/BiasAdd/ReadVariableOp conv2d_67/BiasAdd/ReadVariableOp2B
conv2d_67/Conv2D/ReadVariableOpconv2d_67/Conv2D/ReadVariableOp2D
 conv2d_68/BiasAdd/ReadVariableOp conv2d_68/BiasAdd/ReadVariableOp2B
conv2d_68/Conv2D/ReadVariableOpconv2d_68/Conv2D/ReadVariableOp2D
 conv2d_69/BiasAdd/ReadVariableOp conv2d_69/BiasAdd/ReadVariableOp2B
conv2d_69/Conv2D/ReadVariableOpconv2d_69/Conv2D/ReadVariableOp2X
*conv2d_transpose_27/BiasAdd/ReadVariableOp*conv2d_transpose_27/BiasAdd/ReadVariableOp2j
3conv2d_transpose_27/conv2d_transpose/ReadVariableOp3conv2d_transpose_27/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_28/BiasAdd/ReadVariableOp*conv2d_transpose_28/BiasAdd/ReadVariableOp2j
3conv2d_transpose_28/conv2d_transpose/ReadVariableOp3conv2d_transpose_28/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_29/BiasAdd/ReadVariableOp*conv2d_transpose_29/BiasAdd/ReadVariableOp2j
3conv2d_transpose_29/conv2d_transpose/ReadVariableOp3conv2d_transpose_29/conv2d_transpose/ReadVariableOp2@
decoded/BiasAdd/ReadVariableOpdecoded/BiasAdd/ReadVariableOp2R
'decoded/conv2d_transpose/ReadVariableOp'decoded/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_69_layer_call_fn_3374527

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
F__inference_conv2d_69_layer_call_and_return_conditional_losses_3372167w
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
?
?
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3371611

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
8__inference_batch_normalization_65_layer_call_fn_3374199

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
S__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3371514?
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
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3374326

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
?
?
%__inference_signature_wrapper_3373296
conv2d_63_input!
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_63_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
"__inference__wrapped_model_3371333y
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
_user_specified_nameconv2d_63_input
?	
?
8__inference_batch_normalization_64_layer_call_fn_3374095

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
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3371419?
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
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_3371675

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
D__inference_decoded_layer_call_and_return_conditional_losses_3371951

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
?
K
/__inference_activation_85_layer_call_fn_3374422

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
J__inference_activation_85_layer_call_and_return_conditional_losses_3372123h
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
?

?
F__inference_conv2d_68_layer_call_and_return_conditional_losses_3372135

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
?	
?
8__inference_batch_normalization_63_layer_call_fn_3374004

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
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3371355?
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
8__inference_batch_normalization_69_layer_call_fn_3374550

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
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_3371739?
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
8__inference_batch_normalization_68_layer_call_fn_3374459

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
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_3371675?
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
?
?
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3371355

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
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_3371739

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
F__inference_conv2d_63_layer_call_and_return_conditional_losses_3373991

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
?
?
.__inference_sequential_9_layer_call_fn_3372915
conv2d_63_input!
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_63_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_3372707y
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
_user_specified_nameconv2d_63_input
?
?
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3374417

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
J__inference_activation_89_layer_call_and_return_conditional_losses_3372225

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
8__inference_batch_normalization_67_layer_call_fn_3374381

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
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3371642?
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
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3371386

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
J__inference_activation_84_layer_call_and_return_conditional_losses_3374336

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
?
f
J__inference_activation_87_layer_call_and_return_conditional_losses_3374661

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
?
?
+__inference_conv2d_66_layer_call_fn_3374254

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
F__inference_conv2d_66_layer_call_and_return_conditional_losses_3372071w
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
?
?
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_3371706

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
?
?
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3374144

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
??
?
I__inference_sequential_9_layer_call_and_return_conditional_losses_3373183
conv2d_63_input+
conv2d_63_3373052:
conv2d_63_3373054:,
batch_normalization_63_3373057:,
batch_normalization_63_3373059:,
batch_normalization_63_3373061:,
batch_normalization_63_3373063:+
conv2d_64_3373067:
conv2d_64_3373069:,
batch_normalization_64_3373072:,
batch_normalization_64_3373074:,
batch_normalization_64_3373076:,
batch_normalization_64_3373078:+
conv2d_65_3373082:
conv2d_65_3373084:,
batch_normalization_65_3373087:,
batch_normalization_65_3373089:,
batch_normalization_65_3373091:,
batch_normalization_65_3373093:+
conv2d_66_3373097: 
conv2d_66_3373099: ,
batch_normalization_66_3373102: ,
batch_normalization_66_3373104: ,
batch_normalization_66_3373106: ,
batch_normalization_66_3373108: +
conv2d_67_3373112:  
conv2d_67_3373114: ,
batch_normalization_67_3373117: ,
batch_normalization_67_3373119: ,
batch_normalization_67_3373121: ,
batch_normalization_67_3373123: +
conv2d_68_3373127: @
conv2d_68_3373129:@,
batch_normalization_68_3373132:@,
batch_normalization_68_3373134:@,
batch_normalization_68_3373136:@,
batch_normalization_68_3373138:@+
conv2d_69_3373142:@ 
conv2d_69_3373144: ,
batch_normalization_69_3373147: ,
batch_normalization_69_3373149: ,
batch_normalization_69_3373151: ,
batch_normalization_69_3373153: 5
conv2d_transpose_27_3373159:@ )
conv2d_transpose_27_3373161:@5
conv2d_transpose_28_3373165: @)
conv2d_transpose_28_3373167: 5
conv2d_transpose_29_3373171: )
conv2d_transpose_29_3373173:)
decoded_3373177:
decoded_3373179:
identity??.batch_normalization_63/StatefulPartitionedCall?.batch_normalization_64/StatefulPartitionedCall?.batch_normalization_65/StatefulPartitionedCall?.batch_normalization_66/StatefulPartitionedCall?.batch_normalization_67/StatefulPartitionedCall?.batch_normalization_68/StatefulPartitionedCall?.batch_normalization_69/StatefulPartitionedCall?!conv2d_63/StatefulPartitionedCall?!conv2d_64/StatefulPartitionedCall?!conv2d_65/StatefulPartitionedCall?!conv2d_66/StatefulPartitionedCall?!conv2d_67/StatefulPartitionedCall?!conv2d_68/StatefulPartitionedCall?!conv2d_69/StatefulPartitionedCall?+conv2d_transpose_27/StatefulPartitionedCall?+conv2d_transpose_28/StatefulPartitionedCall?+conv2d_transpose_29/StatefulPartitionedCall?decoded/StatefulPartitionedCall?
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCallconv2d_63_inputconv2d_63_3373052conv2d_63_3373054*
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
F__inference_conv2d_63_layer_call_and_return_conditional_losses_3371975?
.batch_normalization_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0batch_normalization_63_3373057batch_normalization_63_3373059batch_normalization_63_3373061batch_normalization_63_3373063*
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
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3371386?
activation_81/PartitionedCallPartitionedCall7batch_normalization_63/StatefulPartitionedCall:output:0*
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
J__inference_activation_81_layer_call_and_return_conditional_losses_3371995?
!conv2d_64/StatefulPartitionedCallStatefulPartitionedCall&activation_81/PartitionedCall:output:0conv2d_64_3373067conv2d_64_3373069*
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
F__inference_conv2d_64_layer_call_and_return_conditional_losses_3372007?
.batch_normalization_64/StatefulPartitionedCallStatefulPartitionedCall*conv2d_64/StatefulPartitionedCall:output:0batch_normalization_64_3373072batch_normalization_64_3373074batch_normalization_64_3373076batch_normalization_64_3373078*
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
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3371450?
activation_82/PartitionedCallPartitionedCall7batch_normalization_64/StatefulPartitionedCall:output:0*
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
J__inference_activation_82_layer_call_and_return_conditional_losses_3372027?
!conv2d_65/StatefulPartitionedCallStatefulPartitionedCall&activation_82/PartitionedCall:output:0conv2d_65_3373082conv2d_65_3373084*
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
F__inference_conv2d_65_layer_call_and_return_conditional_losses_3372039?
.batch_normalization_65/StatefulPartitionedCallStatefulPartitionedCall*conv2d_65/StatefulPartitionedCall:output:0batch_normalization_65_3373087batch_normalization_65_3373089batch_normalization_65_3373091batch_normalization_65_3373093*
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
S__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3371514?
activation_83/PartitionedCallPartitionedCall7batch_normalization_65/StatefulPartitionedCall:output:0*
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
J__inference_activation_83_layer_call_and_return_conditional_losses_3372059?
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCall&activation_83/PartitionedCall:output:0conv2d_66_3373097conv2d_66_3373099*
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
F__inference_conv2d_66_layer_call_and_return_conditional_losses_3372071?
.batch_normalization_66/StatefulPartitionedCallStatefulPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0batch_normalization_66_3373102batch_normalization_66_3373104batch_normalization_66_3373106batch_normalization_66_3373108*
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
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3371578?
activation_84/PartitionedCallPartitionedCall7batch_normalization_66/StatefulPartitionedCall:output:0*
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
J__inference_activation_84_layer_call_and_return_conditional_losses_3372091?
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall&activation_84/PartitionedCall:output:0conv2d_67_3373112conv2d_67_3373114*
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
F__inference_conv2d_67_layer_call_and_return_conditional_losses_3372103?
.batch_normalization_67/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0batch_normalization_67_3373117batch_normalization_67_3373119batch_normalization_67_3373121batch_normalization_67_3373123*
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
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3371642?
activation_85/PartitionedCallPartitionedCall7batch_normalization_67/StatefulPartitionedCall:output:0*
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
J__inference_activation_85_layer_call_and_return_conditional_losses_3372123?
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall&activation_85/PartitionedCall:output:0conv2d_68_3373127conv2d_68_3373129*
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
F__inference_conv2d_68_layer_call_and_return_conditional_losses_3372135?
.batch_normalization_68/StatefulPartitionedCallStatefulPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0batch_normalization_68_3373132batch_normalization_68_3373134batch_normalization_68_3373136batch_normalization_68_3373138*
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
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_3371706?
activation_86/PartitionedCallPartitionedCall7batch_normalization_68/StatefulPartitionedCall:output:0*
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
J__inference_activation_86_layer_call_and_return_conditional_losses_3372155?
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCall&activation_86/PartitionedCall:output:0conv2d_69_3373142conv2d_69_3373144*
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
F__inference_conv2d_69_layer_call_and_return_conditional_losses_3372167?
.batch_normalization_69/StatefulPartitionedCallStatefulPartitionedCall*conv2d_69/StatefulPartitionedCall:output:0batch_normalization_69_3373147batch_normalization_69_3373149batch_normalization_69_3373151batch_normalization_69_3373153*
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
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_3371770?
encoded/CastCast7batch_normalization_69/StatefulPartitionedCall:output:0*

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
D__inference_encoded_layer_call_and_return_conditional_losses_3372188?
conv2d_transpose_27/CastCast encoded/PartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_27/Cast:y:0conv2d_transpose_27_3373159conv2d_transpose_27_3373161*
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
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_3371818?
activation_87/PartitionedCallPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0*
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
J__inference_activation_87_layer_call_and_return_conditional_losses_3372201?
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall&activation_87/PartitionedCall:output:0conv2d_transpose_28_3373165conv2d_transpose_28_3373167*
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
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_3371862?
activation_88/PartitionedCallPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0*
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
J__inference_activation_88_layer_call_and_return_conditional_losses_3372213?
+conv2d_transpose_29/StatefulPartitionedCallStatefulPartitionedCall&activation_88/PartitionedCall:output:0conv2d_transpose_29_3373171conv2d_transpose_29_3373173*
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
P__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_3371906?
activation_89/PartitionedCallPartitionedCall4conv2d_transpose_29/StatefulPartitionedCall:output:0*
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
J__inference_activation_89_layer_call_and_return_conditional_losses_3372225?
decoded/StatefulPartitionedCallStatefulPartitionedCall&activation_89/PartitionedCall:output:0decoded_3373177decoded_3373179*
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
D__inference_decoded_layer_call_and_return_conditional_losses_3371951?
IdentityIdentity(decoded/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp/^batch_normalization_63/StatefulPartitionedCall/^batch_normalization_64/StatefulPartitionedCall/^batch_normalization_65/StatefulPartitionedCall/^batch_normalization_66/StatefulPartitionedCall/^batch_normalization_67/StatefulPartitionedCall/^batch_normalization_68/StatefulPartitionedCall/^batch_normalization_69/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall"^conv2d_64/StatefulPartitionedCall"^conv2d_65/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall,^conv2d_transpose_29/StatefulPartitionedCall ^decoded/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_63/StatefulPartitionedCall.batch_normalization_63/StatefulPartitionedCall2`
.batch_normalization_64/StatefulPartitionedCall.batch_normalization_64/StatefulPartitionedCall2`
.batch_normalization_65/StatefulPartitionedCall.batch_normalization_65/StatefulPartitionedCall2`
.batch_normalization_66/StatefulPartitionedCall.batch_normalization_66/StatefulPartitionedCall2`
.batch_normalization_67/StatefulPartitionedCall.batch_normalization_67/StatefulPartitionedCall2`
.batch_normalization_68/StatefulPartitionedCall.batch_normalization_68/StatefulPartitionedCall2`
.batch_normalization_69/StatefulPartitionedCall.batch_normalization_69/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2F
!conv2d_64/StatefulPartitionedCall!conv2d_64/StatefulPartitionedCall2F
!conv2d_65/StatefulPartitionedCall!conv2d_65/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall2Z
+conv2d_transpose_29/StatefulPartitionedCall+conv2d_transpose_29/StatefulPartitionedCall2B
decoded/StatefulPartitionedCalldecoded/StatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_63_input
?	
?
8__inference_batch_normalization_66_layer_call_fn_3374290

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
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3371578?
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
?
f
J__inference_activation_83_layer_call_and_return_conditional_losses_3372059

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
??
?1
I__inference_sequential_9_layer_call_and_return_conditional_losses_3373972

inputsB
(conv2d_63_conv2d_readvariableop_resource:7
)conv2d_63_biasadd_readvariableop_resource:<
.batch_normalization_63_readvariableop_resource:>
0batch_normalization_63_readvariableop_1_resource:M
?batch_normalization_63_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_63_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_64_conv2d_readvariableop_resource:7
)conv2d_64_biasadd_readvariableop_resource:<
.batch_normalization_64_readvariableop_resource:>
0batch_normalization_64_readvariableop_1_resource:M
?batch_normalization_64_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_64_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_65_conv2d_readvariableop_resource:7
)conv2d_65_biasadd_readvariableop_resource:<
.batch_normalization_65_readvariableop_resource:>
0batch_normalization_65_readvariableop_1_resource:M
?batch_normalization_65_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_65_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_66_conv2d_readvariableop_resource: 7
)conv2d_66_biasadd_readvariableop_resource: <
.batch_normalization_66_readvariableop_resource: >
0batch_normalization_66_readvariableop_1_resource: M
?batch_normalization_66_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_66_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_67_conv2d_readvariableop_resource:  7
)conv2d_67_biasadd_readvariableop_resource: <
.batch_normalization_67_readvariableop_resource: >
0batch_normalization_67_readvariableop_1_resource: M
?batch_normalization_67_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_67_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_68_conv2d_readvariableop_resource: @7
)conv2d_68_biasadd_readvariableop_resource:@<
.batch_normalization_68_readvariableop_resource:@>
0batch_normalization_68_readvariableop_1_resource:@M
?batch_normalization_68_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_68_fusedbatchnormv3_readvariableop_1_resource:@B
(conv2d_69_conv2d_readvariableop_resource:@ 7
)conv2d_69_biasadd_readvariableop_resource: <
.batch_normalization_69_readvariableop_resource: >
0batch_normalization_69_readvariableop_1_resource: M
?batch_normalization_69_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_69_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_27_conv2d_transpose_readvariableop_resource:@ A
3conv2d_transpose_27_biasadd_readvariableop_resource:@V
<conv2d_transpose_28_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_28_biasadd_readvariableop_resource: V
<conv2d_transpose_29_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_29_biasadd_readvariableop_resource:J
0decoded_conv2d_transpose_readvariableop_resource:5
'decoded_biasadd_readvariableop_resource:
identity??%batch_normalization_63/AssignNewValue?'batch_normalization_63/AssignNewValue_1?6batch_normalization_63/FusedBatchNormV3/ReadVariableOp?8batch_normalization_63/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_63/ReadVariableOp?'batch_normalization_63/ReadVariableOp_1?%batch_normalization_64/AssignNewValue?'batch_normalization_64/AssignNewValue_1?6batch_normalization_64/FusedBatchNormV3/ReadVariableOp?8batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_64/ReadVariableOp?'batch_normalization_64/ReadVariableOp_1?%batch_normalization_65/AssignNewValue?'batch_normalization_65/AssignNewValue_1?6batch_normalization_65/FusedBatchNormV3/ReadVariableOp?8batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_65/ReadVariableOp?'batch_normalization_65/ReadVariableOp_1?%batch_normalization_66/AssignNewValue?'batch_normalization_66/AssignNewValue_1?6batch_normalization_66/FusedBatchNormV3/ReadVariableOp?8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_66/ReadVariableOp?'batch_normalization_66/ReadVariableOp_1?%batch_normalization_67/AssignNewValue?'batch_normalization_67/AssignNewValue_1?6batch_normalization_67/FusedBatchNormV3/ReadVariableOp?8batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_67/ReadVariableOp?'batch_normalization_67/ReadVariableOp_1?%batch_normalization_68/AssignNewValue?'batch_normalization_68/AssignNewValue_1?6batch_normalization_68/FusedBatchNormV3/ReadVariableOp?8batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_68/ReadVariableOp?'batch_normalization_68/ReadVariableOp_1?%batch_normalization_69/AssignNewValue?'batch_normalization_69/AssignNewValue_1?6batch_normalization_69/FusedBatchNormV3/ReadVariableOp?8batch_normalization_69/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_69/ReadVariableOp?'batch_normalization_69/ReadVariableOp_1? conv2d_63/BiasAdd/ReadVariableOp?conv2d_63/Conv2D/ReadVariableOp? conv2d_64/BiasAdd/ReadVariableOp?conv2d_64/Conv2D/ReadVariableOp? conv2d_65/BiasAdd/ReadVariableOp?conv2d_65/Conv2D/ReadVariableOp? conv2d_66/BiasAdd/ReadVariableOp?conv2d_66/Conv2D/ReadVariableOp? conv2d_67/BiasAdd/ReadVariableOp?conv2d_67/Conv2D/ReadVariableOp? conv2d_68/BiasAdd/ReadVariableOp?conv2d_68/Conv2D/ReadVariableOp? conv2d_69/BiasAdd/ReadVariableOp?conv2d_69/Conv2D/ReadVariableOp?*conv2d_transpose_27/BiasAdd/ReadVariableOp?3conv2d_transpose_27/conv2d_transpose/ReadVariableOp?*conv2d_transpose_28/BiasAdd/ReadVariableOp?3conv2d_transpose_28/conv2d_transpose/ReadVariableOp?*conv2d_transpose_29/BiasAdd/ReadVariableOp?3conv2d_transpose_29/conv2d_transpose/ReadVariableOp?decoded/BiasAdd/ReadVariableOp?'decoded/conv2d_transpose/ReadVariableOp?
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_63/Conv2DConv2Dinputs'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
%batch_normalization_63/ReadVariableOpReadVariableOp.batch_normalization_63_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_63/ReadVariableOp_1ReadVariableOp0batch_normalization_63_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_63/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_63_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_63/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_63_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_63/FusedBatchNormV3FusedBatchNormV3conv2d_63/BiasAdd:output:0-batch_normalization_63/ReadVariableOp:value:0/batch_normalization_63/ReadVariableOp_1:value:0>batch_normalization_63/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_63/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_63/AssignNewValueAssignVariableOp?batch_normalization_63_fusedbatchnormv3_readvariableop_resource4batch_normalization_63/FusedBatchNormV3:batch_mean:07^batch_normalization_63/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_63/AssignNewValue_1AssignVariableOpAbatch_normalization_63_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_63/FusedBatchNormV3:batch_variance:09^batch_normalization_63/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_81/LeakyRelu	LeakyRelu+batch_normalization_63/FusedBatchNormV3:y:0*1
_output_shapes
:????????????
conv2d_64/Conv2D/ReadVariableOpReadVariableOp(conv2d_64_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_64/Conv2DConv2D%activation_81/LeakyRelu:activations:0'conv2d_64/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
 conv2d_64/BiasAdd/ReadVariableOpReadVariableOp)conv2d_64_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_64/BiasAddBiasAddconv2d_64/Conv2D:output:0(conv2d_64/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
%batch_normalization_64/ReadVariableOpReadVariableOp.batch_normalization_64_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_64/ReadVariableOp_1ReadVariableOp0batch_normalization_64_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_64/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_64_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_64_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_64/FusedBatchNormV3FusedBatchNormV3conv2d_64/BiasAdd:output:0-batch_normalization_64/ReadVariableOp:value:0/batch_normalization_64/ReadVariableOp_1:value:0>batch_normalization_64/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_64/AssignNewValueAssignVariableOp?batch_normalization_64_fusedbatchnormv3_readvariableop_resource4batch_normalization_64/FusedBatchNormV3:batch_mean:07^batch_normalization_64/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_64/AssignNewValue_1AssignVariableOpAbatch_normalization_64_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_64/FusedBatchNormV3:batch_variance:09^batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_82/LeakyRelu	LeakyRelu+batch_normalization_64/FusedBatchNormV3:y:0*1
_output_shapes
:????????????
conv2d_65/Conv2D/ReadVariableOpReadVariableOp(conv2d_65_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_65/Conv2DConv2D%activation_82/LeakyRelu:activations:0'conv2d_65/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
 conv2d_65/BiasAdd/ReadVariableOpReadVariableOp)conv2d_65_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_65/BiasAddBiasAddconv2d_65/Conv2D:output:0(conv2d_65/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
%batch_normalization_65/ReadVariableOpReadVariableOp.batch_normalization_65_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_65/ReadVariableOp_1ReadVariableOp0batch_normalization_65_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_65/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_65_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_65_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_65/FusedBatchNormV3FusedBatchNormV3conv2d_65/BiasAdd:output:0-batch_normalization_65/ReadVariableOp:value:0/batch_normalization_65/ReadVariableOp_1:value:0>batch_normalization_65/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_65/AssignNewValueAssignVariableOp?batch_normalization_65_fusedbatchnormv3_readvariableop_resource4batch_normalization_65/FusedBatchNormV3:batch_mean:07^batch_normalization_65/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_65/AssignNewValue_1AssignVariableOpAbatch_normalization_65_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_65/FusedBatchNormV3:batch_variance:09^batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_83/LeakyRelu	LeakyRelu+batch_normalization_65/FusedBatchNormV3:y:0*1
_output_shapes
:????????????
conv2d_66/Conv2D/ReadVariableOpReadVariableOp(conv2d_66_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_66/Conv2DConv2D%activation_83/LeakyRelu:activations:0'conv2d_66/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
 conv2d_66/BiasAdd/ReadVariableOpReadVariableOp)conv2d_66_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_66/BiasAddBiasAddconv2d_66/Conv2D:output:0(conv2d_66/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
%batch_normalization_66/ReadVariableOpReadVariableOp.batch_normalization_66_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_66/ReadVariableOp_1ReadVariableOp0batch_normalization_66_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_66/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_66_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_66_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_66/FusedBatchNormV3FusedBatchNormV3conv2d_66/BiasAdd:output:0-batch_normalization_66/ReadVariableOp:value:0/batch_normalization_66/ReadVariableOp_1:value:0>batch_normalization_66/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_66/AssignNewValueAssignVariableOp?batch_normalization_66_fusedbatchnormv3_readvariableop_resource4batch_normalization_66/FusedBatchNormV3:batch_mean:07^batch_normalization_66/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_66/AssignNewValue_1AssignVariableOpAbatch_normalization_66_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_66/FusedBatchNormV3:batch_variance:09^batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_84/LeakyRelu	LeakyRelu+batch_normalization_66/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@ ?
conv2d_67/Conv2D/ReadVariableOpReadVariableOp(conv2d_67_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_67/Conv2DConv2D%activation_84/LeakyRelu:activations:0'conv2d_67/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
 conv2d_67/BiasAdd/ReadVariableOpReadVariableOp)conv2d_67_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_67/BiasAddBiasAddconv2d_67/Conv2D:output:0(conv2d_67/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
%batch_normalization_67/ReadVariableOpReadVariableOp.batch_normalization_67_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_67/ReadVariableOp_1ReadVariableOp0batch_normalization_67_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_67/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_67_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_67_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_67/FusedBatchNormV3FusedBatchNormV3conv2d_67/BiasAdd:output:0-batch_normalization_67/ReadVariableOp:value:0/batch_normalization_67/ReadVariableOp_1:value:0>batch_normalization_67/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_67/AssignNewValueAssignVariableOp?batch_normalization_67_fusedbatchnormv3_readvariableop_resource4batch_normalization_67/FusedBatchNormV3:batch_mean:07^batch_normalization_67/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_67/AssignNewValue_1AssignVariableOpAbatch_normalization_67_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_67/FusedBatchNormV3:batch_variance:09^batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_85/LeakyRelu	LeakyRelu+batch_normalization_67/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@ ?
conv2d_68/Conv2D/ReadVariableOpReadVariableOp(conv2d_68_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_68/Conv2DConv2D%activation_85/LeakyRelu:activations:0'conv2d_68/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
 conv2d_68/BiasAdd/ReadVariableOpReadVariableOp)conv2d_68_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_68/BiasAddBiasAddconv2d_68/Conv2D:output:0(conv2d_68/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
%batch_normalization_68/ReadVariableOpReadVariableOp.batch_normalization_68_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_68/ReadVariableOp_1ReadVariableOp0batch_normalization_68_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_68/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_68_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_68_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_68/FusedBatchNormV3FusedBatchNormV3conv2d_68/BiasAdd:output:0-batch_normalization_68/ReadVariableOp:value:0/batch_normalization_68/ReadVariableOp_1:value:0>batch_normalization_68/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_68/AssignNewValueAssignVariableOp?batch_normalization_68_fusedbatchnormv3_readvariableop_resource4batch_normalization_68/FusedBatchNormV3:batch_mean:07^batch_normalization_68/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_68/AssignNewValue_1AssignVariableOpAbatch_normalization_68_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_68/FusedBatchNormV3:batch_variance:09^batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_86/LeakyRelu	LeakyRelu+batch_normalization_68/FusedBatchNormV3:y:0*/
_output_shapes
:?????????  @?
conv2d_69/Conv2D/ReadVariableOpReadVariableOp(conv2d_69_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
conv2d_69/Conv2DConv2D%activation_86/LeakyRelu:activations:0'conv2d_69/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_69/BiasAdd/ReadVariableOpReadVariableOp)conv2d_69_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_69/BiasAddBiasAddconv2d_69/Conv2D:output:0(conv2d_69/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
%batch_normalization_69/ReadVariableOpReadVariableOp.batch_normalization_69_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_69/ReadVariableOp_1ReadVariableOp0batch_normalization_69_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_69/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_69_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_69/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_69_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_69/FusedBatchNormV3FusedBatchNormV3conv2d_69/BiasAdd:output:0-batch_normalization_69/ReadVariableOp:value:0/batch_normalization_69/ReadVariableOp_1:value:0>batch_normalization_69/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_69/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_69/AssignNewValueAssignVariableOp?batch_normalization_69_fusedbatchnormv3_readvariableop_resource4batch_normalization_69/FusedBatchNormV3:batch_mean:07^batch_normalization_69/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_69/AssignNewValue_1AssignVariableOpAbatch_normalization_69_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_69/FusedBatchNormV3:batch_variance:09^batch_normalization_69/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
encoded/CastCast+batch_normalization_69/FusedBatchNormV3:y:0*

DstT0*

SrcT0*/
_output_shapes
:????????? j
encoded/LeakyRelu	LeakyReluencoded/Cast:y:0*
T0*/
_output_shapes
:????????? ?
conv2d_transpose_27/CastCastencoded/LeakyRelu:activations:0*

DstT0*

SrcT0*/
_output_shapes
:????????? e
conv2d_transpose_27/ShapeShapeconv2d_transpose_27/Cast:y:0*
T0*
_output_shapes
:q
'conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_27/strided_sliceStridedSlice"conv2d_transpose_27/Shape:output:00conv2d_transpose_27/strided_slice/stack:output:02conv2d_transpose_27/strided_slice/stack_1:output:02conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_27/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_27/stack/2Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose_27/stackPack*conv2d_transpose_27/strided_slice:output:0$conv2d_transpose_27/stack/1:output:0$conv2d_transpose_27/stack/2:output:0$conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_27/strided_slice_1StridedSlice"conv2d_transpose_27/stack:output:02conv2d_transpose_27/strided_slice_1/stack:output:04conv2d_transpose_27/strided_slice_1/stack_1:output:04conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_27_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
$conv2d_transpose_27/conv2d_transposeConv2DBackpropInput"conv2d_transpose_27/stack:output:0;conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0conv2d_transpose_27/Cast:y:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
*conv2d_transpose_27/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_transpose_27/BiasAddBiasAdd-conv2d_transpose_27/conv2d_transpose:output:02conv2d_transpose_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @{
activation_87/LeakyRelu	LeakyRelu$conv2d_transpose_27/BiasAdd:output:0*/
_output_shapes
:?????????  @n
conv2d_transpose_28/ShapeShape%activation_87/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_28/strided_sliceStridedSlice"conv2d_transpose_28/Shape:output:00conv2d_transpose_28/strided_slice/stack:output:02conv2d_transpose_28/strided_slice/stack_1:output:02conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_28/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@]
conv2d_transpose_28/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@]
conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_28/stackPack*conv2d_transpose_28/strided_slice:output:0$conv2d_transpose_28/stack/1:output:0$conv2d_transpose_28/stack/2:output:0$conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_28/strided_slice_1StridedSlice"conv2d_transpose_28/stack:output:02conv2d_transpose_28/strided_slice_1/stack:output:04conv2d_transpose_28/strided_slice_1/stack_1:output:04conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_28_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
$conv2d_transpose_28/conv2d_transposeConv2DBackpropInput"conv2d_transpose_28/stack:output:0;conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:0%activation_87/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
*conv2d_transpose_28/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_28/BiasAddBiasAdd-conv2d_transpose_28/conv2d_transpose:output:02conv2d_transpose_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ {
activation_88/LeakyRelu	LeakyRelu$conv2d_transpose_28/BiasAdd:output:0*/
_output_shapes
:?????????@@ n
conv2d_transpose_29/ShapeShape%activation_88/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_29/strided_sliceStridedSlice"conv2d_transpose_29/Shape:output:00conv2d_transpose_29/strided_slice/stack:output:02conv2d_transpose_29/strided_slice/stack_1:output:02conv2d_transpose_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_29/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?^
conv2d_transpose_29/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?]
conv2d_transpose_29/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_29/stackPack*conv2d_transpose_29/strided_slice:output:0$conv2d_transpose_29/stack/1:output:0$conv2d_transpose_29/stack/2:output:0$conv2d_transpose_29/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_29/strided_slice_1StridedSlice"conv2d_transpose_29/stack:output:02conv2d_transpose_29/strided_slice_1/stack:output:04conv2d_transpose_29/strided_slice_1/stack_1:output:04conv2d_transpose_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_29/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_29_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
$conv2d_transpose_29/conv2d_transposeConv2DBackpropInput"conv2d_transpose_29/stack:output:0;conv2d_transpose_29/conv2d_transpose/ReadVariableOp:value:0%activation_88/LeakyRelu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
*conv2d_transpose_29/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_29/BiasAddBiasAdd-conv2d_transpose_29/conv2d_transpose:output:02conv2d_transpose_29/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????}
activation_89/LeakyRelu	LeakyRelu$conv2d_transpose_29/BiasAdd:output:0*1
_output_shapes
:???????????b
decoded/ShapeShape%activation_89/LeakyRelu:activations:0*
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
decoded/conv2d_transposeConv2DBackpropInputdecoded/stack:output:0/decoded/conv2d_transpose/ReadVariableOp:value:0%activation_89/LeakyRelu:activations:0*
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
NoOpNoOp&^batch_normalization_63/AssignNewValue(^batch_normalization_63/AssignNewValue_17^batch_normalization_63/FusedBatchNormV3/ReadVariableOp9^batch_normalization_63/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_63/ReadVariableOp(^batch_normalization_63/ReadVariableOp_1&^batch_normalization_64/AssignNewValue(^batch_normalization_64/AssignNewValue_17^batch_normalization_64/FusedBatchNormV3/ReadVariableOp9^batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_64/ReadVariableOp(^batch_normalization_64/ReadVariableOp_1&^batch_normalization_65/AssignNewValue(^batch_normalization_65/AssignNewValue_17^batch_normalization_65/FusedBatchNormV3/ReadVariableOp9^batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_65/ReadVariableOp(^batch_normalization_65/ReadVariableOp_1&^batch_normalization_66/AssignNewValue(^batch_normalization_66/AssignNewValue_17^batch_normalization_66/FusedBatchNormV3/ReadVariableOp9^batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_66/ReadVariableOp(^batch_normalization_66/ReadVariableOp_1&^batch_normalization_67/AssignNewValue(^batch_normalization_67/AssignNewValue_17^batch_normalization_67/FusedBatchNormV3/ReadVariableOp9^batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_67/ReadVariableOp(^batch_normalization_67/ReadVariableOp_1&^batch_normalization_68/AssignNewValue(^batch_normalization_68/AssignNewValue_17^batch_normalization_68/FusedBatchNormV3/ReadVariableOp9^batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_68/ReadVariableOp(^batch_normalization_68/ReadVariableOp_1&^batch_normalization_69/AssignNewValue(^batch_normalization_69/AssignNewValue_17^batch_normalization_69/FusedBatchNormV3/ReadVariableOp9^batch_normalization_69/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_69/ReadVariableOp(^batch_normalization_69/ReadVariableOp_1!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp!^conv2d_64/BiasAdd/ReadVariableOp ^conv2d_64/Conv2D/ReadVariableOp!^conv2d_65/BiasAdd/ReadVariableOp ^conv2d_65/Conv2D/ReadVariableOp!^conv2d_66/BiasAdd/ReadVariableOp ^conv2d_66/Conv2D/ReadVariableOp!^conv2d_67/BiasAdd/ReadVariableOp ^conv2d_67/Conv2D/ReadVariableOp!^conv2d_68/BiasAdd/ReadVariableOp ^conv2d_68/Conv2D/ReadVariableOp!^conv2d_69/BiasAdd/ReadVariableOp ^conv2d_69/Conv2D/ReadVariableOp+^conv2d_transpose_27/BiasAdd/ReadVariableOp4^conv2d_transpose_27/conv2d_transpose/ReadVariableOp+^conv2d_transpose_28/BiasAdd/ReadVariableOp4^conv2d_transpose_28/conv2d_transpose/ReadVariableOp+^conv2d_transpose_29/BiasAdd/ReadVariableOp4^conv2d_transpose_29/conv2d_transpose/ReadVariableOp^decoded/BiasAdd/ReadVariableOp(^decoded/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_63/AssignNewValue%batch_normalization_63/AssignNewValue2R
'batch_normalization_63/AssignNewValue_1'batch_normalization_63/AssignNewValue_12p
6batch_normalization_63/FusedBatchNormV3/ReadVariableOp6batch_normalization_63/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_63/FusedBatchNormV3/ReadVariableOp_18batch_normalization_63/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_63/ReadVariableOp%batch_normalization_63/ReadVariableOp2R
'batch_normalization_63/ReadVariableOp_1'batch_normalization_63/ReadVariableOp_12N
%batch_normalization_64/AssignNewValue%batch_normalization_64/AssignNewValue2R
'batch_normalization_64/AssignNewValue_1'batch_normalization_64/AssignNewValue_12p
6batch_normalization_64/FusedBatchNormV3/ReadVariableOp6batch_normalization_64/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_64/FusedBatchNormV3/ReadVariableOp_18batch_normalization_64/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_64/ReadVariableOp%batch_normalization_64/ReadVariableOp2R
'batch_normalization_64/ReadVariableOp_1'batch_normalization_64/ReadVariableOp_12N
%batch_normalization_65/AssignNewValue%batch_normalization_65/AssignNewValue2R
'batch_normalization_65/AssignNewValue_1'batch_normalization_65/AssignNewValue_12p
6batch_normalization_65/FusedBatchNormV3/ReadVariableOp6batch_normalization_65/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_65/FusedBatchNormV3/ReadVariableOp_18batch_normalization_65/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_65/ReadVariableOp%batch_normalization_65/ReadVariableOp2R
'batch_normalization_65/ReadVariableOp_1'batch_normalization_65/ReadVariableOp_12N
%batch_normalization_66/AssignNewValue%batch_normalization_66/AssignNewValue2R
'batch_normalization_66/AssignNewValue_1'batch_normalization_66/AssignNewValue_12p
6batch_normalization_66/FusedBatchNormV3/ReadVariableOp6batch_normalization_66/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_18batch_normalization_66/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_66/ReadVariableOp%batch_normalization_66/ReadVariableOp2R
'batch_normalization_66/ReadVariableOp_1'batch_normalization_66/ReadVariableOp_12N
%batch_normalization_67/AssignNewValue%batch_normalization_67/AssignNewValue2R
'batch_normalization_67/AssignNewValue_1'batch_normalization_67/AssignNewValue_12p
6batch_normalization_67/FusedBatchNormV3/ReadVariableOp6batch_normalization_67/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_67/FusedBatchNormV3/ReadVariableOp_18batch_normalization_67/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_67/ReadVariableOp%batch_normalization_67/ReadVariableOp2R
'batch_normalization_67/ReadVariableOp_1'batch_normalization_67/ReadVariableOp_12N
%batch_normalization_68/AssignNewValue%batch_normalization_68/AssignNewValue2R
'batch_normalization_68/AssignNewValue_1'batch_normalization_68/AssignNewValue_12p
6batch_normalization_68/FusedBatchNormV3/ReadVariableOp6batch_normalization_68/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_68/FusedBatchNormV3/ReadVariableOp_18batch_normalization_68/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_68/ReadVariableOp%batch_normalization_68/ReadVariableOp2R
'batch_normalization_68/ReadVariableOp_1'batch_normalization_68/ReadVariableOp_12N
%batch_normalization_69/AssignNewValue%batch_normalization_69/AssignNewValue2R
'batch_normalization_69/AssignNewValue_1'batch_normalization_69/AssignNewValue_12p
6batch_normalization_69/FusedBatchNormV3/ReadVariableOp6batch_normalization_69/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_69/FusedBatchNormV3/ReadVariableOp_18batch_normalization_69/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_69/ReadVariableOp%batch_normalization_69/ReadVariableOp2R
'batch_normalization_69/ReadVariableOp_1'batch_normalization_69/ReadVariableOp_12D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp2D
 conv2d_64/BiasAdd/ReadVariableOp conv2d_64/BiasAdd/ReadVariableOp2B
conv2d_64/Conv2D/ReadVariableOpconv2d_64/Conv2D/ReadVariableOp2D
 conv2d_65/BiasAdd/ReadVariableOp conv2d_65/BiasAdd/ReadVariableOp2B
conv2d_65/Conv2D/ReadVariableOpconv2d_65/Conv2D/ReadVariableOp2D
 conv2d_66/BiasAdd/ReadVariableOp conv2d_66/BiasAdd/ReadVariableOp2B
conv2d_66/Conv2D/ReadVariableOpconv2d_66/Conv2D/ReadVariableOp2D
 conv2d_67/BiasAdd/ReadVariableOp conv2d_67/BiasAdd/ReadVariableOp2B
conv2d_67/Conv2D/ReadVariableOpconv2d_67/Conv2D/ReadVariableOp2D
 conv2d_68/BiasAdd/ReadVariableOp conv2d_68/BiasAdd/ReadVariableOp2B
conv2d_68/Conv2D/ReadVariableOpconv2d_68/Conv2D/ReadVariableOp2D
 conv2d_69/BiasAdd/ReadVariableOp conv2d_69/BiasAdd/ReadVariableOp2B
conv2d_69/Conv2D/ReadVariableOpconv2d_69/Conv2D/ReadVariableOp2X
*conv2d_transpose_27/BiasAdd/ReadVariableOp*conv2d_transpose_27/BiasAdd/ReadVariableOp2j
3conv2d_transpose_27/conv2d_transpose/ReadVariableOp3conv2d_transpose_27/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_28/BiasAdd/ReadVariableOp*conv2d_transpose_28/BiasAdd/ReadVariableOp2j
3conv2d_transpose_28/conv2d_transpose/ReadVariableOp3conv2d_transpose_28/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_29/BiasAdd/ReadVariableOp*conv2d_transpose_29/BiasAdd/ReadVariableOp2j
3conv2d_transpose_29/conv2d_transpose/ReadVariableOp3conv2d_transpose_29/conv2d_transpose/ReadVariableOp2@
decoded/BiasAdd/ReadVariableOpdecoded/BiasAdd/ReadVariableOp2R
'decoded/conv2d_transpose/ReadVariableOp'decoded/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3371642

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
P__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_3371906

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

?
F__inference_conv2d_64_layer_call_and_return_conditional_losses_3372007

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
?
f
J__inference_activation_82_layer_call_and_return_conditional_losses_3374154

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
??
?<
 __inference__traced_save_3375218
file_prefix/
+savev2_conv2d_63_kernel_read_readvariableop-
)savev2_conv2d_63_bias_read_readvariableop;
7savev2_batch_normalization_63_gamma_read_readvariableop:
6savev2_batch_normalization_63_beta_read_readvariableopA
=savev2_batch_normalization_63_moving_mean_read_readvariableopE
Asavev2_batch_normalization_63_moving_variance_read_readvariableop/
+savev2_conv2d_64_kernel_read_readvariableop-
)savev2_conv2d_64_bias_read_readvariableop;
7savev2_batch_normalization_64_gamma_read_readvariableop:
6savev2_batch_normalization_64_beta_read_readvariableopA
=savev2_batch_normalization_64_moving_mean_read_readvariableopE
Asavev2_batch_normalization_64_moving_variance_read_readvariableop/
+savev2_conv2d_65_kernel_read_readvariableop-
)savev2_conv2d_65_bias_read_readvariableop;
7savev2_batch_normalization_65_gamma_read_readvariableop:
6savev2_batch_normalization_65_beta_read_readvariableopA
=savev2_batch_normalization_65_moving_mean_read_readvariableopE
Asavev2_batch_normalization_65_moving_variance_read_readvariableop/
+savev2_conv2d_66_kernel_read_readvariableop-
)savev2_conv2d_66_bias_read_readvariableop;
7savev2_batch_normalization_66_gamma_read_readvariableop:
6savev2_batch_normalization_66_beta_read_readvariableopA
=savev2_batch_normalization_66_moving_mean_read_readvariableopE
Asavev2_batch_normalization_66_moving_variance_read_readvariableop/
+savev2_conv2d_67_kernel_read_readvariableop-
)savev2_conv2d_67_bias_read_readvariableop;
7savev2_batch_normalization_67_gamma_read_readvariableop:
6savev2_batch_normalization_67_beta_read_readvariableopA
=savev2_batch_normalization_67_moving_mean_read_readvariableopE
Asavev2_batch_normalization_67_moving_variance_read_readvariableop/
+savev2_conv2d_68_kernel_read_readvariableop-
)savev2_conv2d_68_bias_read_readvariableop;
7savev2_batch_normalization_68_gamma_read_readvariableop:
6savev2_batch_normalization_68_beta_read_readvariableopA
=savev2_batch_normalization_68_moving_mean_read_readvariableopE
Asavev2_batch_normalization_68_moving_variance_read_readvariableop/
+savev2_conv2d_69_kernel_read_readvariableop-
)savev2_conv2d_69_bias_read_readvariableop;
7savev2_batch_normalization_69_gamma_read_readvariableop:
6savev2_batch_normalization_69_beta_read_readvariableopA
=savev2_batch_normalization_69_moving_mean_read_readvariableopE
Asavev2_batch_normalization_69_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_27_kernel_read_readvariableop7
3savev2_conv2d_transpose_27_bias_read_readvariableop9
5savev2_conv2d_transpose_28_kernel_read_readvariableop7
3savev2_conv2d_transpose_28_bias_read_readvariableop9
5savev2_conv2d_transpose_29_kernel_read_readvariableop7
3savev2_conv2d_transpose_29_bias_read_readvariableop-
)savev2_decoded_kernel_read_readvariableop+
'savev2_decoded_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_63_kernel_m_read_readvariableop4
0savev2_adam_conv2d_63_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_63_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_63_beta_m_read_readvariableop6
2savev2_adam_conv2d_64_kernel_m_read_readvariableop4
0savev2_adam_conv2d_64_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_64_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_64_beta_m_read_readvariableop6
2savev2_adam_conv2d_65_kernel_m_read_readvariableop4
0savev2_adam_conv2d_65_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_65_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_65_beta_m_read_readvariableop6
2savev2_adam_conv2d_66_kernel_m_read_readvariableop4
0savev2_adam_conv2d_66_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_66_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_66_beta_m_read_readvariableop6
2savev2_adam_conv2d_67_kernel_m_read_readvariableop4
0savev2_adam_conv2d_67_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_67_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_67_beta_m_read_readvariableop6
2savev2_adam_conv2d_68_kernel_m_read_readvariableop4
0savev2_adam_conv2d_68_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_68_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_68_beta_m_read_readvariableop6
2savev2_adam_conv2d_69_kernel_m_read_readvariableop4
0savev2_adam_conv2d_69_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_69_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_69_beta_m_read_readvariableop@
<savev2_adam_conv2d_transpose_27_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_27_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_28_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_28_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_29_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_29_bias_m_read_readvariableop4
0savev2_adam_decoded_kernel_m_read_readvariableop2
.savev2_adam_decoded_bias_m_read_readvariableop6
2savev2_adam_conv2d_63_kernel_v_read_readvariableop4
0savev2_adam_conv2d_63_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_63_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_63_beta_v_read_readvariableop6
2savev2_adam_conv2d_64_kernel_v_read_readvariableop4
0savev2_adam_conv2d_64_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_64_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_64_beta_v_read_readvariableop6
2savev2_adam_conv2d_65_kernel_v_read_readvariableop4
0savev2_adam_conv2d_65_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_65_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_65_beta_v_read_readvariableop6
2savev2_adam_conv2d_66_kernel_v_read_readvariableop4
0savev2_adam_conv2d_66_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_66_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_66_beta_v_read_readvariableop6
2savev2_adam_conv2d_67_kernel_v_read_readvariableop4
0savev2_adam_conv2d_67_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_67_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_67_beta_v_read_readvariableop6
2savev2_adam_conv2d_68_kernel_v_read_readvariableop4
0savev2_adam_conv2d_68_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_68_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_68_beta_v_read_readvariableop6
2savev2_adam_conv2d_69_kernel_v_read_readvariableop4
0savev2_adam_conv2d_69_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_69_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_69_beta_v_read_readvariableop@
<savev2_adam_conv2d_transpose_27_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_27_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_28_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_28_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_29_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_29_bias_v_read_readvariableop4
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_63_kernel_read_readvariableop)savev2_conv2d_63_bias_read_readvariableop7savev2_batch_normalization_63_gamma_read_readvariableop6savev2_batch_normalization_63_beta_read_readvariableop=savev2_batch_normalization_63_moving_mean_read_readvariableopAsavev2_batch_normalization_63_moving_variance_read_readvariableop+savev2_conv2d_64_kernel_read_readvariableop)savev2_conv2d_64_bias_read_readvariableop7savev2_batch_normalization_64_gamma_read_readvariableop6savev2_batch_normalization_64_beta_read_readvariableop=savev2_batch_normalization_64_moving_mean_read_readvariableopAsavev2_batch_normalization_64_moving_variance_read_readvariableop+savev2_conv2d_65_kernel_read_readvariableop)savev2_conv2d_65_bias_read_readvariableop7savev2_batch_normalization_65_gamma_read_readvariableop6savev2_batch_normalization_65_beta_read_readvariableop=savev2_batch_normalization_65_moving_mean_read_readvariableopAsavev2_batch_normalization_65_moving_variance_read_readvariableop+savev2_conv2d_66_kernel_read_readvariableop)savev2_conv2d_66_bias_read_readvariableop7savev2_batch_normalization_66_gamma_read_readvariableop6savev2_batch_normalization_66_beta_read_readvariableop=savev2_batch_normalization_66_moving_mean_read_readvariableopAsavev2_batch_normalization_66_moving_variance_read_readvariableop+savev2_conv2d_67_kernel_read_readvariableop)savev2_conv2d_67_bias_read_readvariableop7savev2_batch_normalization_67_gamma_read_readvariableop6savev2_batch_normalization_67_beta_read_readvariableop=savev2_batch_normalization_67_moving_mean_read_readvariableopAsavev2_batch_normalization_67_moving_variance_read_readvariableop+savev2_conv2d_68_kernel_read_readvariableop)savev2_conv2d_68_bias_read_readvariableop7savev2_batch_normalization_68_gamma_read_readvariableop6savev2_batch_normalization_68_beta_read_readvariableop=savev2_batch_normalization_68_moving_mean_read_readvariableopAsavev2_batch_normalization_68_moving_variance_read_readvariableop+savev2_conv2d_69_kernel_read_readvariableop)savev2_conv2d_69_bias_read_readvariableop7savev2_batch_normalization_69_gamma_read_readvariableop6savev2_batch_normalization_69_beta_read_readvariableop=savev2_batch_normalization_69_moving_mean_read_readvariableopAsavev2_batch_normalization_69_moving_variance_read_readvariableop5savev2_conv2d_transpose_27_kernel_read_readvariableop3savev2_conv2d_transpose_27_bias_read_readvariableop5savev2_conv2d_transpose_28_kernel_read_readvariableop3savev2_conv2d_transpose_28_bias_read_readvariableop5savev2_conv2d_transpose_29_kernel_read_readvariableop3savev2_conv2d_transpose_29_bias_read_readvariableop)savev2_decoded_kernel_read_readvariableop'savev2_decoded_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_63_kernel_m_read_readvariableop0savev2_adam_conv2d_63_bias_m_read_readvariableop>savev2_adam_batch_normalization_63_gamma_m_read_readvariableop=savev2_adam_batch_normalization_63_beta_m_read_readvariableop2savev2_adam_conv2d_64_kernel_m_read_readvariableop0savev2_adam_conv2d_64_bias_m_read_readvariableop>savev2_adam_batch_normalization_64_gamma_m_read_readvariableop=savev2_adam_batch_normalization_64_beta_m_read_readvariableop2savev2_adam_conv2d_65_kernel_m_read_readvariableop0savev2_adam_conv2d_65_bias_m_read_readvariableop>savev2_adam_batch_normalization_65_gamma_m_read_readvariableop=savev2_adam_batch_normalization_65_beta_m_read_readvariableop2savev2_adam_conv2d_66_kernel_m_read_readvariableop0savev2_adam_conv2d_66_bias_m_read_readvariableop>savev2_adam_batch_normalization_66_gamma_m_read_readvariableop=savev2_adam_batch_normalization_66_beta_m_read_readvariableop2savev2_adam_conv2d_67_kernel_m_read_readvariableop0savev2_adam_conv2d_67_bias_m_read_readvariableop>savev2_adam_batch_normalization_67_gamma_m_read_readvariableop=savev2_adam_batch_normalization_67_beta_m_read_readvariableop2savev2_adam_conv2d_68_kernel_m_read_readvariableop0savev2_adam_conv2d_68_bias_m_read_readvariableop>savev2_adam_batch_normalization_68_gamma_m_read_readvariableop=savev2_adam_batch_normalization_68_beta_m_read_readvariableop2savev2_adam_conv2d_69_kernel_m_read_readvariableop0savev2_adam_conv2d_69_bias_m_read_readvariableop>savev2_adam_batch_normalization_69_gamma_m_read_readvariableop=savev2_adam_batch_normalization_69_beta_m_read_readvariableop<savev2_adam_conv2d_transpose_27_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_27_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_28_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_28_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_29_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_29_bias_m_read_readvariableop0savev2_adam_decoded_kernel_m_read_readvariableop.savev2_adam_decoded_bias_m_read_readvariableop2savev2_adam_conv2d_63_kernel_v_read_readvariableop0savev2_adam_conv2d_63_bias_v_read_readvariableop>savev2_adam_batch_normalization_63_gamma_v_read_readvariableop=savev2_adam_batch_normalization_63_beta_v_read_readvariableop2savev2_adam_conv2d_64_kernel_v_read_readvariableop0savev2_adam_conv2d_64_bias_v_read_readvariableop>savev2_adam_batch_normalization_64_gamma_v_read_readvariableop=savev2_adam_batch_normalization_64_beta_v_read_readvariableop2savev2_adam_conv2d_65_kernel_v_read_readvariableop0savev2_adam_conv2d_65_bias_v_read_readvariableop>savev2_adam_batch_normalization_65_gamma_v_read_readvariableop=savev2_adam_batch_normalization_65_beta_v_read_readvariableop2savev2_adam_conv2d_66_kernel_v_read_readvariableop0savev2_adam_conv2d_66_bias_v_read_readvariableop>savev2_adam_batch_normalization_66_gamma_v_read_readvariableop=savev2_adam_batch_normalization_66_beta_v_read_readvariableop2savev2_adam_conv2d_67_kernel_v_read_readvariableop0savev2_adam_conv2d_67_bias_v_read_readvariableop>savev2_adam_batch_normalization_67_gamma_v_read_readvariableop=savev2_adam_batch_normalization_67_beta_v_read_readvariableop2savev2_adam_conv2d_68_kernel_v_read_readvariableop0savev2_adam_conv2d_68_bias_v_read_readvariableop>savev2_adam_batch_normalization_68_gamma_v_read_readvariableop=savev2_adam_batch_normalization_68_beta_v_read_readvariableop2savev2_adam_conv2d_69_kernel_v_read_readvariableop0savev2_adam_conv2d_69_bias_v_read_readvariableop>savev2_adam_batch_normalization_69_gamma_v_read_readvariableop=savev2_adam_batch_normalization_69_beta_v_read_readvariableop<savev2_adam_conv2d_transpose_27_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_27_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_28_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_28_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_29_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_29_bias_v_read_readvariableop0savev2_adam_decoded_kernel_v_read_readvariableop.savev2_adam_decoded_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?
?
S__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3371514

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
S__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3374235

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
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3374126

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
J__inference_activation_86_layer_call_and_return_conditional_losses_3374518

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
8__inference_batch_normalization_66_layer_call_fn_3374277

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
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3371547?
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
?
?
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3371547

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
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3374035

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
S__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3374217

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
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3374399

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
F__inference_conv2d_65_layer_call_and_return_conditional_losses_3372039

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
8__inference_batch_normalization_63_layer_call_fn_3374017

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
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3371386?
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
??
?7
"__inference__wrapped_model_3371333
conv2d_63_inputO
5sequential_9_conv2d_63_conv2d_readvariableop_resource:D
6sequential_9_conv2d_63_biasadd_readvariableop_resource:I
;sequential_9_batch_normalization_63_readvariableop_resource:K
=sequential_9_batch_normalization_63_readvariableop_1_resource:Z
Lsequential_9_batch_normalization_63_fusedbatchnormv3_readvariableop_resource:\
Nsequential_9_batch_normalization_63_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_64_conv2d_readvariableop_resource:D
6sequential_9_conv2d_64_biasadd_readvariableop_resource:I
;sequential_9_batch_normalization_64_readvariableop_resource:K
=sequential_9_batch_normalization_64_readvariableop_1_resource:Z
Lsequential_9_batch_normalization_64_fusedbatchnormv3_readvariableop_resource:\
Nsequential_9_batch_normalization_64_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_65_conv2d_readvariableop_resource:D
6sequential_9_conv2d_65_biasadd_readvariableop_resource:I
;sequential_9_batch_normalization_65_readvariableop_resource:K
=sequential_9_batch_normalization_65_readvariableop_1_resource:Z
Lsequential_9_batch_normalization_65_fusedbatchnormv3_readvariableop_resource:\
Nsequential_9_batch_normalization_65_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_9_conv2d_66_conv2d_readvariableop_resource: D
6sequential_9_conv2d_66_biasadd_readvariableop_resource: I
;sequential_9_batch_normalization_66_readvariableop_resource: K
=sequential_9_batch_normalization_66_readvariableop_1_resource: Z
Lsequential_9_batch_normalization_66_fusedbatchnormv3_readvariableop_resource: \
Nsequential_9_batch_normalization_66_fusedbatchnormv3_readvariableop_1_resource: O
5sequential_9_conv2d_67_conv2d_readvariableop_resource:  D
6sequential_9_conv2d_67_biasadd_readvariableop_resource: I
;sequential_9_batch_normalization_67_readvariableop_resource: K
=sequential_9_batch_normalization_67_readvariableop_1_resource: Z
Lsequential_9_batch_normalization_67_fusedbatchnormv3_readvariableop_resource: \
Nsequential_9_batch_normalization_67_fusedbatchnormv3_readvariableop_1_resource: O
5sequential_9_conv2d_68_conv2d_readvariableop_resource: @D
6sequential_9_conv2d_68_biasadd_readvariableop_resource:@I
;sequential_9_batch_normalization_68_readvariableop_resource:@K
=sequential_9_batch_normalization_68_readvariableop_1_resource:@Z
Lsequential_9_batch_normalization_68_fusedbatchnormv3_readvariableop_resource:@\
Nsequential_9_batch_normalization_68_fusedbatchnormv3_readvariableop_1_resource:@O
5sequential_9_conv2d_69_conv2d_readvariableop_resource:@ D
6sequential_9_conv2d_69_biasadd_readvariableop_resource: I
;sequential_9_batch_normalization_69_readvariableop_resource: K
=sequential_9_batch_normalization_69_readvariableop_1_resource: Z
Lsequential_9_batch_normalization_69_fusedbatchnormv3_readvariableop_resource: \
Nsequential_9_batch_normalization_69_fusedbatchnormv3_readvariableop_1_resource: c
Isequential_9_conv2d_transpose_27_conv2d_transpose_readvariableop_resource:@ N
@sequential_9_conv2d_transpose_27_biasadd_readvariableop_resource:@c
Isequential_9_conv2d_transpose_28_conv2d_transpose_readvariableop_resource: @N
@sequential_9_conv2d_transpose_28_biasadd_readvariableop_resource: c
Isequential_9_conv2d_transpose_29_conv2d_transpose_readvariableop_resource: N
@sequential_9_conv2d_transpose_29_biasadd_readvariableop_resource:W
=sequential_9_decoded_conv2d_transpose_readvariableop_resource:B
4sequential_9_decoded_biasadd_readvariableop_resource:
identity??Csequential_9/batch_normalization_63/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_63/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_63/ReadVariableOp?4sequential_9/batch_normalization_63/ReadVariableOp_1?Csequential_9/batch_normalization_64/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_64/ReadVariableOp?4sequential_9/batch_normalization_64/ReadVariableOp_1?Csequential_9/batch_normalization_65/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_65/ReadVariableOp?4sequential_9/batch_normalization_65/ReadVariableOp_1?Csequential_9/batch_normalization_66/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_66/ReadVariableOp?4sequential_9/batch_normalization_66/ReadVariableOp_1?Csequential_9/batch_normalization_67/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_67/ReadVariableOp?4sequential_9/batch_normalization_67/ReadVariableOp_1?Csequential_9/batch_normalization_68/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_68/ReadVariableOp?4sequential_9/batch_normalization_68/ReadVariableOp_1?Csequential_9/batch_normalization_69/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_69/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_69/ReadVariableOp?4sequential_9/batch_normalization_69/ReadVariableOp_1?-sequential_9/conv2d_63/BiasAdd/ReadVariableOp?,sequential_9/conv2d_63/Conv2D/ReadVariableOp?-sequential_9/conv2d_64/BiasAdd/ReadVariableOp?,sequential_9/conv2d_64/Conv2D/ReadVariableOp?-sequential_9/conv2d_65/BiasAdd/ReadVariableOp?,sequential_9/conv2d_65/Conv2D/ReadVariableOp?-sequential_9/conv2d_66/BiasAdd/ReadVariableOp?,sequential_9/conv2d_66/Conv2D/ReadVariableOp?-sequential_9/conv2d_67/BiasAdd/ReadVariableOp?,sequential_9/conv2d_67/Conv2D/ReadVariableOp?-sequential_9/conv2d_68/BiasAdd/ReadVariableOp?,sequential_9/conv2d_68/Conv2D/ReadVariableOp?-sequential_9/conv2d_69/BiasAdd/ReadVariableOp?,sequential_9/conv2d_69/Conv2D/ReadVariableOp?7sequential_9/conv2d_transpose_27/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_27/conv2d_transpose/ReadVariableOp?7sequential_9/conv2d_transpose_28/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_28/conv2d_transpose/ReadVariableOp?7sequential_9/conv2d_transpose_29/BiasAdd/ReadVariableOp?@sequential_9/conv2d_transpose_29/conv2d_transpose/ReadVariableOp?+sequential_9/decoded/BiasAdd/ReadVariableOp?4sequential_9/decoded/conv2d_transpose/ReadVariableOp?
,sequential_9/conv2d_63/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_63_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_9/conv2d_63/Conv2DConv2Dconv2d_63_input4sequential_9/conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
-sequential_9/conv2d_63/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_9/conv2d_63/BiasAddBiasAdd&sequential_9/conv2d_63/Conv2D:output:05sequential_9/conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
2sequential_9/batch_normalization_63/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_63_readvariableop_resource*
_output_shapes
:*
dtype0?
4sequential_9/batch_normalization_63/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_63_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Csequential_9/batch_normalization_63/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_63_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Esequential_9/batch_normalization_63/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_63_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
4sequential_9/batch_normalization_63/FusedBatchNormV3FusedBatchNormV3'sequential_9/conv2d_63/BiasAdd:output:0:sequential_9/batch_normalization_63/ReadVariableOp:value:0<sequential_9/batch_normalization_63/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_63/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_63/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
$sequential_9/activation_81/LeakyRelu	LeakyRelu8sequential_9/batch_normalization_63/FusedBatchNormV3:y:0*1
_output_shapes
:????????????
,sequential_9/conv2d_64/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_64_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_9/conv2d_64/Conv2DConv2D2sequential_9/activation_81/LeakyRelu:activations:04sequential_9/conv2d_64/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
-sequential_9/conv2d_64/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_64_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_9/conv2d_64/BiasAddBiasAdd&sequential_9/conv2d_64/Conv2D:output:05sequential_9/conv2d_64/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
2sequential_9/batch_normalization_64/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_64_readvariableop_resource*
_output_shapes
:*
dtype0?
4sequential_9/batch_normalization_64/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_64_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Csequential_9/batch_normalization_64/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_64_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Esequential_9/batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_64_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
4sequential_9/batch_normalization_64/FusedBatchNormV3FusedBatchNormV3'sequential_9/conv2d_64/BiasAdd:output:0:sequential_9/batch_normalization_64/ReadVariableOp:value:0<sequential_9/batch_normalization_64/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_64/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
$sequential_9/activation_82/LeakyRelu	LeakyRelu8sequential_9/batch_normalization_64/FusedBatchNormV3:y:0*1
_output_shapes
:????????????
,sequential_9/conv2d_65/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_65_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_9/conv2d_65/Conv2DConv2D2sequential_9/activation_82/LeakyRelu:activations:04sequential_9/conv2d_65/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
-sequential_9/conv2d_65/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_65_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_9/conv2d_65/BiasAddBiasAdd&sequential_9/conv2d_65/Conv2D:output:05sequential_9/conv2d_65/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
2sequential_9/batch_normalization_65/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_65_readvariableop_resource*
_output_shapes
:*
dtype0?
4sequential_9/batch_normalization_65/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_65_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Csequential_9/batch_normalization_65/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_65_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Esequential_9/batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_65_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
4sequential_9/batch_normalization_65/FusedBatchNormV3FusedBatchNormV3'sequential_9/conv2d_65/BiasAdd:output:0:sequential_9/batch_normalization_65/ReadVariableOp:value:0<sequential_9/batch_normalization_65/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_65/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
$sequential_9/activation_83/LeakyRelu	LeakyRelu8sequential_9/batch_normalization_65/FusedBatchNormV3:y:0*1
_output_shapes
:????????????
,sequential_9/conv2d_66/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_66_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
sequential_9/conv2d_66/Conv2DConv2D2sequential_9/activation_83/LeakyRelu:activations:04sequential_9/conv2d_66/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
-sequential_9/conv2d_66/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_66_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_9/conv2d_66/BiasAddBiasAdd&sequential_9/conv2d_66/Conv2D:output:05sequential_9/conv2d_66/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
2sequential_9/batch_normalization_66/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_66_readvariableop_resource*
_output_shapes
: *
dtype0?
4sequential_9/batch_normalization_66/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_66_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Csequential_9/batch_normalization_66/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_66_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Esequential_9/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_66_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
4sequential_9/batch_normalization_66/FusedBatchNormV3FusedBatchNormV3'sequential_9/conv2d_66/BiasAdd:output:0:sequential_9/batch_normalization_66/ReadVariableOp:value:0<sequential_9/batch_normalization_66/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_66/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( ?
$sequential_9/activation_84/LeakyRelu	LeakyRelu8sequential_9/batch_normalization_66/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@ ?
,sequential_9/conv2d_67/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_67_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
sequential_9/conv2d_67/Conv2DConv2D2sequential_9/activation_84/LeakyRelu:activations:04sequential_9/conv2d_67/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
-sequential_9/conv2d_67/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_67_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_9/conv2d_67/BiasAddBiasAdd&sequential_9/conv2d_67/Conv2D:output:05sequential_9/conv2d_67/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
2sequential_9/batch_normalization_67/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_67_readvariableop_resource*
_output_shapes
: *
dtype0?
4sequential_9/batch_normalization_67/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_67_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Csequential_9/batch_normalization_67/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_67_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Esequential_9/batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_67_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
4sequential_9/batch_normalization_67/FusedBatchNormV3FusedBatchNormV3'sequential_9/conv2d_67/BiasAdd:output:0:sequential_9/batch_normalization_67/ReadVariableOp:value:0<sequential_9/batch_normalization_67/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_67/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( ?
$sequential_9/activation_85/LeakyRelu	LeakyRelu8sequential_9/batch_normalization_67/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@ ?
,sequential_9/conv2d_68/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_68_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
sequential_9/conv2d_68/Conv2DConv2D2sequential_9/activation_85/LeakyRelu:activations:04sequential_9/conv2d_68/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
-sequential_9/conv2d_68/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_68_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_9/conv2d_68/BiasAddBiasAdd&sequential_9/conv2d_68/Conv2D:output:05sequential_9/conv2d_68/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
2sequential_9/batch_normalization_68/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_68_readvariableop_resource*
_output_shapes
:@*
dtype0?
4sequential_9/batch_normalization_68/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_68_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Csequential_9/batch_normalization_68/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_68_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Esequential_9/batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_68_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
4sequential_9/batch_normalization_68/FusedBatchNormV3FusedBatchNormV3'sequential_9/conv2d_68/BiasAdd:output:0:sequential_9/batch_normalization_68/ReadVariableOp:value:0<sequential_9/batch_normalization_68/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_68/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( ?
$sequential_9/activation_86/LeakyRelu	LeakyRelu8sequential_9/batch_normalization_68/FusedBatchNormV3:y:0*/
_output_shapes
:?????????  @?
,sequential_9/conv2d_69/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_69_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
sequential_9/conv2d_69/Conv2DConv2D2sequential_9/activation_86/LeakyRelu:activations:04sequential_9/conv2d_69/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
-sequential_9/conv2d_69/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_69_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_9/conv2d_69/BiasAddBiasAdd&sequential_9/conv2d_69/Conv2D:output:05sequential_9/conv2d_69/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
2sequential_9/batch_normalization_69/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_69_readvariableop_resource*
_output_shapes
: *
dtype0?
4sequential_9/batch_normalization_69/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_69_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Csequential_9/batch_normalization_69/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_69_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Esequential_9/batch_normalization_69/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_69_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
4sequential_9/batch_normalization_69/FusedBatchNormV3FusedBatchNormV3'sequential_9/conv2d_69/BiasAdd:output:0:sequential_9/batch_normalization_69/ReadVariableOp:value:0<sequential_9/batch_normalization_69/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_69/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_69/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
sequential_9/encoded/CastCast8sequential_9/batch_normalization_69/FusedBatchNormV3:y:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
sequential_9/encoded/LeakyRelu	LeakyRelusequential_9/encoded/Cast:y:0*
T0*/
_output_shapes
:????????? ?
%sequential_9/conv2d_transpose_27/CastCast,sequential_9/encoded/LeakyRelu:activations:0*

DstT0*

SrcT0*/
_output_shapes
:????????? 
&sequential_9/conv2d_transpose_27/ShapeShape)sequential_9/conv2d_transpose_27/Cast:y:0*
T0*
_output_shapes
:~
4sequential_9/conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6sequential_9/conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential_9/conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.sequential_9/conv2d_transpose_27/strided_sliceStridedSlice/sequential_9/conv2d_transpose_27/Shape:output:0=sequential_9/conv2d_transpose_27/strided_slice/stack:output:0?sequential_9/conv2d_transpose_27/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_9/conv2d_transpose_27/stack/1Const*
_output_shapes
: *
dtype0*
value	B : j
(sequential_9/conv2d_transpose_27/stack/2Const*
_output_shapes
: *
dtype0*
value	B : j
(sequential_9/conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
&sequential_9/conv2d_transpose_27/stackPack7sequential_9/conv2d_transpose_27/strided_slice:output:01sequential_9/conv2d_transpose_27/stack/1:output:01sequential_9/conv2d_transpose_27/stack/2:output:01sequential_9/conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:?
6sequential_9/conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8sequential_9/conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_9/conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0sequential_9/conv2d_transpose_27/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_27/stack:output:0?sequential_9/conv2d_transpose_27/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_27/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_9/conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_27_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
1sequential_9/conv2d_transpose_27/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_27/stack:output:0Hsequential_9/conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0)sequential_9/conv2d_transpose_27/Cast:y:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
7sequential_9/conv2d_transpose_27/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
(sequential_9/conv2d_transpose_27/BiasAddBiasAdd:sequential_9/conv2d_transpose_27/conv2d_transpose:output:0?sequential_9/conv2d_transpose_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
$sequential_9/activation_87/LeakyRelu	LeakyRelu1sequential_9/conv2d_transpose_27/BiasAdd:output:0*/
_output_shapes
:?????????  @?
&sequential_9/conv2d_transpose_28/ShapeShape2sequential_9/activation_87/LeakyRelu:activations:0*
T0*
_output_shapes
:~
4sequential_9/conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6sequential_9/conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential_9/conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.sequential_9/conv2d_transpose_28/strided_sliceStridedSlice/sequential_9/conv2d_transpose_28/Shape:output:0=sequential_9/conv2d_transpose_28/strided_slice/stack:output:0?sequential_9/conv2d_transpose_28/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_9/conv2d_transpose_28/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@j
(sequential_9/conv2d_transpose_28/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@j
(sequential_9/conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_9/conv2d_transpose_28/stackPack7sequential_9/conv2d_transpose_28/strided_slice:output:01sequential_9/conv2d_transpose_28/stack/1:output:01sequential_9/conv2d_transpose_28/stack/2:output:01sequential_9/conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:?
6sequential_9/conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8sequential_9/conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_9/conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0sequential_9/conv2d_transpose_28/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_28/stack:output:0?sequential_9/conv2d_transpose_28/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_28/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_9/conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_28_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
1sequential_9/conv2d_transpose_28/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_28/stack:output:0Hsequential_9/conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:02sequential_9/activation_87/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
7sequential_9/conv2d_transpose_28/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
(sequential_9/conv2d_transpose_28/BiasAddBiasAdd:sequential_9/conv2d_transpose_28/conv2d_transpose:output:0?sequential_9/conv2d_transpose_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
$sequential_9/activation_88/LeakyRelu	LeakyRelu1sequential_9/conv2d_transpose_28/BiasAdd:output:0*/
_output_shapes
:?????????@@ ?
&sequential_9/conv2d_transpose_29/ShapeShape2sequential_9/activation_88/LeakyRelu:activations:0*
T0*
_output_shapes
:~
4sequential_9/conv2d_transpose_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6sequential_9/conv2d_transpose_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential_9/conv2d_transpose_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.sequential_9/conv2d_transpose_29/strided_sliceStridedSlice/sequential_9/conv2d_transpose_29/Shape:output:0=sequential_9/conv2d_transpose_29/strided_slice/stack:output:0?sequential_9/conv2d_transpose_29/strided_slice/stack_1:output:0?sequential_9/conv2d_transpose_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
(sequential_9/conv2d_transpose_29/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?k
(sequential_9/conv2d_transpose_29/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?j
(sequential_9/conv2d_transpose_29/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
&sequential_9/conv2d_transpose_29/stackPack7sequential_9/conv2d_transpose_29/strided_slice:output:01sequential_9/conv2d_transpose_29/stack/1:output:01sequential_9/conv2d_transpose_29/stack/2:output:01sequential_9/conv2d_transpose_29/stack/3:output:0*
N*
T0*
_output_shapes
:?
6sequential_9/conv2d_transpose_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8sequential_9/conv2d_transpose_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_9/conv2d_transpose_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0sequential_9/conv2d_transpose_29/strided_slice_1StridedSlice/sequential_9/conv2d_transpose_29/stack:output:0?sequential_9/conv2d_transpose_29/strided_slice_1/stack:output:0Asequential_9/conv2d_transpose_29/strided_slice_1/stack_1:output:0Asequential_9/conv2d_transpose_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_9/conv2d_transpose_29/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_9_conv2d_transpose_29_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
1sequential_9/conv2d_transpose_29/conv2d_transposeConv2DBackpropInput/sequential_9/conv2d_transpose_29/stack:output:0Hsequential_9/conv2d_transpose_29/conv2d_transpose/ReadVariableOp:value:02sequential_9/activation_88/LeakyRelu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
7sequential_9/conv2d_transpose_29/BiasAdd/ReadVariableOpReadVariableOp@sequential_9_conv2d_transpose_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
(sequential_9/conv2d_transpose_29/BiasAddBiasAdd:sequential_9/conv2d_transpose_29/conv2d_transpose:output:0?sequential_9/conv2d_transpose_29/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
$sequential_9/activation_89/LeakyRelu	LeakyRelu1sequential_9/conv2d_transpose_29/BiasAdd:output:0*1
_output_shapes
:???????????|
sequential_9/decoded/ShapeShape2sequential_9/activation_89/LeakyRelu:activations:0*
T0*
_output_shapes
:r
(sequential_9/decoded/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_9/decoded/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_9/decoded/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"sequential_9/decoded/strided_sliceStridedSlice#sequential_9/decoded/Shape:output:01sequential_9/decoded/strided_slice/stack:output:03sequential_9/decoded/strided_slice/stack_1:output:03sequential_9/decoded/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
sequential_9/decoded/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?_
sequential_9/decoded/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?^
sequential_9/decoded/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
sequential_9/decoded/stackPack+sequential_9/decoded/strided_slice:output:0%sequential_9/decoded/stack/1:output:0%sequential_9/decoded/stack/2:output:0%sequential_9/decoded/stack/3:output:0*
N*
T0*
_output_shapes
:t
*sequential_9/decoded/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_9/decoded/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_9/decoded/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$sequential_9/decoded/strided_slice_1StridedSlice#sequential_9/decoded/stack:output:03sequential_9/decoded/strided_slice_1/stack:output:05sequential_9/decoded/strided_slice_1/stack_1:output:05sequential_9/decoded/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
4sequential_9/decoded/conv2d_transpose/ReadVariableOpReadVariableOp=sequential_9_decoded_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
%sequential_9/decoded/conv2d_transposeConv2DBackpropInput#sequential_9/decoded/stack:output:0<sequential_9/decoded/conv2d_transpose/ReadVariableOp:value:02sequential_9/activation_89/LeakyRelu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
+sequential_9/decoded/BiasAdd/ReadVariableOpReadVariableOp4sequential_9_decoded_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_9/decoded/BiasAddBiasAdd.sequential_9/decoded/conv2d_transpose:output:03sequential_9/decoded/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
sequential_9/decoded/TanhTanh%sequential_9/decoded/BiasAdd:output:0*
T0*1
_output_shapes
:???????????v
IdentityIdentitysequential_9/decoded/Tanh:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOpD^sequential_9/batch_normalization_63/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_63/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_63/ReadVariableOp5^sequential_9/batch_normalization_63/ReadVariableOp_1D^sequential_9/batch_normalization_64/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_64/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_64/ReadVariableOp5^sequential_9/batch_normalization_64/ReadVariableOp_1D^sequential_9/batch_normalization_65/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_65/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_65/ReadVariableOp5^sequential_9/batch_normalization_65/ReadVariableOp_1D^sequential_9/batch_normalization_66/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_66/ReadVariableOp5^sequential_9/batch_normalization_66/ReadVariableOp_1D^sequential_9/batch_normalization_67/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_67/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_67/ReadVariableOp5^sequential_9/batch_normalization_67/ReadVariableOp_1D^sequential_9/batch_normalization_68/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_68/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_68/ReadVariableOp5^sequential_9/batch_normalization_68/ReadVariableOp_1D^sequential_9/batch_normalization_69/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_69/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_69/ReadVariableOp5^sequential_9/batch_normalization_69/ReadVariableOp_1.^sequential_9/conv2d_63/BiasAdd/ReadVariableOp-^sequential_9/conv2d_63/Conv2D/ReadVariableOp.^sequential_9/conv2d_64/BiasAdd/ReadVariableOp-^sequential_9/conv2d_64/Conv2D/ReadVariableOp.^sequential_9/conv2d_65/BiasAdd/ReadVariableOp-^sequential_9/conv2d_65/Conv2D/ReadVariableOp.^sequential_9/conv2d_66/BiasAdd/ReadVariableOp-^sequential_9/conv2d_66/Conv2D/ReadVariableOp.^sequential_9/conv2d_67/BiasAdd/ReadVariableOp-^sequential_9/conv2d_67/Conv2D/ReadVariableOp.^sequential_9/conv2d_68/BiasAdd/ReadVariableOp-^sequential_9/conv2d_68/Conv2D/ReadVariableOp.^sequential_9/conv2d_69/BiasAdd/ReadVariableOp-^sequential_9/conv2d_69/Conv2D/ReadVariableOp8^sequential_9/conv2d_transpose_27/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_27/conv2d_transpose/ReadVariableOp8^sequential_9/conv2d_transpose_28/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_28/conv2d_transpose/ReadVariableOp8^sequential_9/conv2d_transpose_29/BiasAdd/ReadVariableOpA^sequential_9/conv2d_transpose_29/conv2d_transpose/ReadVariableOp,^sequential_9/decoded/BiasAdd/ReadVariableOp5^sequential_9/decoded/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2?
Csequential_9/batch_normalization_63/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_63/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_63/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_63/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_63/ReadVariableOp2sequential_9/batch_normalization_63/ReadVariableOp2l
4sequential_9/batch_normalization_63/ReadVariableOp_14sequential_9/batch_normalization_63/ReadVariableOp_12?
Csequential_9/batch_normalization_64/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_64/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_64/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_64/ReadVariableOp2sequential_9/batch_normalization_64/ReadVariableOp2l
4sequential_9/batch_normalization_64/ReadVariableOp_14sequential_9/batch_normalization_64/ReadVariableOp_12?
Csequential_9/batch_normalization_65/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_65/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_65/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_65/ReadVariableOp2sequential_9/batch_normalization_65/ReadVariableOp2l
4sequential_9/batch_normalization_65/ReadVariableOp_14sequential_9/batch_normalization_65/ReadVariableOp_12?
Csequential_9/batch_normalization_66/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_66/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_66/ReadVariableOp2sequential_9/batch_normalization_66/ReadVariableOp2l
4sequential_9/batch_normalization_66/ReadVariableOp_14sequential_9/batch_normalization_66/ReadVariableOp_12?
Csequential_9/batch_normalization_67/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_67/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_67/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_67/ReadVariableOp2sequential_9/batch_normalization_67/ReadVariableOp2l
4sequential_9/batch_normalization_67/ReadVariableOp_14sequential_9/batch_normalization_67/ReadVariableOp_12?
Csequential_9/batch_normalization_68/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_68/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_68/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_68/ReadVariableOp2sequential_9/batch_normalization_68/ReadVariableOp2l
4sequential_9/batch_normalization_68/ReadVariableOp_14sequential_9/batch_normalization_68/ReadVariableOp_12?
Csequential_9/batch_normalization_69/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_69/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_69/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_69/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_69/ReadVariableOp2sequential_9/batch_normalization_69/ReadVariableOp2l
4sequential_9/batch_normalization_69/ReadVariableOp_14sequential_9/batch_normalization_69/ReadVariableOp_12^
-sequential_9/conv2d_63/BiasAdd/ReadVariableOp-sequential_9/conv2d_63/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_63/Conv2D/ReadVariableOp,sequential_9/conv2d_63/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_64/BiasAdd/ReadVariableOp-sequential_9/conv2d_64/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_64/Conv2D/ReadVariableOp,sequential_9/conv2d_64/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_65/BiasAdd/ReadVariableOp-sequential_9/conv2d_65/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_65/Conv2D/ReadVariableOp,sequential_9/conv2d_65/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_66/BiasAdd/ReadVariableOp-sequential_9/conv2d_66/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_66/Conv2D/ReadVariableOp,sequential_9/conv2d_66/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_67/BiasAdd/ReadVariableOp-sequential_9/conv2d_67/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_67/Conv2D/ReadVariableOp,sequential_9/conv2d_67/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_68/BiasAdd/ReadVariableOp-sequential_9/conv2d_68/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_68/Conv2D/ReadVariableOp,sequential_9/conv2d_68/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_69/BiasAdd/ReadVariableOp-sequential_9/conv2d_69/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_69/Conv2D/ReadVariableOp,sequential_9/conv2d_69/Conv2D/ReadVariableOp2r
7sequential_9/conv2d_transpose_27/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_27/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_27/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_27/conv2d_transpose/ReadVariableOp2r
7sequential_9/conv2d_transpose_28/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_28/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_28/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_28/conv2d_transpose/ReadVariableOp2r
7sequential_9/conv2d_transpose_29/BiasAdd/ReadVariableOp7sequential_9/conv2d_transpose_29/BiasAdd/ReadVariableOp2?
@sequential_9/conv2d_transpose_29/conv2d_transpose/ReadVariableOp@sequential_9/conv2d_transpose_29/conv2d_transpose/ReadVariableOp2Z
+sequential_9/decoded/BiasAdd/ReadVariableOp+sequential_9/decoded/BiasAdd/ReadVariableOp2l
4sequential_9/decoded/conv2d_transpose/ReadVariableOp4sequential_9/decoded/conv2d_transpose/ReadVariableOp:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_63_input
?	
?
8__inference_batch_normalization_69_layer_call_fn_3374563

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
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_3371770?
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
5__inference_conv2d_transpose_28_layer_call_fn_3374670

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
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_3371862?
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
f
J__inference_activation_86_layer_call_and_return_conditional_losses_3372155

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
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_3374508

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
?
K
/__inference_activation_84_layer_call_fn_3374331

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
J__inference_activation_84_layer_call_and_return_conditional_losses_3372091h
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
J__inference_activation_88_layer_call_and_return_conditional_losses_3374713

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
?
`
D__inference_encoded_layer_call_and_return_conditional_losses_3372188

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
?
f
J__inference_activation_88_layer_call_and_return_conditional_losses_3372213

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
8__inference_batch_normalization_64_layer_call_fn_3374108

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
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3371450?
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
?
f
J__inference_activation_84_layer_call_and_return_conditional_losses_3372091

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
F__inference_conv2d_63_layer_call_and_return_conditional_losses_3371975

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
?

?
F__inference_conv2d_64_layer_call_and_return_conditional_losses_3374082

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
?
?
+__inference_conv2d_67_layer_call_fn_3374345

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
F__inference_conv2d_67_layer_call_and_return_conditional_losses_3372103w
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
?
K
/__inference_activation_82_layer_call_fn_3374149

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
J__inference_activation_82_layer_call_and_return_conditional_losses_3372027j
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
?
P__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_3374755

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
5__inference_conv2d_transpose_29_layer_call_fn_3374722

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
P__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_3371906?
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
J__inference_activation_85_layer_call_and_return_conditional_losses_3372123

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
8__inference_batch_normalization_67_layer_call_fn_3374368

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
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3371611?
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
?
f
J__inference_activation_82_layer_call_and_return_conditional_losses_3372027

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
F__inference_conv2d_67_layer_call_and_return_conditional_losses_3372103

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
??
?X
#__inference__traced_restore_3375615
file_prefix;
!assignvariableop_conv2d_63_kernel:/
!assignvariableop_1_conv2d_63_bias:=
/assignvariableop_2_batch_normalization_63_gamma:<
.assignvariableop_3_batch_normalization_63_beta:C
5assignvariableop_4_batch_normalization_63_moving_mean:G
9assignvariableop_5_batch_normalization_63_moving_variance:=
#assignvariableop_6_conv2d_64_kernel:/
!assignvariableop_7_conv2d_64_bias:=
/assignvariableop_8_batch_normalization_64_gamma:<
.assignvariableop_9_batch_normalization_64_beta:D
6assignvariableop_10_batch_normalization_64_moving_mean:H
:assignvariableop_11_batch_normalization_64_moving_variance:>
$assignvariableop_12_conv2d_65_kernel:0
"assignvariableop_13_conv2d_65_bias:>
0assignvariableop_14_batch_normalization_65_gamma:=
/assignvariableop_15_batch_normalization_65_beta:D
6assignvariableop_16_batch_normalization_65_moving_mean:H
:assignvariableop_17_batch_normalization_65_moving_variance:>
$assignvariableop_18_conv2d_66_kernel: 0
"assignvariableop_19_conv2d_66_bias: >
0assignvariableop_20_batch_normalization_66_gamma: =
/assignvariableop_21_batch_normalization_66_beta: D
6assignvariableop_22_batch_normalization_66_moving_mean: H
:assignvariableop_23_batch_normalization_66_moving_variance: >
$assignvariableop_24_conv2d_67_kernel:  0
"assignvariableop_25_conv2d_67_bias: >
0assignvariableop_26_batch_normalization_67_gamma: =
/assignvariableop_27_batch_normalization_67_beta: D
6assignvariableop_28_batch_normalization_67_moving_mean: H
:assignvariableop_29_batch_normalization_67_moving_variance: >
$assignvariableop_30_conv2d_68_kernel: @0
"assignvariableop_31_conv2d_68_bias:@>
0assignvariableop_32_batch_normalization_68_gamma:@=
/assignvariableop_33_batch_normalization_68_beta:@D
6assignvariableop_34_batch_normalization_68_moving_mean:@H
:assignvariableop_35_batch_normalization_68_moving_variance:@>
$assignvariableop_36_conv2d_69_kernel:@ 0
"assignvariableop_37_conv2d_69_bias: >
0assignvariableop_38_batch_normalization_69_gamma: =
/assignvariableop_39_batch_normalization_69_beta: D
6assignvariableop_40_batch_normalization_69_moving_mean: H
:assignvariableop_41_batch_normalization_69_moving_variance: H
.assignvariableop_42_conv2d_transpose_27_kernel:@ :
,assignvariableop_43_conv2d_transpose_27_bias:@H
.assignvariableop_44_conv2d_transpose_28_kernel: @:
,assignvariableop_45_conv2d_transpose_28_bias: H
.assignvariableop_46_conv2d_transpose_29_kernel: :
,assignvariableop_47_conv2d_transpose_29_bias:<
"assignvariableop_48_decoded_kernel:.
 assignvariableop_49_decoded_bias:'
assignvariableop_50_adam_iter:	 )
assignvariableop_51_adam_beta_1: )
assignvariableop_52_adam_beta_2: (
assignvariableop_53_adam_decay: 0
&assignvariableop_54_adam_learning_rate: #
assignvariableop_55_total: #
assignvariableop_56_count: E
+assignvariableop_57_adam_conv2d_63_kernel_m:7
)assignvariableop_58_adam_conv2d_63_bias_m:E
7assignvariableop_59_adam_batch_normalization_63_gamma_m:D
6assignvariableop_60_adam_batch_normalization_63_beta_m:E
+assignvariableop_61_adam_conv2d_64_kernel_m:7
)assignvariableop_62_adam_conv2d_64_bias_m:E
7assignvariableop_63_adam_batch_normalization_64_gamma_m:D
6assignvariableop_64_adam_batch_normalization_64_beta_m:E
+assignvariableop_65_adam_conv2d_65_kernel_m:7
)assignvariableop_66_adam_conv2d_65_bias_m:E
7assignvariableop_67_adam_batch_normalization_65_gamma_m:D
6assignvariableop_68_adam_batch_normalization_65_beta_m:E
+assignvariableop_69_adam_conv2d_66_kernel_m: 7
)assignvariableop_70_adam_conv2d_66_bias_m: E
7assignvariableop_71_adam_batch_normalization_66_gamma_m: D
6assignvariableop_72_adam_batch_normalization_66_beta_m: E
+assignvariableop_73_adam_conv2d_67_kernel_m:  7
)assignvariableop_74_adam_conv2d_67_bias_m: E
7assignvariableop_75_adam_batch_normalization_67_gamma_m: D
6assignvariableop_76_adam_batch_normalization_67_beta_m: E
+assignvariableop_77_adam_conv2d_68_kernel_m: @7
)assignvariableop_78_adam_conv2d_68_bias_m:@E
7assignvariableop_79_adam_batch_normalization_68_gamma_m:@D
6assignvariableop_80_adam_batch_normalization_68_beta_m:@E
+assignvariableop_81_adam_conv2d_69_kernel_m:@ 7
)assignvariableop_82_adam_conv2d_69_bias_m: E
7assignvariableop_83_adam_batch_normalization_69_gamma_m: D
6assignvariableop_84_adam_batch_normalization_69_beta_m: O
5assignvariableop_85_adam_conv2d_transpose_27_kernel_m:@ A
3assignvariableop_86_adam_conv2d_transpose_27_bias_m:@O
5assignvariableop_87_adam_conv2d_transpose_28_kernel_m: @A
3assignvariableop_88_adam_conv2d_transpose_28_bias_m: O
5assignvariableop_89_adam_conv2d_transpose_29_kernel_m: A
3assignvariableop_90_adam_conv2d_transpose_29_bias_m:C
)assignvariableop_91_adam_decoded_kernel_m:5
'assignvariableop_92_adam_decoded_bias_m:E
+assignvariableop_93_adam_conv2d_63_kernel_v:7
)assignvariableop_94_adam_conv2d_63_bias_v:E
7assignvariableop_95_adam_batch_normalization_63_gamma_v:D
6assignvariableop_96_adam_batch_normalization_63_beta_v:E
+assignvariableop_97_adam_conv2d_64_kernel_v:7
)assignvariableop_98_adam_conv2d_64_bias_v:E
7assignvariableop_99_adam_batch_normalization_64_gamma_v:E
7assignvariableop_100_adam_batch_normalization_64_beta_v:F
,assignvariableop_101_adam_conv2d_65_kernel_v:8
*assignvariableop_102_adam_conv2d_65_bias_v:F
8assignvariableop_103_adam_batch_normalization_65_gamma_v:E
7assignvariableop_104_adam_batch_normalization_65_beta_v:F
,assignvariableop_105_adam_conv2d_66_kernel_v: 8
*assignvariableop_106_adam_conv2d_66_bias_v: F
8assignvariableop_107_adam_batch_normalization_66_gamma_v: E
7assignvariableop_108_adam_batch_normalization_66_beta_v: F
,assignvariableop_109_adam_conv2d_67_kernel_v:  8
*assignvariableop_110_adam_conv2d_67_bias_v: F
8assignvariableop_111_adam_batch_normalization_67_gamma_v: E
7assignvariableop_112_adam_batch_normalization_67_beta_v: F
,assignvariableop_113_adam_conv2d_68_kernel_v: @8
*assignvariableop_114_adam_conv2d_68_bias_v:@F
8assignvariableop_115_adam_batch_normalization_68_gamma_v:@E
7assignvariableop_116_adam_batch_normalization_68_beta_v:@F
,assignvariableop_117_adam_conv2d_69_kernel_v:@ 8
*assignvariableop_118_adam_conv2d_69_bias_v: F
8assignvariableop_119_adam_batch_normalization_69_gamma_v: E
7assignvariableop_120_adam_batch_normalization_69_beta_v: P
6assignvariableop_121_adam_conv2d_transpose_27_kernel_v:@ B
4assignvariableop_122_adam_conv2d_transpose_27_bias_v:@P
6assignvariableop_123_adam_conv2d_transpose_28_kernel_v: @B
4assignvariableop_124_adam_conv2d_transpose_28_bias_v: P
6assignvariableop_125_adam_conv2d_transpose_29_kernel_v: B
4assignvariableop_126_adam_conv2d_transpose_29_bias_v:D
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
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_63_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_63_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_63_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_63_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_63_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_63_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_64_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_64_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_64_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_64_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_64_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_64_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_65_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_65_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_65_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_65_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_65_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_65_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_66_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_66_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_66_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_66_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_66_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_66_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_67_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_67_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_normalization_67_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_67_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp6assignvariableop_28_batch_normalization_67_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_normalization_67_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv2d_68_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv2d_68_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_68_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_68_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_68_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_68_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp$assignvariableop_36_conv2d_69_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp"assignvariableop_37_conv2d_69_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp0assignvariableop_38_batch_normalization_69_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp/assignvariableop_39_batch_normalization_69_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp6assignvariableop_40_batch_normalization_69_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp:assignvariableop_41_batch_normalization_69_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp.assignvariableop_42_conv2d_transpose_27_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp,assignvariableop_43_conv2d_transpose_27_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp.assignvariableop_44_conv2d_transpose_28_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp,assignvariableop_45_conv2d_transpose_28_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp.assignvariableop_46_conv2d_transpose_29_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp,assignvariableop_47_conv2d_transpose_29_biasIdentity_47:output:0"/device:CPU:0*
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
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv2d_63_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv2d_63_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp7assignvariableop_59_adam_batch_normalization_63_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adam_batch_normalization_63_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_conv2d_64_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_conv2d_64_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp7assignvariableop_63_adam_batch_normalization_64_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_batch_normalization_64_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv2d_65_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv2d_65_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp7assignvariableop_67_adam_batch_normalization_65_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adam_batch_normalization_65_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv2d_66_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv2d_66_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp7assignvariableop_71_adam_batch_normalization_66_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adam_batch_normalization_66_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_conv2d_67_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_conv2d_67_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp7assignvariableop_75_adam_batch_normalization_67_gamma_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOp6assignvariableop_76_adam_batch_normalization_67_beta_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_conv2d_68_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_conv2d_68_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp7assignvariableop_79_adam_batch_normalization_68_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOp6assignvariableop_80_adam_batch_normalization_68_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOp+assignvariableop_81_adam_conv2d_69_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_conv2d_69_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOp7assignvariableop_83_adam_batch_normalization_69_gamma_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOp6assignvariableop_84_adam_batch_normalization_69_beta_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOp5assignvariableop_85_adam_conv2d_transpose_27_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOp3assignvariableop_86_adam_conv2d_transpose_27_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_87AssignVariableOp5assignvariableop_87_adam_conv2d_transpose_28_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_88AssignVariableOp3assignvariableop_88_adam_conv2d_transpose_28_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_89AssignVariableOp5assignvariableop_89_adam_conv2d_transpose_29_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_90AssignVariableOp3assignvariableop_90_adam_conv2d_transpose_29_bias_mIdentity_90:output:0"/device:CPU:0*
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
AssignVariableOp_93AssignVariableOp+assignvariableop_93_adam_conv2d_63_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_94AssignVariableOp)assignvariableop_94_adam_conv2d_63_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_95AssignVariableOp7assignvariableop_95_adam_batch_normalization_63_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_96AssignVariableOp6assignvariableop_96_adam_batch_normalization_63_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_97AssignVariableOp+assignvariableop_97_adam_conv2d_64_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_98AssignVariableOp)assignvariableop_98_adam_conv2d_64_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_99AssignVariableOp7assignvariableop_99_adam_batch_normalization_64_gamma_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_100AssignVariableOp7assignvariableop_100_adam_batch_normalization_64_beta_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_101AssignVariableOp,assignvariableop_101_adam_conv2d_65_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_102AssignVariableOp*assignvariableop_102_adam_conv2d_65_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_103AssignVariableOp8assignvariableop_103_adam_batch_normalization_65_gamma_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_104AssignVariableOp7assignvariableop_104_adam_batch_normalization_65_beta_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_105AssignVariableOp,assignvariableop_105_adam_conv2d_66_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_106AssignVariableOp*assignvariableop_106_adam_conv2d_66_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_107AssignVariableOp8assignvariableop_107_adam_batch_normalization_66_gamma_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_108AssignVariableOp7assignvariableop_108_adam_batch_normalization_66_beta_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_109AssignVariableOp,assignvariableop_109_adam_conv2d_67_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_110AssignVariableOp*assignvariableop_110_adam_conv2d_67_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_111AssignVariableOp8assignvariableop_111_adam_batch_normalization_67_gamma_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_112AssignVariableOp7assignvariableop_112_adam_batch_normalization_67_beta_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_113AssignVariableOp,assignvariableop_113_adam_conv2d_68_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_114AssignVariableOp*assignvariableop_114_adam_conv2d_68_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_115AssignVariableOp8assignvariableop_115_adam_batch_normalization_68_gamma_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_116AssignVariableOp7assignvariableop_116_adam_batch_normalization_68_beta_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_117AssignVariableOp,assignvariableop_117_adam_conv2d_69_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_118AssignVariableOp*assignvariableop_118_adam_conv2d_69_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_119AssignVariableOp8assignvariableop_119_adam_batch_normalization_69_gamma_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_120AssignVariableOp7assignvariableop_120_adam_batch_normalization_69_beta_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_121AssignVariableOp6assignvariableop_121_adam_conv2d_transpose_27_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_122AssignVariableOp4assignvariableop_122_adam_conv2d_transpose_27_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_123AssignVariableOp6assignvariableop_123_adam_conv2d_transpose_28_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_124AssignVariableOp4assignvariableop_124_adam_conv2d_transpose_28_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_125AssignVariableOp6assignvariableop_125_adam_conv2d_transpose_29_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_126AssignVariableOp4assignvariableop_126_adam_conv2d_transpose_29_bias_vIdentity_126:output:0"/device:CPU:0*
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
?
`
D__inference_encoded_layer_call_and_return_conditional_losses_3374609

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
/__inference_activation_86_layer_call_fn_3374513

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
J__inference_activation_86_layer_call_and_return_conditional_losses_3372155h
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
?
?
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_3374581

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
?
K
/__inference_activation_81_layer_call_fn_3374058

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
J__inference_activation_81_layer_call_and_return_conditional_losses_3371995j
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
J__inference_activation_87_layer_call_and_return_conditional_losses_3372201

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
?
?
+__inference_conv2d_65_layer_call_fn_3374163

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
F__inference_conv2d_65_layer_call_and_return_conditional_losses_3372039y
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
?
?
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3371419

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
?

?
F__inference_conv2d_68_layer_call_and_return_conditional_losses_3374446

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
?
f
J__inference_activation_83_layer_call_and_return_conditional_losses_3374245

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
?
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_3374703

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
??
?
I__inference_sequential_9_layer_call_and_return_conditional_losses_3373049
conv2d_63_input+
conv2d_63_3372918:
conv2d_63_3372920:,
batch_normalization_63_3372923:,
batch_normalization_63_3372925:,
batch_normalization_63_3372927:,
batch_normalization_63_3372929:+
conv2d_64_3372933:
conv2d_64_3372935:,
batch_normalization_64_3372938:,
batch_normalization_64_3372940:,
batch_normalization_64_3372942:,
batch_normalization_64_3372944:+
conv2d_65_3372948:
conv2d_65_3372950:,
batch_normalization_65_3372953:,
batch_normalization_65_3372955:,
batch_normalization_65_3372957:,
batch_normalization_65_3372959:+
conv2d_66_3372963: 
conv2d_66_3372965: ,
batch_normalization_66_3372968: ,
batch_normalization_66_3372970: ,
batch_normalization_66_3372972: ,
batch_normalization_66_3372974: +
conv2d_67_3372978:  
conv2d_67_3372980: ,
batch_normalization_67_3372983: ,
batch_normalization_67_3372985: ,
batch_normalization_67_3372987: ,
batch_normalization_67_3372989: +
conv2d_68_3372993: @
conv2d_68_3372995:@,
batch_normalization_68_3372998:@,
batch_normalization_68_3373000:@,
batch_normalization_68_3373002:@,
batch_normalization_68_3373004:@+
conv2d_69_3373008:@ 
conv2d_69_3373010: ,
batch_normalization_69_3373013: ,
batch_normalization_69_3373015: ,
batch_normalization_69_3373017: ,
batch_normalization_69_3373019: 5
conv2d_transpose_27_3373025:@ )
conv2d_transpose_27_3373027:@5
conv2d_transpose_28_3373031: @)
conv2d_transpose_28_3373033: 5
conv2d_transpose_29_3373037: )
conv2d_transpose_29_3373039:)
decoded_3373043:
decoded_3373045:
identity??.batch_normalization_63/StatefulPartitionedCall?.batch_normalization_64/StatefulPartitionedCall?.batch_normalization_65/StatefulPartitionedCall?.batch_normalization_66/StatefulPartitionedCall?.batch_normalization_67/StatefulPartitionedCall?.batch_normalization_68/StatefulPartitionedCall?.batch_normalization_69/StatefulPartitionedCall?!conv2d_63/StatefulPartitionedCall?!conv2d_64/StatefulPartitionedCall?!conv2d_65/StatefulPartitionedCall?!conv2d_66/StatefulPartitionedCall?!conv2d_67/StatefulPartitionedCall?!conv2d_68/StatefulPartitionedCall?!conv2d_69/StatefulPartitionedCall?+conv2d_transpose_27/StatefulPartitionedCall?+conv2d_transpose_28/StatefulPartitionedCall?+conv2d_transpose_29/StatefulPartitionedCall?decoded/StatefulPartitionedCall?
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCallconv2d_63_inputconv2d_63_3372918conv2d_63_3372920*
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
F__inference_conv2d_63_layer_call_and_return_conditional_losses_3371975?
.batch_normalization_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0batch_normalization_63_3372923batch_normalization_63_3372925batch_normalization_63_3372927batch_normalization_63_3372929*
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
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3371355?
activation_81/PartitionedCallPartitionedCall7batch_normalization_63/StatefulPartitionedCall:output:0*
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
J__inference_activation_81_layer_call_and_return_conditional_losses_3371995?
!conv2d_64/StatefulPartitionedCallStatefulPartitionedCall&activation_81/PartitionedCall:output:0conv2d_64_3372933conv2d_64_3372935*
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
F__inference_conv2d_64_layer_call_and_return_conditional_losses_3372007?
.batch_normalization_64/StatefulPartitionedCallStatefulPartitionedCall*conv2d_64/StatefulPartitionedCall:output:0batch_normalization_64_3372938batch_normalization_64_3372940batch_normalization_64_3372942batch_normalization_64_3372944*
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
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3371419?
activation_82/PartitionedCallPartitionedCall7batch_normalization_64/StatefulPartitionedCall:output:0*
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
J__inference_activation_82_layer_call_and_return_conditional_losses_3372027?
!conv2d_65/StatefulPartitionedCallStatefulPartitionedCall&activation_82/PartitionedCall:output:0conv2d_65_3372948conv2d_65_3372950*
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
F__inference_conv2d_65_layer_call_and_return_conditional_losses_3372039?
.batch_normalization_65/StatefulPartitionedCallStatefulPartitionedCall*conv2d_65/StatefulPartitionedCall:output:0batch_normalization_65_3372953batch_normalization_65_3372955batch_normalization_65_3372957batch_normalization_65_3372959*
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
S__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3371483?
activation_83/PartitionedCallPartitionedCall7batch_normalization_65/StatefulPartitionedCall:output:0*
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
J__inference_activation_83_layer_call_and_return_conditional_losses_3372059?
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCall&activation_83/PartitionedCall:output:0conv2d_66_3372963conv2d_66_3372965*
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
F__inference_conv2d_66_layer_call_and_return_conditional_losses_3372071?
.batch_normalization_66/StatefulPartitionedCallStatefulPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0batch_normalization_66_3372968batch_normalization_66_3372970batch_normalization_66_3372972batch_normalization_66_3372974*
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
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3371547?
activation_84/PartitionedCallPartitionedCall7batch_normalization_66/StatefulPartitionedCall:output:0*
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
J__inference_activation_84_layer_call_and_return_conditional_losses_3372091?
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall&activation_84/PartitionedCall:output:0conv2d_67_3372978conv2d_67_3372980*
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
F__inference_conv2d_67_layer_call_and_return_conditional_losses_3372103?
.batch_normalization_67/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0batch_normalization_67_3372983batch_normalization_67_3372985batch_normalization_67_3372987batch_normalization_67_3372989*
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
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3371611?
activation_85/PartitionedCallPartitionedCall7batch_normalization_67/StatefulPartitionedCall:output:0*
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
J__inference_activation_85_layer_call_and_return_conditional_losses_3372123?
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall&activation_85/PartitionedCall:output:0conv2d_68_3372993conv2d_68_3372995*
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
F__inference_conv2d_68_layer_call_and_return_conditional_losses_3372135?
.batch_normalization_68/StatefulPartitionedCallStatefulPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0batch_normalization_68_3372998batch_normalization_68_3373000batch_normalization_68_3373002batch_normalization_68_3373004*
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
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_3371675?
activation_86/PartitionedCallPartitionedCall7batch_normalization_68/StatefulPartitionedCall:output:0*
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
J__inference_activation_86_layer_call_and_return_conditional_losses_3372155?
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCall&activation_86/PartitionedCall:output:0conv2d_69_3373008conv2d_69_3373010*
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
F__inference_conv2d_69_layer_call_and_return_conditional_losses_3372167?
.batch_normalization_69/StatefulPartitionedCallStatefulPartitionedCall*conv2d_69/StatefulPartitionedCall:output:0batch_normalization_69_3373013batch_normalization_69_3373015batch_normalization_69_3373017batch_normalization_69_3373019*
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
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_3371739?
encoded/CastCast7batch_normalization_69/StatefulPartitionedCall:output:0*

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
D__inference_encoded_layer_call_and_return_conditional_losses_3372188?
conv2d_transpose_27/CastCast encoded/PartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_27/Cast:y:0conv2d_transpose_27_3373025conv2d_transpose_27_3373027*
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
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_3371818?
activation_87/PartitionedCallPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0*
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
J__inference_activation_87_layer_call_and_return_conditional_losses_3372201?
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall&activation_87/PartitionedCall:output:0conv2d_transpose_28_3373031conv2d_transpose_28_3373033*
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
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_3371862?
activation_88/PartitionedCallPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0*
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
J__inference_activation_88_layer_call_and_return_conditional_losses_3372213?
+conv2d_transpose_29/StatefulPartitionedCallStatefulPartitionedCall&activation_88/PartitionedCall:output:0conv2d_transpose_29_3373037conv2d_transpose_29_3373039*
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
P__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_3371906?
activation_89/PartitionedCallPartitionedCall4conv2d_transpose_29/StatefulPartitionedCall:output:0*
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
J__inference_activation_89_layer_call_and_return_conditional_losses_3372225?
decoded/StatefulPartitionedCallStatefulPartitionedCall&activation_89/PartitionedCall:output:0decoded_3373043decoded_3373045*
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
D__inference_decoded_layer_call_and_return_conditional_losses_3371951?
IdentityIdentity(decoded/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp/^batch_normalization_63/StatefulPartitionedCall/^batch_normalization_64/StatefulPartitionedCall/^batch_normalization_65/StatefulPartitionedCall/^batch_normalization_66/StatefulPartitionedCall/^batch_normalization_67/StatefulPartitionedCall/^batch_normalization_68/StatefulPartitionedCall/^batch_normalization_69/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall"^conv2d_64/StatefulPartitionedCall"^conv2d_65/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall,^conv2d_transpose_29/StatefulPartitionedCall ^decoded/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_63/StatefulPartitionedCall.batch_normalization_63/StatefulPartitionedCall2`
.batch_normalization_64/StatefulPartitionedCall.batch_normalization_64/StatefulPartitionedCall2`
.batch_normalization_65/StatefulPartitionedCall.batch_normalization_65/StatefulPartitionedCall2`
.batch_normalization_66/StatefulPartitionedCall.batch_normalization_66/StatefulPartitionedCall2`
.batch_normalization_67/StatefulPartitionedCall.batch_normalization_67/StatefulPartitionedCall2`
.batch_normalization_68/StatefulPartitionedCall.batch_normalization_68/StatefulPartitionedCall2`
.batch_normalization_69/StatefulPartitionedCall.batch_normalization_69/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2F
!conv2d_64/StatefulPartitionedCall!conv2d_64/StatefulPartitionedCall2F
!conv2d_65/StatefulPartitionedCall!conv2d_65/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall2Z
+conv2d_transpose_29/StatefulPartitionedCall+conv2d_transpose_29/StatefulPartitionedCall2B
decoded/StatefulPartitionedCalldecoded/StatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_63_input
??
?
I__inference_sequential_9_layer_call_and_return_conditional_losses_3372707

inputs+
conv2d_63_3372576:
conv2d_63_3372578:,
batch_normalization_63_3372581:,
batch_normalization_63_3372583:,
batch_normalization_63_3372585:,
batch_normalization_63_3372587:+
conv2d_64_3372591:
conv2d_64_3372593:,
batch_normalization_64_3372596:,
batch_normalization_64_3372598:,
batch_normalization_64_3372600:,
batch_normalization_64_3372602:+
conv2d_65_3372606:
conv2d_65_3372608:,
batch_normalization_65_3372611:,
batch_normalization_65_3372613:,
batch_normalization_65_3372615:,
batch_normalization_65_3372617:+
conv2d_66_3372621: 
conv2d_66_3372623: ,
batch_normalization_66_3372626: ,
batch_normalization_66_3372628: ,
batch_normalization_66_3372630: ,
batch_normalization_66_3372632: +
conv2d_67_3372636:  
conv2d_67_3372638: ,
batch_normalization_67_3372641: ,
batch_normalization_67_3372643: ,
batch_normalization_67_3372645: ,
batch_normalization_67_3372647: +
conv2d_68_3372651: @
conv2d_68_3372653:@,
batch_normalization_68_3372656:@,
batch_normalization_68_3372658:@,
batch_normalization_68_3372660:@,
batch_normalization_68_3372662:@+
conv2d_69_3372666:@ 
conv2d_69_3372668: ,
batch_normalization_69_3372671: ,
batch_normalization_69_3372673: ,
batch_normalization_69_3372675: ,
batch_normalization_69_3372677: 5
conv2d_transpose_27_3372683:@ )
conv2d_transpose_27_3372685:@5
conv2d_transpose_28_3372689: @)
conv2d_transpose_28_3372691: 5
conv2d_transpose_29_3372695: )
conv2d_transpose_29_3372697:)
decoded_3372701:
decoded_3372703:
identity??.batch_normalization_63/StatefulPartitionedCall?.batch_normalization_64/StatefulPartitionedCall?.batch_normalization_65/StatefulPartitionedCall?.batch_normalization_66/StatefulPartitionedCall?.batch_normalization_67/StatefulPartitionedCall?.batch_normalization_68/StatefulPartitionedCall?.batch_normalization_69/StatefulPartitionedCall?!conv2d_63/StatefulPartitionedCall?!conv2d_64/StatefulPartitionedCall?!conv2d_65/StatefulPartitionedCall?!conv2d_66/StatefulPartitionedCall?!conv2d_67/StatefulPartitionedCall?!conv2d_68/StatefulPartitionedCall?!conv2d_69/StatefulPartitionedCall?+conv2d_transpose_27/StatefulPartitionedCall?+conv2d_transpose_28/StatefulPartitionedCall?+conv2d_transpose_29/StatefulPartitionedCall?decoded/StatefulPartitionedCall?
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_63_3372576conv2d_63_3372578*
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
F__inference_conv2d_63_layer_call_and_return_conditional_losses_3371975?
.batch_normalization_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0batch_normalization_63_3372581batch_normalization_63_3372583batch_normalization_63_3372585batch_normalization_63_3372587*
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
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3371386?
activation_81/PartitionedCallPartitionedCall7batch_normalization_63/StatefulPartitionedCall:output:0*
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
J__inference_activation_81_layer_call_and_return_conditional_losses_3371995?
!conv2d_64/StatefulPartitionedCallStatefulPartitionedCall&activation_81/PartitionedCall:output:0conv2d_64_3372591conv2d_64_3372593*
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
F__inference_conv2d_64_layer_call_and_return_conditional_losses_3372007?
.batch_normalization_64/StatefulPartitionedCallStatefulPartitionedCall*conv2d_64/StatefulPartitionedCall:output:0batch_normalization_64_3372596batch_normalization_64_3372598batch_normalization_64_3372600batch_normalization_64_3372602*
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
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3371450?
activation_82/PartitionedCallPartitionedCall7batch_normalization_64/StatefulPartitionedCall:output:0*
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
J__inference_activation_82_layer_call_and_return_conditional_losses_3372027?
!conv2d_65/StatefulPartitionedCallStatefulPartitionedCall&activation_82/PartitionedCall:output:0conv2d_65_3372606conv2d_65_3372608*
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
F__inference_conv2d_65_layer_call_and_return_conditional_losses_3372039?
.batch_normalization_65/StatefulPartitionedCallStatefulPartitionedCall*conv2d_65/StatefulPartitionedCall:output:0batch_normalization_65_3372611batch_normalization_65_3372613batch_normalization_65_3372615batch_normalization_65_3372617*
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
S__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3371514?
activation_83/PartitionedCallPartitionedCall7batch_normalization_65/StatefulPartitionedCall:output:0*
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
J__inference_activation_83_layer_call_and_return_conditional_losses_3372059?
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCall&activation_83/PartitionedCall:output:0conv2d_66_3372621conv2d_66_3372623*
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
F__inference_conv2d_66_layer_call_and_return_conditional_losses_3372071?
.batch_normalization_66/StatefulPartitionedCallStatefulPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0batch_normalization_66_3372626batch_normalization_66_3372628batch_normalization_66_3372630batch_normalization_66_3372632*
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
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3371578?
activation_84/PartitionedCallPartitionedCall7batch_normalization_66/StatefulPartitionedCall:output:0*
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
J__inference_activation_84_layer_call_and_return_conditional_losses_3372091?
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall&activation_84/PartitionedCall:output:0conv2d_67_3372636conv2d_67_3372638*
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
F__inference_conv2d_67_layer_call_and_return_conditional_losses_3372103?
.batch_normalization_67/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0batch_normalization_67_3372641batch_normalization_67_3372643batch_normalization_67_3372645batch_normalization_67_3372647*
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
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3371642?
activation_85/PartitionedCallPartitionedCall7batch_normalization_67/StatefulPartitionedCall:output:0*
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
J__inference_activation_85_layer_call_and_return_conditional_losses_3372123?
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall&activation_85/PartitionedCall:output:0conv2d_68_3372651conv2d_68_3372653*
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
F__inference_conv2d_68_layer_call_and_return_conditional_losses_3372135?
.batch_normalization_68/StatefulPartitionedCallStatefulPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0batch_normalization_68_3372656batch_normalization_68_3372658batch_normalization_68_3372660batch_normalization_68_3372662*
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
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_3371706?
activation_86/PartitionedCallPartitionedCall7batch_normalization_68/StatefulPartitionedCall:output:0*
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
J__inference_activation_86_layer_call_and_return_conditional_losses_3372155?
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCall&activation_86/PartitionedCall:output:0conv2d_69_3372666conv2d_69_3372668*
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
F__inference_conv2d_69_layer_call_and_return_conditional_losses_3372167?
.batch_normalization_69/StatefulPartitionedCallStatefulPartitionedCall*conv2d_69/StatefulPartitionedCall:output:0batch_normalization_69_3372671batch_normalization_69_3372673batch_normalization_69_3372675batch_normalization_69_3372677*
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
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_3371770?
encoded/CastCast7batch_normalization_69/StatefulPartitionedCall:output:0*

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
D__inference_encoded_layer_call_and_return_conditional_losses_3372188?
conv2d_transpose_27/CastCast encoded/PartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_27/Cast:y:0conv2d_transpose_27_3372683conv2d_transpose_27_3372685*
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
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_3371818?
activation_87/PartitionedCallPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0*
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
J__inference_activation_87_layer_call_and_return_conditional_losses_3372201?
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall&activation_87/PartitionedCall:output:0conv2d_transpose_28_3372689conv2d_transpose_28_3372691*
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
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_3371862?
activation_88/PartitionedCallPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0*
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
J__inference_activation_88_layer_call_and_return_conditional_losses_3372213?
+conv2d_transpose_29/StatefulPartitionedCallStatefulPartitionedCall&activation_88/PartitionedCall:output:0conv2d_transpose_29_3372695conv2d_transpose_29_3372697*
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
P__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_3371906?
activation_89/PartitionedCallPartitionedCall4conv2d_transpose_29/StatefulPartitionedCall:output:0*
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
J__inference_activation_89_layer_call_and_return_conditional_losses_3372225?
decoded/StatefulPartitionedCallStatefulPartitionedCall&activation_89/PartitionedCall:output:0decoded_3372701decoded_3372703*
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
D__inference_decoded_layer_call_and_return_conditional_losses_3371951?
IdentityIdentity(decoded/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp/^batch_normalization_63/StatefulPartitionedCall/^batch_normalization_64/StatefulPartitionedCall/^batch_normalization_65/StatefulPartitionedCall/^batch_normalization_66/StatefulPartitionedCall/^batch_normalization_67/StatefulPartitionedCall/^batch_normalization_68/StatefulPartitionedCall/^batch_normalization_69/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall"^conv2d_64/StatefulPartitionedCall"^conv2d_65/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall,^conv2d_transpose_29/StatefulPartitionedCall ^decoded/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_63/StatefulPartitionedCall.batch_normalization_63/StatefulPartitionedCall2`
.batch_normalization_64/StatefulPartitionedCall.batch_normalization_64/StatefulPartitionedCall2`
.batch_normalization_65/StatefulPartitionedCall.batch_normalization_65/StatefulPartitionedCall2`
.batch_normalization_66/StatefulPartitionedCall.batch_normalization_66/StatefulPartitionedCall2`
.batch_normalization_67/StatefulPartitionedCall.batch_normalization_67/StatefulPartitionedCall2`
.batch_normalization_68/StatefulPartitionedCall.batch_normalization_68/StatefulPartitionedCall2`
.batch_normalization_69/StatefulPartitionedCall.batch_normalization_69/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2F
!conv2d_64/StatefulPartitionedCall!conv2d_64/StatefulPartitionedCall2F
!conv2d_65/StatefulPartitionedCall!conv2d_65/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall2Z
+conv2d_transpose_29/StatefulPartitionedCall+conv2d_transpose_29/StatefulPartitionedCall2B
decoded/StatefulPartitionedCalldecoded/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
f
J__inference_activation_81_layer_call_and_return_conditional_losses_3371995

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
?
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_3374651

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
?
?
5__inference_conv2d_transpose_27_layer_call_fn_3374618

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
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_3371818?
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
?
?
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3371578

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
??
?
I__inference_sequential_9_layer_call_and_return_conditional_losses_3372233

inputs+
conv2d_63_3371976:
conv2d_63_3371978:,
batch_normalization_63_3371981:,
batch_normalization_63_3371983:,
batch_normalization_63_3371985:,
batch_normalization_63_3371987:+
conv2d_64_3372008:
conv2d_64_3372010:,
batch_normalization_64_3372013:,
batch_normalization_64_3372015:,
batch_normalization_64_3372017:,
batch_normalization_64_3372019:+
conv2d_65_3372040:
conv2d_65_3372042:,
batch_normalization_65_3372045:,
batch_normalization_65_3372047:,
batch_normalization_65_3372049:,
batch_normalization_65_3372051:+
conv2d_66_3372072: 
conv2d_66_3372074: ,
batch_normalization_66_3372077: ,
batch_normalization_66_3372079: ,
batch_normalization_66_3372081: ,
batch_normalization_66_3372083: +
conv2d_67_3372104:  
conv2d_67_3372106: ,
batch_normalization_67_3372109: ,
batch_normalization_67_3372111: ,
batch_normalization_67_3372113: ,
batch_normalization_67_3372115: +
conv2d_68_3372136: @
conv2d_68_3372138:@,
batch_normalization_68_3372141:@,
batch_normalization_68_3372143:@,
batch_normalization_68_3372145:@,
batch_normalization_68_3372147:@+
conv2d_69_3372168:@ 
conv2d_69_3372170: ,
batch_normalization_69_3372173: ,
batch_normalization_69_3372175: ,
batch_normalization_69_3372177: ,
batch_normalization_69_3372179: 5
conv2d_transpose_27_3372191:@ )
conv2d_transpose_27_3372193:@5
conv2d_transpose_28_3372203: @)
conv2d_transpose_28_3372205: 5
conv2d_transpose_29_3372215: )
conv2d_transpose_29_3372217:)
decoded_3372227:
decoded_3372229:
identity??.batch_normalization_63/StatefulPartitionedCall?.batch_normalization_64/StatefulPartitionedCall?.batch_normalization_65/StatefulPartitionedCall?.batch_normalization_66/StatefulPartitionedCall?.batch_normalization_67/StatefulPartitionedCall?.batch_normalization_68/StatefulPartitionedCall?.batch_normalization_69/StatefulPartitionedCall?!conv2d_63/StatefulPartitionedCall?!conv2d_64/StatefulPartitionedCall?!conv2d_65/StatefulPartitionedCall?!conv2d_66/StatefulPartitionedCall?!conv2d_67/StatefulPartitionedCall?!conv2d_68/StatefulPartitionedCall?!conv2d_69/StatefulPartitionedCall?+conv2d_transpose_27/StatefulPartitionedCall?+conv2d_transpose_28/StatefulPartitionedCall?+conv2d_transpose_29/StatefulPartitionedCall?decoded/StatefulPartitionedCall?
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_63_3371976conv2d_63_3371978*
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
F__inference_conv2d_63_layer_call_and_return_conditional_losses_3371975?
.batch_normalization_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0batch_normalization_63_3371981batch_normalization_63_3371983batch_normalization_63_3371985batch_normalization_63_3371987*
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
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3371355?
activation_81/PartitionedCallPartitionedCall7batch_normalization_63/StatefulPartitionedCall:output:0*
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
J__inference_activation_81_layer_call_and_return_conditional_losses_3371995?
!conv2d_64/StatefulPartitionedCallStatefulPartitionedCall&activation_81/PartitionedCall:output:0conv2d_64_3372008conv2d_64_3372010*
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
F__inference_conv2d_64_layer_call_and_return_conditional_losses_3372007?
.batch_normalization_64/StatefulPartitionedCallStatefulPartitionedCall*conv2d_64/StatefulPartitionedCall:output:0batch_normalization_64_3372013batch_normalization_64_3372015batch_normalization_64_3372017batch_normalization_64_3372019*
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
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3371419?
activation_82/PartitionedCallPartitionedCall7batch_normalization_64/StatefulPartitionedCall:output:0*
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
J__inference_activation_82_layer_call_and_return_conditional_losses_3372027?
!conv2d_65/StatefulPartitionedCallStatefulPartitionedCall&activation_82/PartitionedCall:output:0conv2d_65_3372040conv2d_65_3372042*
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
F__inference_conv2d_65_layer_call_and_return_conditional_losses_3372039?
.batch_normalization_65/StatefulPartitionedCallStatefulPartitionedCall*conv2d_65/StatefulPartitionedCall:output:0batch_normalization_65_3372045batch_normalization_65_3372047batch_normalization_65_3372049batch_normalization_65_3372051*
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
S__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3371483?
activation_83/PartitionedCallPartitionedCall7batch_normalization_65/StatefulPartitionedCall:output:0*
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
J__inference_activation_83_layer_call_and_return_conditional_losses_3372059?
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCall&activation_83/PartitionedCall:output:0conv2d_66_3372072conv2d_66_3372074*
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
F__inference_conv2d_66_layer_call_and_return_conditional_losses_3372071?
.batch_normalization_66/StatefulPartitionedCallStatefulPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0batch_normalization_66_3372077batch_normalization_66_3372079batch_normalization_66_3372081batch_normalization_66_3372083*
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
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3371547?
activation_84/PartitionedCallPartitionedCall7batch_normalization_66/StatefulPartitionedCall:output:0*
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
J__inference_activation_84_layer_call_and_return_conditional_losses_3372091?
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall&activation_84/PartitionedCall:output:0conv2d_67_3372104conv2d_67_3372106*
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
F__inference_conv2d_67_layer_call_and_return_conditional_losses_3372103?
.batch_normalization_67/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0batch_normalization_67_3372109batch_normalization_67_3372111batch_normalization_67_3372113batch_normalization_67_3372115*
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
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3371611?
activation_85/PartitionedCallPartitionedCall7batch_normalization_67/StatefulPartitionedCall:output:0*
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
J__inference_activation_85_layer_call_and_return_conditional_losses_3372123?
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall&activation_85/PartitionedCall:output:0conv2d_68_3372136conv2d_68_3372138*
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
F__inference_conv2d_68_layer_call_and_return_conditional_losses_3372135?
.batch_normalization_68/StatefulPartitionedCallStatefulPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0batch_normalization_68_3372141batch_normalization_68_3372143batch_normalization_68_3372145batch_normalization_68_3372147*
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
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_3371675?
activation_86/PartitionedCallPartitionedCall7batch_normalization_68/StatefulPartitionedCall:output:0*
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
J__inference_activation_86_layer_call_and_return_conditional_losses_3372155?
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCall&activation_86/PartitionedCall:output:0conv2d_69_3372168conv2d_69_3372170*
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
F__inference_conv2d_69_layer_call_and_return_conditional_losses_3372167?
.batch_normalization_69/StatefulPartitionedCallStatefulPartitionedCall*conv2d_69/StatefulPartitionedCall:output:0batch_normalization_69_3372173batch_normalization_69_3372175batch_normalization_69_3372177batch_normalization_69_3372179*
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
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_3371739?
encoded/CastCast7batch_normalization_69/StatefulPartitionedCall:output:0*

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
D__inference_encoded_layer_call_and_return_conditional_losses_3372188?
conv2d_transpose_27/CastCast encoded/PartitionedCall:output:0*

DstT0*

SrcT0*/
_output_shapes
:????????? ?
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_27/Cast:y:0conv2d_transpose_27_3372191conv2d_transpose_27_3372193*
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
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_3371818?
activation_87/PartitionedCallPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0*
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
J__inference_activation_87_layer_call_and_return_conditional_losses_3372201?
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall&activation_87/PartitionedCall:output:0conv2d_transpose_28_3372203conv2d_transpose_28_3372205*
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
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_3371862?
activation_88/PartitionedCallPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0*
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
J__inference_activation_88_layer_call_and_return_conditional_losses_3372213?
+conv2d_transpose_29/StatefulPartitionedCallStatefulPartitionedCall&activation_88/PartitionedCall:output:0conv2d_transpose_29_3372215conv2d_transpose_29_3372217*
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
P__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_3371906?
activation_89/PartitionedCallPartitionedCall4conv2d_transpose_29/StatefulPartitionedCall:output:0*
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
J__inference_activation_89_layer_call_and_return_conditional_losses_3372225?
decoded/StatefulPartitionedCallStatefulPartitionedCall&activation_89/PartitionedCall:output:0decoded_3372227decoded_3372229*
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
D__inference_decoded_layer_call_and_return_conditional_losses_3371951?
IdentityIdentity(decoded/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp/^batch_normalization_63/StatefulPartitionedCall/^batch_normalization_64/StatefulPartitionedCall/^batch_normalization_65/StatefulPartitionedCall/^batch_normalization_66/StatefulPartitionedCall/^batch_normalization_67/StatefulPartitionedCall/^batch_normalization_68/StatefulPartitionedCall/^batch_normalization_69/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall"^conv2d_64/StatefulPartitionedCall"^conv2d_65/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall,^conv2d_transpose_29/StatefulPartitionedCall ^decoded/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_63/StatefulPartitionedCall.batch_normalization_63/StatefulPartitionedCall2`
.batch_normalization_64/StatefulPartitionedCall.batch_normalization_64/StatefulPartitionedCall2`
.batch_normalization_65/StatefulPartitionedCall.batch_normalization_65/StatefulPartitionedCall2`
.batch_normalization_66/StatefulPartitionedCall.batch_normalization_66/StatefulPartitionedCall2`
.batch_normalization_67/StatefulPartitionedCall.batch_normalization_67/StatefulPartitionedCall2`
.batch_normalization_68/StatefulPartitionedCall.batch_normalization_68/StatefulPartitionedCall2`
.batch_normalization_69/StatefulPartitionedCall.batch_normalization_69/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2F
!conv2d_64/StatefulPartitionedCall!conv2d_64/StatefulPartitionedCall2F
!conv2d_65/StatefulPartitionedCall!conv2d_65/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall2Z
+conv2d_transpose_29/StatefulPartitionedCall+conv2d_transpose_29/StatefulPartitionedCall2B
decoded/StatefulPartitionedCalldecoded/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
? 
?
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_3371818

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
?

?
F__inference_conv2d_67_layer_call_and_return_conditional_losses_3374355

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
?
?
.__inference_sequential_9_layer_call_fn_3373401

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
I__inference_sequential_9_layer_call_and_return_conditional_losses_3372233y
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
?
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_3371862

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
?
?
.__inference_sequential_9_layer_call_fn_3372336
conv2d_63_input!
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_63_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_3372233y
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
_user_specified_nameconv2d_63_input
?
f
J__inference_activation_81_layer_call_and_return_conditional_losses_3374063

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
J__inference_activation_85_layer_call_and_return_conditional_losses_3374427

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
S__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3371483

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
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3374308

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
?
?
+__inference_conv2d_64_layer_call_fn_3374072

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
F__inference_conv2d_64_layer_call_and_return_conditional_losses_3372007y
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
?
E
)__inference_encoded_layer_call_fn_3374604

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
D__inference_encoded_layer_call_and_return_conditional_losses_3372188h
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
?	
?
8__inference_batch_normalization_65_layer_call_fn_3374186

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
S__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3371483?
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
+__inference_conv2d_63_layer_call_fn_3373981

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
F__inference_conv2d_63_layer_call_and_return_conditional_losses_3371975y
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
?
?
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_3374490

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
?
K
/__inference_activation_89_layer_call_fn_3374760

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
J__inference_activation_89_layer_call_and_return_conditional_losses_3372225j
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
F__inference_conv2d_69_layer_call_and_return_conditional_losses_3374537

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
?!
?
D__inference_decoded_layer_call_and_return_conditional_losses_3374808

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
?
K
/__inference_activation_87_layer_call_fn_3374656

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
J__inference_activation_87_layer_call_and_return_conditional_losses_3372201h
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
F__inference_conv2d_69_layer_call_and_return_conditional_losses_3372167

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
?

?
F__inference_conv2d_66_layer_call_and_return_conditional_losses_3374264

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
?
K
/__inference_activation_83_layer_call_fn_3374240

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
J__inference_activation_83_layer_call_and_return_conditional_losses_3372059j
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
conv2d_63_inputB
!serving_default_conv2d_63_input:0???????????E
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
.__inference_sequential_9_layer_call_fn_3372336
.__inference_sequential_9_layer_call_fn_3373401
.__inference_sequential_9_layer_call_fn_3373506
.__inference_sequential_9_layer_call_fn_3372915?
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_3373739
I__inference_sequential_9_layer_call_and_return_conditional_losses_3373972
I__inference_sequential_9_layer_call_and_return_conditional_losses_3373049
I__inference_sequential_9_layer_call_and_return_conditional_losses_3373183?
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
"__inference__wrapped_model_3371333conv2d_63_input"?
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
+__inference_conv2d_63_layer_call_fn_3373981?
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
F__inference_conv2d_63_layer_call_and_return_conditional_losses_3373991?
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
*:(2conv2d_63/kernel
:2conv2d_63/bias
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
8__inference_batch_normalization_63_layer_call_fn_3374004
8__inference_batch_normalization_63_layer_call_fn_3374017?
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
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3374035
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3374053?
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
*:(2batch_normalization_63/gamma
):'2batch_normalization_63/beta
2:0 (2"batch_normalization_63/moving_mean
6:4 (2&batch_normalization_63/moving_variance
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
/__inference_activation_81_layer_call_fn_3374058?
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
J__inference_activation_81_layer_call_and_return_conditional_losses_3374063?
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
+__inference_conv2d_64_layer_call_fn_3374072?
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
F__inference_conv2d_64_layer_call_and_return_conditional_losses_3374082?
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
*:(2conv2d_64/kernel
:2conv2d_64/bias
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
8__inference_batch_normalization_64_layer_call_fn_3374095
8__inference_batch_normalization_64_layer_call_fn_3374108?
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
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3374126
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3374144?
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
*:(2batch_normalization_64/gamma
):'2batch_normalization_64/beta
2:0 (2"batch_normalization_64/moving_mean
6:4 (2&batch_normalization_64/moving_variance
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
/__inference_activation_82_layer_call_fn_3374149?
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
J__inference_activation_82_layer_call_and_return_conditional_losses_3374154?
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
+__inference_conv2d_65_layer_call_fn_3374163?
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
F__inference_conv2d_65_layer_call_and_return_conditional_losses_3374173?
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
*:(2conv2d_65/kernel
:2conv2d_65/bias
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
8__inference_batch_normalization_65_layer_call_fn_3374186
8__inference_batch_normalization_65_layer_call_fn_3374199?
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
S__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3374217
S__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3374235?
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
*:(2batch_normalization_65/gamma
):'2batch_normalization_65/beta
2:0 (2"batch_normalization_65/moving_mean
6:4 (2&batch_normalization_65/moving_variance
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
/__inference_activation_83_layer_call_fn_3374240?
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
J__inference_activation_83_layer_call_and_return_conditional_losses_3374245?
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
+__inference_conv2d_66_layer_call_fn_3374254?
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
F__inference_conv2d_66_layer_call_and_return_conditional_losses_3374264?
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
*:( 2conv2d_66/kernel
: 2conv2d_66/bias
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
8__inference_batch_normalization_66_layer_call_fn_3374277
8__inference_batch_normalization_66_layer_call_fn_3374290?
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
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3374308
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3374326?
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
*:( 2batch_normalization_66/gamma
):' 2batch_normalization_66/beta
2:0  (2"batch_normalization_66/moving_mean
6:4  (2&batch_normalization_66/moving_variance
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
/__inference_activation_84_layer_call_fn_3374331?
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
J__inference_activation_84_layer_call_and_return_conditional_losses_3374336?
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
+__inference_conv2d_67_layer_call_fn_3374345?
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
F__inference_conv2d_67_layer_call_and_return_conditional_losses_3374355?
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
*:(  2conv2d_67/kernel
: 2conv2d_67/bias
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
8__inference_batch_normalization_67_layer_call_fn_3374368
8__inference_batch_normalization_67_layer_call_fn_3374381?
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
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3374399
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3374417?
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
*:( 2batch_normalization_67/gamma
):' 2batch_normalization_67/beta
2:0  (2"batch_normalization_67/moving_mean
6:4  (2&batch_normalization_67/moving_variance
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
/__inference_activation_85_layer_call_fn_3374422?
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
J__inference_activation_85_layer_call_and_return_conditional_losses_3374427?
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
+__inference_conv2d_68_layer_call_fn_3374436?
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
F__inference_conv2d_68_layer_call_and_return_conditional_losses_3374446?
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
*:( @2conv2d_68/kernel
:@2conv2d_68/bias
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
8__inference_batch_normalization_68_layer_call_fn_3374459
8__inference_batch_normalization_68_layer_call_fn_3374472?
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
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_3374490
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_3374508?
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
*:(@2batch_normalization_68/gamma
):'@2batch_normalization_68/beta
2:0@ (2"batch_normalization_68/moving_mean
6:4@ (2&batch_normalization_68/moving_variance
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
/__inference_activation_86_layer_call_fn_3374513?
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
J__inference_activation_86_layer_call_and_return_conditional_losses_3374518?
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
+__inference_conv2d_69_layer_call_fn_3374527?
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
F__inference_conv2d_69_layer_call_and_return_conditional_losses_3374537?
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
*:(@ 2conv2d_69/kernel
: 2conv2d_69/bias
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
8__inference_batch_normalization_69_layer_call_fn_3374550
8__inference_batch_normalization_69_layer_call_fn_3374563?
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
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_3374581
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_3374599?
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
*:( 2batch_normalization_69/gamma
):' 2batch_normalization_69/beta
2:0  (2"batch_normalization_69/moving_mean
6:4  (2&batch_normalization_69/moving_variance
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
)__inference_encoded_layer_call_fn_3374604?
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
D__inference_encoded_layer_call_and_return_conditional_losses_3374609?
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
5__inference_conv2d_transpose_27_layer_call_fn_3374618?
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
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_3374651?
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
4:2@ 2conv2d_transpose_27/kernel
&:$@2conv2d_transpose_27/bias
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
/__inference_activation_87_layer_call_fn_3374656?
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
J__inference_activation_87_layer_call_and_return_conditional_losses_3374661?
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
5__inference_conv2d_transpose_28_layer_call_fn_3374670?
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
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_3374703?
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
4:2 @2conv2d_transpose_28/kernel
&:$ 2conv2d_transpose_28/bias
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
/__inference_activation_88_layer_call_fn_3374708?
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
J__inference_activation_88_layer_call_and_return_conditional_losses_3374713?
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
5__inference_conv2d_transpose_29_layer_call_fn_3374722?
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
P__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_3374755?
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
4:2 2conv2d_transpose_29/kernel
&:$2conv2d_transpose_29/bias
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
/__inference_activation_89_layer_call_fn_3374760?
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
J__inference_activation_89_layer_call_and_return_conditional_losses_3374765?
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
)__inference_decoded_layer_call_fn_3374774?
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
D__inference_decoded_layer_call_and_return_conditional_losses_3374808?
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
.__inference_sequential_9_layer_call_fn_3372336conv2d_63_input"?
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
.__inference_sequential_9_layer_call_fn_3373401inputs"?
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
.__inference_sequential_9_layer_call_fn_3373506inputs"?
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
.__inference_sequential_9_layer_call_fn_3372915conv2d_63_input"?
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_3373739inputs"?
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_3373972inputs"?
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_3373049conv2d_63_input"?
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_3373183conv2d_63_input"?
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
%__inference_signature_wrapper_3373296conv2d_63_input"?
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
+__inference_conv2d_63_layer_call_fn_3373981inputs"?
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
F__inference_conv2d_63_layer_call_and_return_conditional_losses_3373991inputs"?
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
8__inference_batch_normalization_63_layer_call_fn_3374004inputs"?
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
8__inference_batch_normalization_63_layer_call_fn_3374017inputs"?
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
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3374035inputs"?
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
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3374053inputs"?
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
/__inference_activation_81_layer_call_fn_3374058inputs"?
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
J__inference_activation_81_layer_call_and_return_conditional_losses_3374063inputs"?
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
+__inference_conv2d_64_layer_call_fn_3374072inputs"?
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
F__inference_conv2d_64_layer_call_and_return_conditional_losses_3374082inputs"?
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
8__inference_batch_normalization_64_layer_call_fn_3374095inputs"?
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
8__inference_batch_normalization_64_layer_call_fn_3374108inputs"?
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
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3374126inputs"?
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
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3374144inputs"?
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
/__inference_activation_82_layer_call_fn_3374149inputs"?
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
J__inference_activation_82_layer_call_and_return_conditional_losses_3374154inputs"?
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
+__inference_conv2d_65_layer_call_fn_3374163inputs"?
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
F__inference_conv2d_65_layer_call_and_return_conditional_losses_3374173inputs"?
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
8__inference_batch_normalization_65_layer_call_fn_3374186inputs"?
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
8__inference_batch_normalization_65_layer_call_fn_3374199inputs"?
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
S__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3374217inputs"?
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
S__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3374235inputs"?
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
/__inference_activation_83_layer_call_fn_3374240inputs"?
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
J__inference_activation_83_layer_call_and_return_conditional_losses_3374245inputs"?
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
+__inference_conv2d_66_layer_call_fn_3374254inputs"?
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
F__inference_conv2d_66_layer_call_and_return_conditional_losses_3374264inputs"?
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
8__inference_batch_normalization_66_layer_call_fn_3374277inputs"?
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
8__inference_batch_normalization_66_layer_call_fn_3374290inputs"?
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
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3374308inputs"?
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
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3374326inputs"?
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
/__inference_activation_84_layer_call_fn_3374331inputs"?
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
J__inference_activation_84_layer_call_and_return_conditional_losses_3374336inputs"?
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
+__inference_conv2d_67_layer_call_fn_3374345inputs"?
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
F__inference_conv2d_67_layer_call_and_return_conditional_losses_3374355inputs"?
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
8__inference_batch_normalization_67_layer_call_fn_3374368inputs"?
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
8__inference_batch_normalization_67_layer_call_fn_3374381inputs"?
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
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3374399inputs"?
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
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3374417inputs"?
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
/__inference_activation_85_layer_call_fn_3374422inputs"?
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
J__inference_activation_85_layer_call_and_return_conditional_losses_3374427inputs"?
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
+__inference_conv2d_68_layer_call_fn_3374436inputs"?
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
F__inference_conv2d_68_layer_call_and_return_conditional_losses_3374446inputs"?
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
8__inference_batch_normalization_68_layer_call_fn_3374459inputs"?
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
8__inference_batch_normalization_68_layer_call_fn_3374472inputs"?
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
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_3374490inputs"?
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
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_3374508inputs"?
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
/__inference_activation_86_layer_call_fn_3374513inputs"?
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
J__inference_activation_86_layer_call_and_return_conditional_losses_3374518inputs"?
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
+__inference_conv2d_69_layer_call_fn_3374527inputs"?
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
F__inference_conv2d_69_layer_call_and_return_conditional_losses_3374537inputs"?
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
8__inference_batch_normalization_69_layer_call_fn_3374550inputs"?
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
8__inference_batch_normalization_69_layer_call_fn_3374563inputs"?
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
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_3374581inputs"?
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
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_3374599inputs"?
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
)__inference_encoded_layer_call_fn_3374604inputs"?
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
D__inference_encoded_layer_call_and_return_conditional_losses_3374609inputs"?
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
5__inference_conv2d_transpose_27_layer_call_fn_3374618inputs"?
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
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_3374651inputs"?
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
/__inference_activation_87_layer_call_fn_3374656inputs"?
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
J__inference_activation_87_layer_call_and_return_conditional_losses_3374661inputs"?
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
5__inference_conv2d_transpose_28_layer_call_fn_3374670inputs"?
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
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_3374703inputs"?
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
/__inference_activation_88_layer_call_fn_3374708inputs"?
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
J__inference_activation_88_layer_call_and_return_conditional_losses_3374713inputs"?
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
5__inference_conv2d_transpose_29_layer_call_fn_3374722inputs"?
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
P__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_3374755inputs"?
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
/__inference_activation_89_layer_call_fn_3374760inputs"?
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
J__inference_activation_89_layer_call_and_return_conditional_losses_3374765inputs"?
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
)__inference_decoded_layer_call_fn_3374774inputs"?
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
D__inference_decoded_layer_call_and_return_conditional_losses_3374808inputs"?
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
/:-2Adam/conv2d_63/kernel/m
!:2Adam/conv2d_63/bias/m
/:-2#Adam/batch_normalization_63/gamma/m
.:,2"Adam/batch_normalization_63/beta/m
/:-2Adam/conv2d_64/kernel/m
!:2Adam/conv2d_64/bias/m
/:-2#Adam/batch_normalization_64/gamma/m
.:,2"Adam/batch_normalization_64/beta/m
/:-2Adam/conv2d_65/kernel/m
!:2Adam/conv2d_65/bias/m
/:-2#Adam/batch_normalization_65/gamma/m
.:,2"Adam/batch_normalization_65/beta/m
/:- 2Adam/conv2d_66/kernel/m
!: 2Adam/conv2d_66/bias/m
/:- 2#Adam/batch_normalization_66/gamma/m
.:, 2"Adam/batch_normalization_66/beta/m
/:-  2Adam/conv2d_67/kernel/m
!: 2Adam/conv2d_67/bias/m
/:- 2#Adam/batch_normalization_67/gamma/m
.:, 2"Adam/batch_normalization_67/beta/m
/:- @2Adam/conv2d_68/kernel/m
!:@2Adam/conv2d_68/bias/m
/:-@2#Adam/batch_normalization_68/gamma/m
.:,@2"Adam/batch_normalization_68/beta/m
/:-@ 2Adam/conv2d_69/kernel/m
!: 2Adam/conv2d_69/bias/m
/:- 2#Adam/batch_normalization_69/gamma/m
.:, 2"Adam/batch_normalization_69/beta/m
9:7@ 2!Adam/conv2d_transpose_27/kernel/m
+:)@2Adam/conv2d_transpose_27/bias/m
9:7 @2!Adam/conv2d_transpose_28/kernel/m
+:) 2Adam/conv2d_transpose_28/bias/m
9:7 2!Adam/conv2d_transpose_29/kernel/m
+:)2Adam/conv2d_transpose_29/bias/m
-:+2Adam/decoded/kernel/m
:2Adam/decoded/bias/m
/:-2Adam/conv2d_63/kernel/v
!:2Adam/conv2d_63/bias/v
/:-2#Adam/batch_normalization_63/gamma/v
.:,2"Adam/batch_normalization_63/beta/v
/:-2Adam/conv2d_64/kernel/v
!:2Adam/conv2d_64/bias/v
/:-2#Adam/batch_normalization_64/gamma/v
.:,2"Adam/batch_normalization_64/beta/v
/:-2Adam/conv2d_65/kernel/v
!:2Adam/conv2d_65/bias/v
/:-2#Adam/batch_normalization_65/gamma/v
.:,2"Adam/batch_normalization_65/beta/v
/:- 2Adam/conv2d_66/kernel/v
!: 2Adam/conv2d_66/bias/v
/:- 2#Adam/batch_normalization_66/gamma/v
.:, 2"Adam/batch_normalization_66/beta/v
/:-  2Adam/conv2d_67/kernel/v
!: 2Adam/conv2d_67/bias/v
/:- 2#Adam/batch_normalization_67/gamma/v
.:, 2"Adam/batch_normalization_67/beta/v
/:- @2Adam/conv2d_68/kernel/v
!:@2Adam/conv2d_68/bias/v
/:-@2#Adam/batch_normalization_68/gamma/v
.:,@2"Adam/batch_normalization_68/beta/v
/:-@ 2Adam/conv2d_69/kernel/v
!: 2Adam/conv2d_69/bias/v
/:- 2#Adam/batch_normalization_69/gamma/v
.:, 2"Adam/batch_normalization_69/beta/v
9:7@ 2!Adam/conv2d_transpose_27/kernel/v
+:)@2Adam/conv2d_transpose_27/bias/v
9:7 @2!Adam/conv2d_transpose_28/kernel/v
+:) 2Adam/conv2d_transpose_28/bias/v
9:7 2!Adam/conv2d_transpose_29/kernel/v
+:)2Adam/conv2d_transpose_29/bias/v
-:+2Adam/decoded/kernel/v
:2Adam/decoded/bias/v?
"__inference__wrapped_model_3371333?P,-6789FGPQRS`ajklmz{??????????????????????????????B??
8?5
3?0
conv2d_63_input???????????
? ";?8
6
decoded+?(
decoded????????????
J__inference_activation_81_layer_call_and_return_conditional_losses_3374063l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
/__inference_activation_81_layer_call_fn_3374058_9?6
/?,
*?'
inputs???????????
? ""?????????????
J__inference_activation_82_layer_call_and_return_conditional_losses_3374154l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
/__inference_activation_82_layer_call_fn_3374149_9?6
/?,
*?'
inputs???????????
? ""?????????????
J__inference_activation_83_layer_call_and_return_conditional_losses_3374245l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
/__inference_activation_83_layer_call_fn_3374240_9?6
/?,
*?'
inputs???????????
? ""?????????????
J__inference_activation_84_layer_call_and_return_conditional_losses_3374336h7?4
-?*
(?%
inputs?????????@@ 
? "-?*
#? 
0?????????@@ 
? ?
/__inference_activation_84_layer_call_fn_3374331[7?4
-?*
(?%
inputs?????????@@ 
? " ??????????@@ ?
J__inference_activation_85_layer_call_and_return_conditional_losses_3374427h7?4
-?*
(?%
inputs?????????@@ 
? "-?*
#? 
0?????????@@ 
? ?
/__inference_activation_85_layer_call_fn_3374422[7?4
-?*
(?%
inputs?????????@@ 
? " ??????????@@ ?
J__inference_activation_86_layer_call_and_return_conditional_losses_3374518h7?4
-?*
(?%
inputs?????????  @
? "-?*
#? 
0?????????  @
? ?
/__inference_activation_86_layer_call_fn_3374513[7?4
-?*
(?%
inputs?????????  @
? " ??????????  @?
J__inference_activation_87_layer_call_and_return_conditional_losses_3374661h7?4
-?*
(?%
inputs?????????  @
? "-?*
#? 
0?????????  @
? ?
/__inference_activation_87_layer_call_fn_3374656[7?4
-?*
(?%
inputs?????????  @
? " ??????????  @?
J__inference_activation_88_layer_call_and_return_conditional_losses_3374713h7?4
-?*
(?%
inputs?????????@@ 
? "-?*
#? 
0?????????@@ 
? ?
/__inference_activation_88_layer_call_fn_3374708[7?4
-?*
(?%
inputs?????????@@ 
? " ??????????@@ ?
J__inference_activation_89_layer_call_and_return_conditional_losses_3374765l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
/__inference_activation_89_layer_call_fn_3374760_9?6
/?,
*?'
inputs???????????
? ""?????????????
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3374035?6789M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3374053?6789M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_63_layer_call_fn_3374004?6789M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_63_layer_call_fn_3374017?6789M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3374126?PQRSM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3374144?PQRSM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_64_layer_call_fn_3374095?PQRSM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_64_layer_call_fn_3374108?PQRSM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
S__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3374217?jklmM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3374235?jklmM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_65_layer_call_fn_3374186?jklmM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_65_layer_call_fn_3374199?jklmM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3374308?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3374326?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
8__inference_batch_normalization_66_layer_call_fn_3374277?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
8__inference_batch_normalization_66_layer_call_fn_3374290?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3374399?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3374417?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
8__inference_batch_normalization_67_layer_call_fn_3374368?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
8__inference_batch_normalization_67_layer_call_fn_3374381?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_3374490?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_3374508?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
8__inference_batch_normalization_68_layer_call_fn_3374459?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
8__inference_batch_normalization_68_layer_call_fn_3374472?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_3374581?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_3374599?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
8__inference_batch_normalization_69_layer_call_fn_3374550?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
8__inference_batch_normalization_69_layer_call_fn_3374563?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
F__inference_conv2d_63_layer_call_and_return_conditional_losses_3373991p,-9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
+__inference_conv2d_63_layer_call_fn_3373981c,-9?6
/?,
*?'
inputs???????????
? ""?????????????
F__inference_conv2d_64_layer_call_and_return_conditional_losses_3374082pFG9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
+__inference_conv2d_64_layer_call_fn_3374072cFG9?6
/?,
*?'
inputs???????????
? ""?????????????
F__inference_conv2d_65_layer_call_and_return_conditional_losses_3374173p`a9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
+__inference_conv2d_65_layer_call_fn_3374163c`a9?6
/?,
*?'
inputs???????????
? ""?????????????
F__inference_conv2d_66_layer_call_and_return_conditional_losses_3374264nz{9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????@@ 
? ?
+__inference_conv2d_66_layer_call_fn_3374254az{9?6
/?,
*?'
inputs???????????
? " ??????????@@ ?
F__inference_conv2d_67_layer_call_and_return_conditional_losses_3374355n??7?4
-?*
(?%
inputs?????????@@ 
? "-?*
#? 
0?????????@@ 
? ?
+__inference_conv2d_67_layer_call_fn_3374345a??7?4
-?*
(?%
inputs?????????@@ 
? " ??????????@@ ?
F__inference_conv2d_68_layer_call_and_return_conditional_losses_3374446n??7?4
-?*
(?%
inputs?????????@@ 
? "-?*
#? 
0?????????  @
? ?
+__inference_conv2d_68_layer_call_fn_3374436a??7?4
-?*
(?%
inputs?????????@@ 
? " ??????????  @?
F__inference_conv2d_69_layer_call_and_return_conditional_losses_3374537n??7?4
-?*
(?%
inputs?????????  @
? "-?*
#? 
0????????? 
? ?
+__inference_conv2d_69_layer_call_fn_3374527a??7?4
-?*
(?%
inputs?????????  @
? " ?????????? ?
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_3374651???I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????@
? ?
5__inference_conv2d_transpose_27_layer_call_fn_3374618???I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+???????????????????????????@?
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_3374703???I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
5__inference_conv2d_transpose_28_layer_call_fn_3374670???I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
P__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_3374755???I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
5__inference_conv2d_transpose_29_layer_call_fn_3374722???I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
D__inference_decoded_layer_call_and_return_conditional_losses_3374808???I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
)__inference_decoded_layer_call_fn_3374774???I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
D__inference_encoded_layer_call_and_return_conditional_losses_3374609h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
)__inference_encoded_layer_call_fn_3374604[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
I__inference_sequential_9_layer_call_and_return_conditional_losses_3373049?P,-6789FGPQRS`ajklmz{??????????????????????????????J?G
@?=
3?0
conv2d_63_input???????????
p 

 
? "/?,
%?"
0???????????
? ?
I__inference_sequential_9_layer_call_and_return_conditional_losses_3373183?P,-6789FGPQRS`ajklmz{??????????????????????????????J?G
@?=
3?0
conv2d_63_input???????????
p

 
? "/?,
%?"
0???????????
? ?
I__inference_sequential_9_layer_call_and_return_conditional_losses_3373739?P,-6789FGPQRS`ajklmz{??????????????????????????????A?>
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_3373972?P,-6789FGPQRS`ajklmz{??????????????????????????????A?>
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
.__inference_sequential_9_layer_call_fn_3372336?P,-6789FGPQRS`ajklmz{??????????????????????????????J?G
@?=
3?0
conv2d_63_input???????????
p 

 
? ""?????????????
.__inference_sequential_9_layer_call_fn_3372915?P,-6789FGPQRS`ajklmz{??????????????????????????????J?G
@?=
3?0
conv2d_63_input???????????
p

 
? ""?????????????
.__inference_sequential_9_layer_call_fn_3373401?P,-6789FGPQRS`ajklmz{??????????????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? ""?????????????
.__inference_sequential_9_layer_call_fn_3373506?P,-6789FGPQRS`ajklmz{??????????????????????????????A?>
7?4
*?'
inputs???????????
p

 
? ""?????????????
%__inference_signature_wrapper_3373296?P,-6789FGPQRS`ajklmz{??????????????????????????????U?R
? 
K?H
F
conv2d_63_input3?0
conv2d_63_input???????????";?8
6
decoded+?(
decoded???????????