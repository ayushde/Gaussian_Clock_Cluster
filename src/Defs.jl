const DEBUG=false

const microint=Int8

const miniint=Int16

const macroint=Int32

# define OpName
@enum OpName id_op=1 field=2 Ising=3

# define Operator
struct Operator{NDIMS}
  op_type::OpName
  site::NTuple{NDIMS,miniint}
  dir::microint
  colour::Tuple{microint,microint}
end
