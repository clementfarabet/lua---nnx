local Narrow, parent = torch.class('nn.Narrow', 'nn.Module')

local help_desc =
[[Selects a subset of a dimension of a nxpxqx.. Tensor.]]

local help_example =
[[mlp=nn.Sequential();
mlp:add(nn.Narrow(1,3,2))

require "lab"
x=lab.randn(10,5)
print(x)
print(mlp:forward(x))

-- gives the output:
 0.9720 -0.0836  0.0831 -0.2059 -0.0871
 0.8750 -2.0432 -0.1295 -2.3932  0.8168
 0.0369  1.1633  0.6483  1.2862  0.6596
 0.1667 -0.5704 -0.7303  0.3697 -2.2941
 0.4794  2.0636  0.3502  0.3560 -0.5500
-0.1898 -1.1547  0.1145 -1.1399  0.1711
-1.5130  1.4445  0.2356 -0.5393 -0.6222
-0.6587  0.4314  1.1916 -1.4509  1.9400
 0.2733  1.0911  0.7667  0.4002  0.1646
 0.5804 -0.5333  1.1621  1.5683 -0.1978
[torch.Tensor of dimension 10x5]

 0.0369  1.1633  0.6483  1.2862  0.6596
 0.1667 -0.5704 -0.7303  0.3697 -2.2941
[torch.Tensor of dimension 2x5] ]]

function Narrow:__init(dimension,offset,length)
   parent.__init(self)
   self.dimension=dimension
   self.index=offset
   self.length=length or 1
   if not dimension or not offset then
      error(xlua.usage('nn.Narrow', help_desc, help_example,
                       {type='number', help='dimension', req=true},
                       {type='number', help='offset', req=true},
                       {type='number', help='length', default=1}))
   end
end

function Narrow:forward(input)
   local output=input:narrow(self.dimension,self.index,self.length);
   self.output:resizeAs(output)
   return self.output:copy(output)
end

function Narrow:backward(input, gradOutput)
   self.gradInput:resizeAs(input)  
   self.gradInput:zero();
   self.gradInput:narrow(self.dimension,self.index,self.length):copy(gradOutput)
   return self.gradInput
end 

function Narrow:write(file) 
   parent.write(self, file)
   file:writeInt(self.dimension)
   file:writeLong(self.index)
   file:writeLong(self.length)
end

function Narrow:read(file, version)
   parent.read(self, file)
   self.dimension = file:readInt()
   self.index = file:readLong()
   self.length = file:readLong()
end
