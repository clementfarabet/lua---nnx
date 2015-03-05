--Based on: http://arxiv.org/pdf/1502.03167v3
--If input dimension is larger than 1, a reshape is needed before and after usage.
--Usage example:
------------------------------------
--   model:add(nn.Reshape(3 * 32 * 32))
--   model:add(batchNormalization(3 * 32 * 32))
--   model:add(nn.Reshape(3 , 32 , 32))
------------------------------------

local Meanvec, parent = torch.class('nn.Meanvec', 'nn.Module')

function Meanvec:__init()
   parent.__init(self)   
end

function Meanvec:updateOutput(input)
   self.output:resizeAs(input) 
   self.size = input:nElement()
   self.std = torch.std(input)  * torch.sqrt((self.size - 1.0) / self.size )
   self.mean = torch.mean(input)
   self.output:copy(input):add(-self.mean):div(self.std)
return self.output
end

function Meanvec:updateGradInput(input, gradOutput)

 
   local der1 = input:clone():fill(1.0)
   der1 = der1:diag()
   der1:add(-1.0/self.size):div(self.std)
   local der2 = input:clone()
   der2:add(-self.mean)
	local temp = torch.Tensor(self.size,self.size):fill(0)
	temp:addr(der1,-1.0/(self.size* torch.pow(self.std,3)),der2,der2)
	self.gradInput:resizeAs(gradOutput):fill(0)
	self.gradInput:addmv(temp,gradOutput)

   return self.gradInput
end

function batchNormalization(inputSize)

local module = nn.Sequential()
   module:add(nn.Meanvec())
   module:add(nn.CMul(inputSize))
   module:add(nn.Add(inputSize,false))
   
   return module
end
