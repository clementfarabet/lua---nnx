--Based on: http://arxiv.org/pdf/1502.03167v3
--Usage example:
------------------------------------
--   model:add(nn.BatchNormalization(3 * 32 * 32))
------------------------------------

require 'nn'
require 'cunn'
local BatchNormalization, parent = torch.class('nn.BatchNormalization', 'nn.Module')

function BatchNormalization:__init(inputSize)
   parent.__init(self)
   self.bias = torch.Tensor(inputSize)
   self.weight = torch.Tensor(inputSize)
   self.gradBias = torch.Tensor(inputSize)
   self.gradWeight = torch.Tensor(inputSize)
   
   self:reset(stdv)
end   

function BatchNormalization:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.bias:nElement())
   end
   
   self.bias:uniform(-stdv,stdv)
   self.weight:uniform(-stdv,stdv)
end

function BatchNormalization:updateOutput(input)
   self.output = self.output or input.new()
   self.output:resizeAs(input) 
   self.size = input:nElement()
   self.std = torch.std(input)  * torch.sqrt((self.size - 1.0) / self.size )
   self.mean = torch.mean(input)
   self.stdcube = torch.pow(self.std,3)
   self.ones = torch.Tensor(self.size):fill(1.0)-- :cuda()
   self.output:copy(input):add(-self.mean):div(self.std)
   self.buffer = self.buffer or input.new()
   self.buffer:resizeAs(self.output):copy(self.output)
   self.output:cmul(self.weight)
   self.output:add(self.bias)
return self.output
end

function BatchNormalization:updateGradInput(input, gradOutput)

   self.buffer = self.buffer or gradOutput.new()
   self.buffer:resizeAs(gradOutput):copy(gradOutput)
   self.buffer:cmul(self.weight)
   self.dotprod1 = torch.dot(self.ones,self.buffer)
   local der1 = self.ones:clone()
   der1:mul(- self.dotprod1 / self.size/self.std)
   -- x_i - mu
   local der2 = input:clone()
   der2:add(-self.mean)

   self.dotprod2 = torch.dot(der2,self.buffer)
   der2:mul(self.dotprod2 / self.size / self.stdcube)

   self.gradInput = self.buffer:clone()
   
   self.gradInput:div(self.std)

   self.gradInput:add(der1)
   self.gradInput:add(-der2)
   return self.gradInput
end

function BatchNormalization:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   
   self.gradBias:add(scale,gradOutput)
   self.gradWeight:addcmul(scale,self.buffer,gradOutput)
end



