--
-- Author: Gaetan Marceau Caron (gaetan.marceau-caron@inria.fr) and Yann Ollivier
-- Description: Implementation of the quasi-diagonal reduction
-- based on the Practical Riemannian Neural Networks paper (http://arxiv.org/abs/1602.08007)
-- 
local QDRiemaNNLinear, parent = torch.class('nnx.QDRiemaNNLinear', 'nn.Linear')

function QDRiemaNNLinear:__init(inputSize, outputSize, gamma, qdFlag)
   parent.__init(self,inputSize, outputSize)
   if qdFlag == nil then -- Flag for choosing between diagonal or quasi-diagonal reductions
	  self.qdFlag = true 
   else
	  self.qdFlag = qdFlag
   end
   self.gamma = gamma  or 0.01 -- update rate of the metric
   self.matReg = 1e-12 -- numerical regularization 
   self.initMetric = true -- flag for first update
   self.Mii = torch.Tensor(outputSize, inputSize)
   if self.qdFlag then self.M0i = torch.Tensor(outputSize, inputSize) end
   self.M00 = torch.Tensor(outputSize)
end

function QDRiemaNNLinear:accGradParameters(input, gradOutput)
   parent.accGradParameters(self,input,gradOutput)

   local gradOutputSqT = torch.pow(gradOutput,2):t()
   
   if self.initMetric then
	  self.Mii:mm(gradOutputSqT,torch.pow(input,2))
	  self.M00:mv(gradOutputSqT,self.addBuffer)
	  if self.qdFlag then self.M0i:mm(gradOutputSqT,input) end
	  self.initMetric = false
   else
	  self.Mii:mul(1.-self.gamma):addmm(self.gamma,gradOutputSqT,torch.pow(input,2))
	  if self.qdFlag then self.M0i:mul(1.-self.gamma):addmm(self.gamma,gradOutputSqT,input) end
	  self.M00:mul(1.-self.gamma):addmv(self.gamma,gradOutputSqT,self.addBuffer)
   end
   
   if self.qdFlag then
	  local numerator = torch.add(torch.cmul(self.gradWeight,self.M00:view(-1,1):expandAs(self.gradWeight)), -1.0, torch.cmul(self.M0i,self.gradBias:view(-1,1):expandAs(self.M0i)))
	  local denominator = torch.add(torch.cmul(self.Mii,self.M00:view(-1,1):expandAs(self.Mii)),-1.0,torch.pow(self.M0i,2)):clamp(self.matReg,1e25)
	  self.gradWeight:copy(numerator:cdiv(denominator))
	  
	  local temp = torch.cmul(self.M0i,self.gradWeight):sum(2)
	  self.gradBias:add(-1.,temp):cdiv(torch.add(self.M00,self.matReg))
	  
   else
	  self.gradWeight:cdiv(self.Mii:add(self.matReg))
	  self.gradBias:cdiv(self.M00:add(self.matReg))
   end
end

function QDRiemaNNLinear:reset()
   self.initMetric = true
   stdv = 1./math.sqrt(self.weight:size(2))
   self.weight:normal(0, stdv)
   self.bias:zero()
   return self
end
