local Classifier, parent = torch.class('nn.SpatialClassifier', 'nn.Sequential')

function Classifier:__init(classifier)
   parent.__init(self)
   -- public:
   self.classifier = classifier or nn.Sequential()
   -- private:
   self.modules = self.classifier
   self.inputF = torch.Tensor()
   self.inputT = torch.Tensor()
   self.outputF = torch.Tensor()
   self.output = torch.Tensor()
   self.gradOutputF = torch.Tensor()
   self.gradOutputT = torch.Tensor()
   self.gradInputF = torch.Tensor()
   self.gradInput = torch.Tensor()
end

function Classifier:add(module)
   self.classifier:add(module)
end

function Classifier:forward(input)
   -- get dims:
   if input:nDimension() ~= 3 then
      error('<nn.SpatialClassifier> input should be 3D: KxHxW')
   end
   local K = input:size(1)
   local H = input:size(2)
   local W = input:size(3)
   local HW = H*W
   -- transpose input:
   self.inputF:set(input):resize(K, HW)
   self.inputT:resize(HW, K):copy(self.inputF:t())
   -- classify all locations:
   self.outputT = self.classifier:forward(self.inputT)
   -- force batch
   if self.outputT:nDimension() == 1 then
      self.outputT:resize(1,self.outputT:size(1))
   end
   -- transpose output:
   local N = self.outputT:size(2)
   self.outputF:resize(N, HW):copy(self.outputT:t())
   self.output:set(self.outputF):resize(N,H,W)
   return self.output
end

function Classifier:backward(input, gradOutput)
   -- get dims:
   local K = input:size(1)
   local H = input:size(2)
   local W = input:size(3)
   local HW = H*W
   local N = gradOutput:size(1)
   -- transpose input
   self.inputF:set(input):resize(K, HW)
   self.inputT:resize(HW, K):copy(self.inputF:t())
   -- transpose gradOutput
   self.gradOutputF:set(gradOutput):resize(N, HW)
   self.gradOutputT:resize(HW, N):copy(self.gradOutputF:t())
   -- backward through classifier:
   self.gradInputT = self.classifier:backward(self.inputT, self.gradOutputT)
   -- force batch
   if self.gradInputT:nDimension() == 1 then
      self.gradInputT:resize(1,self.outputT:size(1))
   end
   -- transpose gradInput
   self.gradInputF:resize(K, HW):copy(self.gradInputT:t())
   self.gradInput:set(self.gradInputF):resize(K,H,W)
   return self.gradInput
end

function Classifier:accGradParameters(input, gradOutput, scale)
   -- get dims:
   local K = input:size(1)
   local H = input:size(2)
   local W = input:size(3)
   local HW = H*W
   local N = gradOutput:size(1)
   -- transpose input
   self.inputF:set(input):resize(K, HW)
   self.inputT:resize(HW, K):copy(self.inputF:t())
   -- transpose gradOutput
   self.gradOutputF:set(gradOutput):resize(N, HW)
   self.gradOutputT:resize(HW, N):copy(self.gradOutputF:t())
   -- backward through classifier:
   self.classifier:accGradParameters(self.inputT, self.gradOutputT, scale)
end
