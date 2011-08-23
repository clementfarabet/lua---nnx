local SGD,parent = torch.class('nn.SGDOptimization', 'nn.Optimization')

function SGD:__init(...)
   parent.__init(self)
   xlua.unpack_class(self, {...},
      'SGDOptimization', nil,
      {arg='learningRate', type='number', help='learning rate (W = W - rate*dE/dW)', default=1e-2},
      {arg='weightDecay', type='number', help='amount of weight decay (W = W - decay*W)', default=0},
      {arg='momentum', type='number', help='amount of momentum on weights (dE/W = dE/dW + momentum*prev(dE/dW))', default=0}
   )
end

function SGD:forward(parameters, gradParameters)
   self:flatten(parameters, gradParameters)

   -- apply momentum
   if self.momentum ~= 0 then
      if not self.currentGradParameters then
         self.currentGradParameters = torch.Tensor():resizeAs(self.gradParameters):copy(self.gradParameters)
      else
         self.currentGradParameters:mul(self.momentum):add(self.gradParameters):div(1+self.momentum)
      end
   else
      self.currentGradParameters = self.gradParameters
   end

   -- weight decay
   if self.weightDecay ~= 0 then
      self.parameters:add(-self.weightDecay, self.parameters)
   end

   -- update parameters
   self.parameters:add(-self.learningRate, self.currentGradParameters)

   self:unflatten(parameters, gradParameters)
end
