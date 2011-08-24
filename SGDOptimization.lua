local SGD,parent = torch.class('nn.SGDOptimization', 'nn.Optimization')

function SGD:__init(...)
   parent.__init(self)
   xlua.unpack_class(self, {...},
      'SGDOptimization', nil,
      {arg='module', type='nn.Module', help='a module to train', req=true},
      {arg='criterion', type='nn.Criterion', help='a criterion to estimate the error', req=true},
      {arg='learningRate', type='number', help='learning rate (W = W - rate*dE/dW)', default=1e-2},
      {arg='weightDecay', type='number', help='amount of weight decay (W = W - decay*W)', default=0},
      {arg='momentum', type='number', help='amount of momentum on weights (dE/W = dE/dW*(1-momentum) + prev(dE/dW)*momentum)', default=0}
   )
   self.parametersT = nnx.getParameters(self.module)
   self.gradParametersT = nnx.getGradParameters(self.module)
end

function SGD:forward(inputs, targets)
   -- reset gradients
   self.module:zeroGradParameters()
   
   -- f is the average of all criterions
   self.output = 0
   
   -- given all inputs, evaluate gradients
   for i = 1,#inputs do
      -- estimate f
      local output = self.module:forward(inputs[i])
      local err = self.criterion:forward(output, targets[i])
      self.output = self.output + err
      
      -- estimate df/dW
      local df_do = self.criterion:backward(output, targets[i])
      self.module:backward(inputs[i], df_do)

      -- user hook
      if self.hook then
         self.hook(self, {inputs[i], targets[i]})
      end
   end

   -- renorm f
   self.output = self.output / #inputs
   
   -- update state from computed parameters
   self:flatten(self.parametersT, self.gradParametersT)

   -- apply momentum
   if self.momentum ~= 0 then
      if not self.currentGradParameters then
         self.currentGradParameters = torch.Tensor():resizeAs(self.gradParameters):copy(self.gradParameters)
      else
         self.currentGradParameters:mul(self.momentum):add(1-self.momentum, self.gradParameters)
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

   -- write compute parameters back in place
   self:unflatten(self.parametersT, self.gradParametersT)

   -- return current output
   return self.output
end
