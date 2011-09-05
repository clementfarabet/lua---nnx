local SGD,parent = torch.class('nn.SGDOptimization', 'nn.BatchOptimization')

function SGD:__init(...)
   parent.__init(self,...)
   xlua.unpack_class(self, {...},
      'SGDOptimization', nil,
      {arg='maxIterations', type='number', 
       help='maximum nb of iterations per pass', default=1},
      {arg='learningRate', type='number', 
       help='learning rate (W = W - rate*dE/dW)', default=1e-2},
      {arg='learningRateDecay', type='number', 
       help='learning rate decay (lr_t = lr_0 / (1 + samplesSeen*lrDecay))', default=0},
      {arg='weightDecay', type='number', 
       help='amount of weight decay (W = W - decay*W)', default=0},
      {arg='momentum', type='number', 
       help='amount of momentum on weights (dE/W = dE/dW*(1-momentum) + prev(dE/dW)*momentum)', default=0}
   )
   self.parameters = nnx.flattenParameters(nnx.getParameters(self.module))
   self.gradParameters = nnx.flattenParameters(nnx.getGradParameters(self.module))
end

function SGD:optimize()
   -- optimize N times
   for i = 1,self.maxIterations do
      -- evaluate f(X) + df/dX
      self.evaluate()

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
      local learningRate = self.learningRate / (1 + self.sampleCounter*self.learningRateDecay)
      self.parameters:add(-learningRate, self.currentGradParameters)
   end
end
