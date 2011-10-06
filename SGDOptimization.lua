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

function SGD:condition(inputs, targets)
   
   -- for now the only conditioning is Yann's optimal learning rate
   -- from Efficient BackProp 1998
   self.alpha = self.alpha or 1e-2 -- 1 / ||parameters|| ?
   self.gamma = self.gamma or 0.95

   if not self.phi then
      -- make tensor in current default type
      self.phi = torch.Tensor(self.gradParameters:size())
      -- no lab functions for CudaTensors so
      local old_type = torch.getdefaulttensortype()
      if (old_type == 'torch.CudaTensor') then
	 torch.setdefaulttensortype('torch.FloatTensor')
      end
      local r = lab.randn(self.gradParameters:size())
      r:div(r:norm()) -- norm 1
      if (old_type == 'torch.CudaTensor') then
	 torch.setdefaulttensortype(old_type)
      end
      self.phi:copy(r)
   end

   -- scratch vectors which we don't want to re-allocate every time
   self.param_bkup = self.param_bkup or torch.Tensor():resizeAs(self.parameters)
   self.grad_bkup = self.grad_bkup or torch.Tensor():resizeAs(self.gradParameters)
   -- single batch (not running average version)

   if type(inputs) == 'table' then      -- slow
      print("<SGD conditioning> slow version ")
      -- (1) compute dE/dw(w)
      -- reset gradients
      self.gradParameters:zero()
      for i = 1,#inputs do
	 -- estimate f
	 local output = self.module:forward(inputs[i])
	 local err  = self.criterion:forward(output, targets[i])
	 -- estimate df/dW
	 local df_do = self.criterion:backward(output, targets[i])
	 self.module:backward(inputs[i], df_do)
	 self.module:accGradParameters(inputs[i], df_do)
      end
      -- normalize gradients
      self.gradParameters:div(#inputs)
      
      -- backup gradient and weights
      self.param_bkup:copy(self.parameters)
      self.grad_bkup:copy(self.gradParameters)
      
      -- (2) compute dE/dw(w + alpha * phi / || phi|| )
      -- normalize + scale phi
      print('norm phi before: ',self.phi:norm(),' alpha: ',self.alpha)
      self.phi:div(self.phi:norm()):mul(self.alpha)
      print('norm phi after: ',self.phi:norm())
      -- perturb weights
      print('norm param before: ',self.parameters:norm())
      self.parameters:add(self.phi)
      print('norm param after: ',self.parameters:norm())
      -- reset gradients
      self.gradParameters:zero()
      --re-estimate f
      for i = 1,#inputs do
	 -- estimate f
	 output = self.module:forward(inputs[i])
	 err  = self.criterion:forward(output, targets[i])
	 -- estimate df/dW
	 df_do = self.criterion:backward(output, targets[i])
	 self.module:backward(inputs[i], df_do)
	 self.module:accGradParameters(inputs[i], df_do)
      end
      -- normalize gradients
      self.gradParameters:div(#inputs)

      -- (3) phi - 1/alpha(dE/dw(w + alpha * oldphi / || oldphi ||) - dE/dw(w))
      -- compute new phi
      self.phi:copy(self.grad_bkup):mul(-1):add(self.gradParameters):mul(1/self.alpha)
      print('norm old_grad: ',self.grad_bkup:norm(),' norm cur_grad: ',self.gradParameters:norm(), ' norm phi: ',self.phi:norm())      
      -- (4) new learning rate eta = 1 / || phi || 
      self.learningRate = 1 / self.phi:norm()
      -- (5) reset parameters and zero gradients
      self.parameters:copy(self.param_bkup)
      self.gradParameters:zero()
   else -- fast
      -- (1) compute dE/dw(w)
      -- reset gradients
      self.gradParameters:zero()
      -- estimate f
      local output = self.module:forward(inputs)
      local err  = self.criterion:forward(output, targets)
      -- estimate df/dW
      local df_do = self.criterion:backward(output, targets)
      self.module:backward(inputs, df_do)
      self.module:accGradParameters(inputs, df_do)
      -- backup gradient and weights
      self.param_bkup:copy(self.parameters)
      self.grad_bkup:copy(self.gradParameters)
      -- divide by number of samples
      -- self.grad_bkup:div(inputs:size(1))

      -- (2) compute dE/dw(w + alpha * phi / || phi|| )
      -- normalize + scale phi
      print('norm phi before: ',self.phi:norm(),' alpha: ',self.alpha)
      self.phi:div(self.phi:norm()):mul(self.alpha)
      print('norm phi after: ',self.phi:norm())
      -- perturb weights
      print('norm param before: ',self.parameters:norm())
      self.parameters:add(self.phi)
      print('norm param after: ',self.parameters:norm())
      -- reset gradients
      self.gradParameters:zero()
      --re-estimate f
      output = self.module:forward(inputs)
      self.output = self.criterion:forward(output, targets)
      -- re-estimate df/dW
      df_do = self.criterion:backward(output, targets)
      self.module:backward(inputs, df_do)
      self.module:accGradParameters(inputs, df_do)
      -- self.gradParameters:div(inputs:size(1))

      -- (3) phi - 1/alpha(dE/dw(w + alpha * oldphi / || oldphi ||) - dE/dw(w))
      -- compute new phi
      if true then
	 -- running average
	 self.phi:mul(self.gamma):add(self.grad_bkup):mul(-1):add(self.gradParameters):mul(1/self.alpha)
      else 
	 self.phi:copy(self.grad_bkup):mul(-1):add(self.gradParameters):mul(1/self.alpha)
      end
      print('norm old_grad: ',self.grad_bkup:norm(),' norm cur_grad: ',self.gradParameters:norm(), ' norm phi: ',self.phi:norm())      
      -- (4) new learning rate eta = 1 / || phi || 
      self.learningRate = 1 / self.phi:norm()
      -- (5) reset parameters and zero gradients
      self.parameters:copy(self.param_bkup)
      self.gradParameters:zero()
   end
end