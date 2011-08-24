local LBFGS,parent = torch.class('nn.LBFGSOptimization', 'nn.Optimization')

function LBFGS:__init(...)
   require 'liblbfgs'
   parent.__init(self)
   xlua.unpack_class(self, {...},
      'LBFGSOptimization', nil,
      {arg='module', type='nn.Module', help='a module to train', req=true},
      {arg='criterion', type='nn.Criterion', help='a criterion to estimate the error', req=true},
      {arg='maxIterations', type='number', help='maximum nb of iterations per pass (0 = no max)', default=0}
   )
   self.parametersT = nnx.getParameters(self.module)
   self.gradParametersT = nnx.getGradParameters(self.module)
end

function LBFGS:forward(inputs, targets)
   -- (1) construct a closure that compute f(inputs) + df/dW
   --     after each call to that function:
   --       + self.parameters contains the current X vector
   --       + self.gradParameters contains the estimated dF/dX vector
   --       + self.output contains the estimated (average) F(X)
   lbfgs.evaluate
      = function()
           -- set parameters from current state
           self:unflatten(self.parametersT, self.gradParametersT)
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
           -- update state from computed parameters
           self:flatten(self.parametersT, self.gradParametersT)
           -- return average f(X)
           return self.output/#inputs
        end

   -- (2) store current parameters/gradParameters
   self:flatten(self.parametersT, self.gradParametersT)

   -- (3) the magic function: will update the parameter vector
   --     according to the l-BFGS method
   self.output = lbfgs.run(self.parameters, self.gradParameters, self.maxIterations)

   -- (4) last: read parameters back into the model
   self:unflatten(self.parametersT, self.gradParametersT)

   -- (5) return current output after optimization
   return self.output
end
