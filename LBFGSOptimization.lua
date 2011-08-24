local LBFGS,parent = torch.class('nn.LBFGSOptimization', 'nn.Optimization')

function LBFGS:__init(...)
   require 'liblbfgs'
   parent.__init(self)
   xlua.unpack_class(self, {...},
      'LBFGSOptimization', nil,
      {arg='module', type='nn.Module', help='a module to train', req=true},
      {arg='criterion', type='nn.Criterion', help='a criterion to estimate the error', req=true},
      {arg='maxIterations', type='number', help='maximum nb of iterations per pass (0 = no max)', default=0},
      {arg='maxLineSearch', type='number', help='maximum nb of steps in line search', default=20},
      {arg='sparsity', type='number', help='sparsity coef (Orthantwise C)', default=0},
      {arg='verbose', type='number', help='verbose level during training [0-2]', default=0}
   )
   self.parametersT = nnx.getParameters(self.module)
   self.gradParametersT = nnx.getGradParameters(self.module)
   lbfgs.verbose = self.verbose
end

function LBFGS:forward(inputs, targets, options)
   options = options or {}
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
              -- user hook
              if self.prehook then
                 self.prehook(self, {inputs[i], targets[i], options[i]})
              end
              -- estimate f
              local output = self.module:forward(inputs[i])
              local err = self.criterion:forward(output, targets[i])
              self.output = self.output + err
              -- estimate df/dW
              local df_do = self.criterion:backward(output, targets[i])
              self.module:backward(inputs[i], df_do)
              -- user hook
              if self.posthook then
                 self.posthook(self, {inputs[i], targets[i], options[i]})
              end
           end
           -- update state from computed parameters
           self:flatten(self.parametersT, self.gradParametersT)
           -- normalize gradients
           self.gradParameters:div(#inputs)
           -- return average f(X)
           return self.output/#inputs
        end

   -- (2) store current parameters/gradParameters
   self:flatten(self.parametersT, self.gradParametersT)

   -- (3) the magic function: will update the parameter vector
   --     according to the l-BFGS method
   self.output = lbfgs.run(self.parameters, self.gradParameters, 
                           self.maxIterations, self.maxLineSearch,
                           self.sparsity)

   -- (4) last: read parameters back into the model
   self:unflatten(self.parametersT, self.gradParametersT)

   -- (5) return current output after optimization
   return self.output
end
