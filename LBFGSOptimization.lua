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
      {arg='parallelize', type='number', help='parallelize onto N cores (experimental!)', default=1},
      {arg='verbose', type='number', help='verbose level during training [0-2]', default=0}
   )
   self.parametersT = nnx.getParameters(self.module)
   self.gradParametersT = nnx.getGradParameters(self.module)
   lbfgs.verbose = self.verbose
end

function LBFGS:forward(inputs, targets, options)
   options = options or {}
   if self.parallelize > 1 then
      return self:forward_mapreduce(inputs, targets, options)
   else
      return self:forward_sequential(inputs, targets, options)
   end
end

function LBFGS:forward_sequential(inputs, targets, options)
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

function LBFGS:forward_mapreduce(inputs, targets, options)
   -- (0) clone module+criterion for parallel evaluations
   local modules = {}
   local criterions = {}
   local outputs = {}
   self.parametersPT = {}
   self.gradParametersPT = {}
   for m = 1,self.parallelize do
      if m == 1 then
         modules[m] = self.module
         criterions[m] = self.criterion
      else
         modules[m] = self.module:clone()
         criterions[m] = self.criterion:clone()
      end
      self.parametersPT[m] = nnx.getParameters(modules[m])
      self.gradParametersPT[m] = nnx.getGradParameters(modules[m])
   end

   -- (1) construct a closure that compute f(inputs) + df/dW
   --     after each call to that function:
   --       + self.parameters contains the current X vector
   --       + self.gradParameters contains the estimated dF/dX vector
   --       + self.output contains the estimated (average) F(X)
   lbfgs.evaluate
      = function()
           for t = 1,self.parallelize do
              lbfgs.evaluate_map(t)
           end
           return lbfgs.evaluate_reduce()
        end

   -- (1a) the map part of the evaluation: compute partial gradients
   --      in separate threads
   lbfgs.evaluate_map
      = function(thread)
           -- set parameters of current state
           self:unflatten(self.parametersPT[thread], self.gradParametersPT[thread])
           -- reset gradients
           modules[thread]:zeroGradParameters()
           -- f is the average of all criterions
           outputs[thread] = 0
           -- evaluate gradients on inputs for this thread
           for i = thread,#inputs,#modules do
              -- estimate f
              local output = modules[thread]:forward(inputs[i])
              local err = criterions[thread]:forward(output, targets[i])
              outputs[thread] = outputs[thread] + err
              -- estimate df/dW
              local df_do = criterions[thread]:backward(output, targets[i])
              modules[thread]:backward(inputs[i], df_do)
           end
        end

   -- (1b) the reduce part of the evaluation: accumulate all
   --      partial estimates of the gradients
   lbfgs.evaluate_reduce
      = function()
           -- temp vectors for accumulation
           self.gradParametersAcc = self.gradParametersAcc or torch.Tensor()
           self.gradParametersAcc:resizeAs(self.gradParameters):zero()
           -- update state from computed parameters
           for t = 1,self.parallelize do
              self:flatten(self.parametersPT[t], self.gradParametersPT[t])
              self.gradParametersAcc:add(self.gradParameters)
           end
           self.gradParameters:copy(self.gradParametersAcc)
           -- normalize gradients
           self.gradParameters:div(#inputs)
           -- return average f(X)
           self.output = 0
           for t = 1,self.parallelize do
              self.output = self.output + outputs[t]
           end
           return self.output/#inputs
        end

   -- (2) store current parameters/gradParameters
   self:flatten(self.parametersT, self.gradParametersT)

   -- (3) the magic function: will update the parameter vector
   --     according to the l-BFGS method
   self.output = lbfgs.run(self.parameters, self.gradParameters,
                           self.maxIterations, self.maxLineSearch,
                           self.sparsity)

   -- (4) last: read parameters back into the main (not parrallel) model
   self:unflatten(self.parametersT, self.gradParametersT)

   -- (5) return current output after optimization
   return self.output
end
