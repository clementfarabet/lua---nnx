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
   if self.parallelize > 1 then
      if not xrequire 'parallel' then
         xerror('install parallel for Lua to enable parallel computing (luarocks install parallel)',
                'nn.LBFGSOptimization')
      end
      parallel.setSharedSize(4*1024*1024)
   end
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
   -- parameters
   local P = self.parallelize

   -- (0a) replicate output and gradParameters
   local outputs = {}
   local gradParameters = {}

   -- (0b) divide input/target batch into N batches
   local inputss = {}
   local targetss = {}
   for t = 1,P do
      inputss[t] = {}
      targetss[t] = {}
      for i = t,#inputs,P do
         table.insert(inputss[t], inputs[i])
         table.insert(targetss[t], targets[i])
      end
   end

   -- (1) construct a closure that compute f(inputs) + df/dW
   --     after each call to that function:
   --       + self.parameters contains the current X vector
   --       + self.gradParameters contains the estimated dF/dX vector
   --       + self.output contains the estimated (average) F(X)
   lbfgs.evaluate
      = function()
           -- reset parallel state
           parallel.reset()
           -- dispatch N parallel jobs
           for t = 1,P do
              parallel.run(lbfgs.evaluate_map)
           end
           -- load parameters into current model
           self:unflatten(self.parametersT, self.gradParametersT)
           -- transmit data to all jobs
           for t = 1,P do
              -- transmit all necessary data
              parallel.children[t]:send(self.module)
              parallel.children[t]:send(self.criterion)
              parallel.children[t]:send(inputss[t])
              parallel.children[t]:send(targetss[t])
           end
           -- then wait for all workers to return their trained modules
           for t = 1,P do
              gradParameters = parallel.children[t]:receive()
              outputs[t] = parallel.children[t]:receive()
           end
           -- and join
           parallel.children:join()
           -- reduce
           return lbfgs.evaluate_reduce()
        end

   -- (1a) the map part of the evaluation: compute partial gradients
   --      in separate threads
   lbfgs.evaluate_map = [[
         -- require packages
         require 'nnx'

         -- retrieve module + criterion + mini-batch
         module = parallel.parent:receive()
         criterion = parallel.parent:receive()
         inputs = parallel.parent:receive()
         targets = parallel.parent:receive()

         -- reset gradients
         module:zeroGradParameters()
         -- f is the average of all criterions
         local output = 0
         -- evaluate gradients on inputs for this thread
         for i = 1,#inputs do
            -- estimate f
            local output = module:forward(inputs[i])
            local err = criterion:forward(output, targets[i])
            output = output + err
            -- estimate df/dW
            local df_do = criterion:backward(output, targets[i])
            module:backward(inputs[i], df_do)
         end

         -- return partial gradParameters + output
         parallel.parent:send( nnx.getGradParameters(module) )
         parallel.parent:send(output)
   ]]

   -- (1b) the reduce part of the evaluation: accumulate all
   --      partial estimates of the gradients
   lbfgs.evaluate_reduce
      = function()
           -- temp vectors for accumulation
           self.gradParametersAcc = self.gradParametersAcc or torch.Tensor()
           self.gradParametersAcc:resizeAs(self.gradParameters):zero()
           -- update state from computed parameters
           for t = 1,P do
              self:flatten(self.parametersT, gradParameters)
              self.gradParametersAcc:add(self.gradParameters)
           end
           self.gradParameters:copy(self.gradParametersAcc)
           -- normalize gradients
           self.gradParameters:div(#inputs)
           -- return average f(X)
           self.output = 0
           for t = 1,P do
              self.output = self.output + outputs[t]
           end
           -- export parameters, again
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
