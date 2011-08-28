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
      self:setup_mapreduce()
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

   -- (0c) send mini-batch to all workers
   for t = 1,P do
      parallel.children[t]:send(inputss[t])
      parallel.children[t]:send(targetss[t])
   end

   -- (1) construct a closure that compute f(inputs) + df/dW
   --     after each call to that function:
   --       + self.parameters contains the current X vector
   --       + self.gradParameters contains the estimated dF/dX vector
   --       + self.output contains the estimated (average) F(X)
   lbfgs.evaluate
      = function()
           lbfgs.evaluate_map()
           return lbfgs.evaluate_reduce()
        end

   -- (1a) the map part of the evaluation: compute partial gradients
   --      in separate threads
   lbfgs.evaluate_map
      = function()
           -- load parameters into current model
           self:unflatten(self.parametersT, self.gradParametersT)
           -- transmit new parameters to workers
           for t = 1,P do
              parallel.children[t]:send(self.parametersT)
           end
           -- then wait for all workers to return their partial gradParameters + outputs
           for t = 1,P do
              gradParameters[t] = parallel.children[t]:receive()
              outputs[t] = parallel.children[t]:receive()
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
           for t = 1,P do
              self:flatten(self.parametersT, gradParameters[t])
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

   -- (6) reset workers so they're ready for next mini-batch
   for t = 1,P do
      parallel.children[t]:send('break')
   end

   -- (5) return current output after optimization
   return self.output
end

function LBFGS:setup_mapreduce ()
   -- (0) startup parallel package
   if not xrequire 'parallel' then
      xerror('install parallel for Lua to enable parallel computing (luarocks install parallel)',
             'nn.LBFGSOptimization')
   end
   parallel.setSharedSize(4*1024*1024)
   local P = self.parallelize

   -- (1) define code for workers
   local worker_code = [[
         -- require packages
         require 'nnx'

         -- retrieve module + criterion at startup
         module = parallel.parent:receive()
         criterion = parallel.parent:receive()

         -- get pointer to parameter and gradParameter vectors
         parameters = nnx.getParameters(module)
         gradParameters = nnx.getGradParameters(module)

         -- outter loop: mini-batches
         while true do
            -- receive new mini-batch
            inputs = parallel.parent:receive()
            if type(inputs) == 'string' and inputs == 'break' then break end
            targets = parallel.parent:receive()

            -- inner loop: evaluations
            while true do
               -- receive new set of parameters
               newParameters = parallel.parent:receive()
               if type(newParameters) == 'string' and newParameters == 'break' then break end
               for i = 1,#newParameters do
                  parameters[i]:copy(newParameters[i])
               end

               -- reset gradients
               module:zeroGradParameters()
               -- f is the average of all criterions
               local f_x = 0
               -- evaluate gradients on inputs for this thread
               for i = 1,#inputs do
                  -- estimate f
                  local output = module:forward(inputs[i])
                  local err = criterion:forward(output, targets[i])
                  f_x = f_x + err
                  -- estimate df/dW
                  local df_do = criterion:backward(output, targets[i])
                  module:backward(inputs[i], df_do)
               end

               -- now send back gradParameters + partial output
               parallel.parent:send(gradParameters)
               parallel.parent:send(f_x)
            end
         end
   ]]

   -- (2) startup all workers
   for t = 1,P do
      parallel.run(worker_code)
   end

   -- (3) and send them the module + criterion architecture
   for t = 1,P do
      parallel.children[t]:send(self.module)
      parallel.children[t]:send(self.criterion)
   end
end