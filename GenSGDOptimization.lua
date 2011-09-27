local GenSGD,parent = torch.class('nn.GenSGDOptimization',
                                  'nn.BatchOptimization') 

-- this module parallelizes SGD in a particular way.  It sends out the
-- same batch to each of several workers, each with a different learning
-- rate.  The workers run and the parameters from the best worker and
-- it's learning rate are kept for the next batch.

function GenSGD:__init(...)
   parent.__init(self,...)
   xlua.unpack_class(self, {...},
                     'GenSGDOptimization', nil,
                     {arg='maxIterations', type='number', 
                      help='maximum nb of iterations per pass', default=1},
                     {arg='learningRate', type='number', 
                      help='learning rate (W = W - rate*dE/dW)', default=1e-2},
                     {arg='learningRateDecay', type='number', 
                      help='learning rate decay (lr_t = lr_0 / (1 + samplesSeen*lrDecay))', 
                      default=0},
                     {arg='weightDecay', type='number', 
                      help='amount of weight decay (W = W - decay*W)', default=0},
                     {arg='momentum', type='number', 
                      help='amount of momentum on weights (dE/W = dE/dW*(1-momentum) + prev(dE/dW)*momentum)', default=0},
                     {arg='sigma', type='number',
                      help='sigma of gaussian used to randomize learningRate',
                      default = 1e3}
                  )
   require 'lab' 
   if self.parallelize < 2 then
      xerror('GenSGD needs to work on several processors: set parallelize',
             'nn.GenSGDOptimization')
   end
   -- change the mapper to send the same batch to each worker
   self.copyBatch = true
   -- create default parameter set which will be randomized for each worker
   self.baseParameters = { momentum          = self.momentum, 
                           weightDecay       = self.weightDecay,
                           learningRate      = self.learningRate,
                           learningRateDecay = self.learningRateDecay,
                           sampleCounter     = self.sampleCounter
                        }
end

-- we are changing the way we map and reduce.  It would be nice to
-- change gradParametersPartial to ParametersPartial, as the logic is
-- different for this kind of parallelization.
function GenSGD:map_hook()
   local P = self.parallelize
   -- transmit new parameters to all workers
   self.children:join()
   self.children:send(self.parameters)
   print('randomizing for '..P..' lr: '..self.learningRate..' sigma: '..self.sigma)
   -- randomize learning rate (could randomize other bits)
   local n = torch.Tensor(P)

   n[1] = self.learningRate
   n[2] = self.learningRate * 10
   n[3] = self.learningRate / 10
   n[4] = self.learningRate / 100 
--  (lab.randn(P) * self.sigma):add(self.learningRate)
   self.baseParameters.sampleCounter = self.sampleCounter

   for t = 1,P do
      self.baseParameters.learningRate = n[t]
      print('lr: '..self.baseParameters.learningRate)
      --self.children[t]:join() 
      self.children[t]:send(self.baseParameters) 
   end
   -- then wait for all workers to return their Parameters + outputs
   -- should rename this to parametersParallel and optionsParallel
   gradParametersPartial = self.children:receive()
   outputsPartial = self.children:receive()
   -- force cleanup
   collectgarbage()
end

function GenSGD:reduce_hook()
   local P = self.parallelize
   local id = 0
   local mx = 1e9
   for t = 1,P do
      if outputsPartial[t].f_x < mx then
         id = t
         mx = outputsPartial[t].f_x
      end
   end
   if id == 0 then
      xerror('diverging','nn.GenSGDOptimization')
   else
      self.baseParameters = outputsPartial[id]
      self.output = self.baseParameters.f_x
      -- in this case we get the parameters back directly
      self.parameters:copy(gradParametersPartial[id])
      print('Winner: output = '..self.output..
            'learningRate = '..self.baseParameters['learningRate'])
      self.learningRate = self.baseParameters.learningRate
   end
end

function GenSGD:optimize()
   self.evaluate()
end

-- optimization (could do others in this mode)
GenSGD.optimizer = 
   function (module,params)
      -- apply momentum (store in the module)
   if params.momentum ~= 0 then
      if not module.currentGradParameters then
         module.currentGradParameters = 
            torch.Tensor():resizeAs(module.gradParameters):copy(module.gradParameters)
      else
         module.currentGradParameters:mul(params.momentum):add(1-params.momentum, module.gradParameters)
      end
   else
      module.currentGradParameters = module.gradParameters
   end

   -- weight decay
   if params.weightDecay ~= 0 then
      module.parameters:add(-params.weightDecay, module.parameters)
   end

   -- update parameters
   local learningRate = 
      params.learningRate / (1 + params.sampleCounter*params.learningRateDecay)
   module.parameters:add(-learningRate, module.currentGradParameters)
   -- make keep track of final rate
   params.learningRate = learningRate
end

function GenSGD:setup_mapreduce ()
   -- (0) startup parallel package
   if not xrequire 'parallel' then
      xerror('install parallel for Lua to enable parallel computing (luarocks install parallel)',
             'nn.GenSGDOptimization')
   end
   local worker_code  =  
      function()
         -- require packages
         require 'nnx'
         
         -- retrieve module + criterion at startup
         parallel.yield()
         
         module    = parallel.parent:receive()
         criterion = parallel.parent:receive()
         optimizer = parallel.parent:receive()
         
         -- retrieve optional prehook/posthook
         prehook = parallel.parent:receive()
         posthook = parallel.parent:receive()
         if type(prehook) ~= 'function' then prehook = nil end
         if type(posthook) ~= 'function' then posthook = nil end

         -- I don't understand this [MS]
         -- get pointer to parameter and gradParameter vectors
         -- (this assumes that parameters+gradParameters are already flat parameters:
         --  it should be the case, as the parent process flattens them at __init)
         function check(tocheck)
            for i = 2,#tocheck do
               if tocheck[i]:storage() ~= tocheck[i-1]:storage() then
                  print('<BatchOptimization> error: inconsistent parameter vector (not flat)')
                  return
               end
            end
         end
         tableParameters = nnx.getParameters(module)
         tableGradParameters = nnx.getGradParameters(module)
         check(tableParameters)
         check(tableGradParameters)
         parameters = torch.Tensor():set(tableParameters[1]:storage())
         gradParameters = torch.Tensor():set(tableGradParameters[1]:storage())
   
         -- outer loop: mini-batches
         while true do
            -- sync
            if parallel.yield() == 'break' then break end
            
            -- receive new mini-batch
            inputs  = parallel.parent:receive()
            targets = parallel.parent:receive()
            options = parallel.parent:receive()
            
            -- inner loop: evaluations
            while true do
               -- sync
               if parallel.yield() == 'break' then break end
               
               -- receive new set of parameters
               parameters:copy(parallel.parent:receive())
               -- receive the learning rate etc. parameters which are
               -- tweaked for each thread
               optimization_parameters = parallel.parent:receive()	 
               
               -- evaluate gradients on inputs for this thread and perform
               -- SGD on these inputs
               -- reset gradients 
               gradParameters:zero()
               module.parameters = parameters
               module.gradParameters = gradParameters
               for i = 1,#inputs do
                  -- estimate f
                  local output = module:forward(inputs[i])
                  local err = criterion:forward(output, targets[i])
                  -- estimate df/dW
                  local df_do = criterion:backward(output, targets[i])
                  module:backward(inputs[i], df_do)
                  module:accGradParameters(inputs[i], df_do)
                  optimizer(module,optimization_parameters) 
               end
               -- we need the result averaged over all the samples _after_
               -- the gradient steps so do one more loop to fprop through
               -- the samples and collect the error _after_ the optimization
               local f_x = 0
               for i = 1,#inputs do
                  -- estimate f
                  local output = module:forward(inputs[i])
                  local err = criterion:forward(output, targets[i])
                  f_x = f_x + err
               end
               -- in this case send back parameters themselves b/c they are
               -- already optimized
               parallel.parent:send(parameters)
               -- need to make sure we keep track of what was used to
               -- compute these params along with the outputs
               optimization_parameters['f_x'] = f_x/#inputs
               parallel.parent:send(optimization_parameters)
               -- force cleanup
               collectgarbage()
            end
         end
      end

   local setup = function()
                    -- (1) optional calibration
                    if parallel.remotes then
                       parallel.calibrate()
                    end

                    -- (2) startup all workers
                    self.children = parallel.sfork(self.parallelize)
                    self.children:exec(worker_code)
                    
                    -- (4) and send them the module + criterion architecture
                    self.children:join()
                    self.children:send(self.module)
                    self.children:send(self.criterion)
                    self.children:send(self.optimizer)
                 end

   local ok,err = pcall(setup)
   if not ok then parallel.close() error(err) end
end