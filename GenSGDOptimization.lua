local GenSGD,parent = torch.class('nn.GenSGDOptimization',
'nn.BatchOptimization') 

-- this module parallelizes SGD in a particular way.  It sends out the
-- same batch to each of several worker each with a different learning
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
      print('ERROR: GenSGD needs to work on several processors')
   end
   -- change the mapper to send the same batch to each worker
   self.copyBatch = true
   -- create default parameter set which will be randomized for each worker
   self.baseParameters = { momentum          = self.momentum, 
                           weightDecay       = self.weightDecay,
                           learningRate      = self.learningRate,
                           learningRateDecay = self.learningRateDecay
                        }
   self.workerParameters = torch.Tensor(self.P)
end

-- we are changing the way we map and reduce.  It would be nice to
-- change gradParametersPartial to ParametersPartial, as the logic is
-- different for this kind of parallelization.
function GenSGD:map_hook()
   -- transmit new parameters to all workers
   self.children:join()
   self.children:send(self.parameters)
   -- randomize learning rate (could randomize other bits)
   local n = self.learningRate + (lab.randn(P) * self.sigma)
   for i = 1,P do
      self.baseParameters[learningRate] = n[i]
      self.children[t]:join()
      self.children[t]:send(self.baseParameters) 
   end
      
   end
   -- then wait for all workers to return their partial gradParameters + outputs
   gradParametersPartial = self.children:receive()
   outputsPartial = self.children:receive()
   -- force cleanup
   collectgarbage()
end

function GenSGD:reduce_hook()
   local id = 0
   local mx = 1e9
   for t = 1,P do
      if outputsPartial[t].f_x < mx then
         id = t
         mx = outputsPartial[t].f_x
      end
   end
   if id == 0 then
      print('ERROR: diverging')
   else
      self.baseParameters = outputsPartial[id]
      self.output = self.currentParameters.f_x
      -- in this case we get the parameters back directly
      self.parameters:copy(gradParametersPartial[id])
      print('Winner: output = '..self.output..
            'learningRate = '..self.baseParameters['learningRate'])
   end
end

function GenSGD:optimize()
   self.evaluate()
end

-- optimization (could do others in this mode)
function GenSGD:optimizer(module,params)
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

function GenSGD:worker_code()
   -- require packages
   require 'nnx'
       
   -- retrieve module + criterion at startup
   parallel.yield()

   module    = parallel.parent:receive()
   criterion = parallel.parent:receive()
   optimizer = parallel.parent:receive()

   module.parameters     = nnx.flattenParameters(nnx.getParameters(module))
   module.gradParameters = nnx.flattenParameters(nnx.getGradParameters(module))
	    
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

function GenSGD:setup()
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

