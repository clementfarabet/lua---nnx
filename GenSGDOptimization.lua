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
       help='learning rate decay (lr_t = lr_0 / (1 + samplesSeen*lrDecay))', default=0},
      {arg='weightDecay', type='number', 
       help='amount of weight decay (W = W - decay*W)', default=0},
      {arg='momentum', type='number', 
       help='amount of momentum on weights (dE/W = dE/dW*(1-momentum) + prev(dE/dW)*momentum)', default=0}
   )
   if self.parallelize < 2 then
      print('ERROR: GenSGD needs to work on several processors')
   end
   -- change the mapper to send the same batch to each worker
   self.copyBatch = true
   self.currentLearningRate = learningRate
   self.workerRates = torch.Tensor(self.P)
end

function GenSGD:map_hook()
end

function GenSGD:reduce_hook()
end

function GenSGD:optimize()
   self.evaluate()
end


function GenSGD:worker_code()
   -- require packages
   require 'nnx'
       
   -- retrieve module + criterion at startup
   parallel.yield()
   module    = parallel.parent:receive()
   criterion = parallel.parent:receive()
   optimizer = parallel.parent:receive()

   parameters     = nnx.flattenParameters(nnx.getParameters(self.module))
   gradParameters = nnx.flattenParameters(nnx.getGradParameters(self.module))
	    
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
	 
	 -- f is the average of all criterions
	 local f_x = 0
	 -- evaluate gradients on inputs for this thread
	 for i = 1,#inputs do
	    -- reset gradients
	    gradParameters:zero()
	    -- estimate f
	    local output = module:forward(inputs[i])
	    local err = criterion:forward(output, targets[i])
	    f_x = f_x + err
	    -- estimate df/dW
	    local df_do = criterion:backward(output, targets[i])
	    module:backward(inputs[i], df_do)
	    module:accGradParameters(inputs[i], df_do)
		     optimizer
		     
	 end
	 -- now send back parameters b/c they are already optimized 
	 parallel.parent:send(parameters)
	 parallel.parent:send(f_x)
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

function GenSGD:post_hook(module,options)
   -- we do the SGD on the worker
   -- apply momentum
   if options.momentum ~= 0 then
      if not module.currentGradParameters then
	 module.currentGradParameters = torch.Tensor():resizeAs(gradParameters):copy(gradParameters)
      else
	 options.currentGradParameters:mul(options.momentum):add(1-options.momentum, gradParameters)
      end
   else
      options.currentGradParameters = gradParameters
   end
   
   -- weight decay
   if options.weightDecay ~= 0 then
      options.parameters:add(-options.weightDecay, options.parameters)
   end
   
   -- update parameters
   local learningRate = self.learningRate / 
      (1 + self.sampleCounter*self.learningRateDecay)
   self.parameters:add(-learningRate, self.currentGradParameters)
end
