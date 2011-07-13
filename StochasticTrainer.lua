local StochasticTrainer, parent = torch.class('nn.StochasticTrainer','nn.Trainer')

function StochasticTrainer:__init(...)
   parent.__init(self)
   -- unpack args
   xlua.unpack_class(self, {...},
      'StochasticTrainer', 

      'A general-purpose stochastic trainer class.\n'
         .. 'Provides 4 user hooks to perform extra work after each sample, or each epoch:\n'
         .. '> trainer = nn.StochasticTrainer(...) \n'
         .. '> trainer.hookTrainSample = function(trainer, sample) ... end \n'
         .. '> trainer.hookTrainEpoch = function(trainer) ... end \n'
         .. '> trainer.hookTestSample = function(trainer, sample) ... end \n'
         .. '> trainer.hookTestEpoch = function(trainer) ... end \n'
         .. '> ',

      {arg='module', type='nn.Module', help='a module to train', req=true},
      {arg='criterion', type='nn.Module', help='a criterion to estimate the error'},
      {arg='preprocessor', type='nn.Module', help='a preprocessor to prime the data before the module'},

      {arg='learningRate', type='number', help='learning rate (W = W - rate*dE/dW)', default=1e-2},
      {arg='learningRateDecay', type='number', help='learning rate decay (rate = rate * (1-decay), at each epoch)', default=0},
      {arg='weightDecay', type='number', help='amount of weight decay (W = W - decay*W)', default=0},
      {arg='momentum', type='number', help='amount of momentum on weights (dE/W = dE/dW + momentum*prev(dE/dW))', default=0},
      {arg='maxEpoch', type='number', help='maximum number of epochs', default=50},

      {arg='maxTarget', type='boolean', help='replaces an CxHxW target map by a HxN target of max values (for NLL criterions)', default=false},
      {arg='dispProgress', type='boolean', help='display a progress bar during training/testing', default=true},
      {arg='skipUniformTargets', type='boolean', help='skip uniform (flat) targets during training', default=false},

      {arg='save', type='string', help='path to save networks and log training'},
      {arg='timestamp', type='boolean', help='if true, appends a timestamp to each network saved', default=false}
   )
   -- private params
   self.errorArray = self.skipUniformTargets
   self.trainOffset = 0
   self.testOffset = 0
end

function StochasticTrainer:log()
   -- save network
   local filename = self.save
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if self.timestamp then
      -- use a timestamp to store all networks uniquely
      filename = filename .. '-' .. os.date("%Y_%m_%d_%X")
   else
      -- if no timestamp, just store the previous one
      if sys.filep(filename) then
         os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
      end
   end
   print('<trainer> saving network to '..filename)
   local file = torch.DiskFile(filename,'w')
   nnx.empty(self.module)
   self.module:write(file)
   file:close()
end

function StochasticTrainer:train(dataset)
   self.epoch = self.epoch or 1
   local currentLearningRate = self.learningRate
   local module = self.module
   local criterion = self.criterion
   self.trainset = dataset

   local shuffledIndices = {}
   if not self.shuffleIndices then
      for t = 1,dataset:size() do
         shuffledIndices[t] = t
      end
   else
      shuffledIndices = lab.randperm(dataset:size())
   end
   
   while true do
      print('<trainer> on training set:')
      print("<trainer> stochastic gradient descent epoch # " .. self.epoch)

      module:zeroGradParameters()

      self.time = sys.clock()
      self.currentError = 0
      for t = 1,dataset:size() do
         -- disp progress
         if self.dispProgress then
            xlua.progress(t, dataset:size())
         end

         -- load new sample
         local sample = dataset[self.trainOffset + shuffledIndices[t]]
         local input = sample[1]
         local target = sample[2]
         local sample_x = sample.x
         local sample_y = sample.y

         -- get max of target ?
         if self.maxTarget then
            target = torch.Tensor(target:nElement()):copy(target)
            _,target = lab.max(target)
            target = target[1]
         end

         -- is target uniform ?
         local isUniform = false
         if self.errorArray and target:min() == target:max() then
            isUniform = true
         end

         -- perform SGD step
         if not (self.skipUniformTargets and isUniform) then
            -- optional preprocess
            if self.preprocessor then input = self.preprocessor:forward(input) end

            -- forward through model and criterion 
            -- (if no criterion, it is assumed to be contained in the model)
            local modelOut, error
            if criterion then
               modelOut = module:forward(input)
               error = criterion:forward(modelOut, target)
            else
               modelOut, error = module:forward(input, target, sample_x, sample_y)
            end

            -- accumulate error
            self.currentError = self.currentError + error

            -- backward through model
            -- (if no criterion, it is assumed that derror is internally generated)
            module:zeroGradParameters(self.momentum)
            if criterion then
               local derror = criterion:backward(module.output, target)
               module:backward(input, derror)
            else
               module:backward(input)
            end

            -- weight decay ?
            if self.weightDecay ~= 0 and module.decayParameters then
               module:decayParameters(self.weightDecay)
            end

            -- update parameters in the model
            module:updateParameters(currentLearningRate)
         end

         -- call user hook, if any
         if self.hookTrainSample then
            self.hookTrainSample(self, sample)
         end
      end

      self.currentError = self.currentError / dataset:size()
      print("<trainer> current error = " .. self.currentError)

      self.time = sys.clock() - self.time
      self.time = self.time / dataset:size()
      print("<trainer> time to learn 1 sample = " .. (self.time*1000) .. 'ms')

      if self.hookTrainEpoch then
         self.hookTrainEpoch(self)
      end

      if self.save then self:log() end

      self.epoch = self.epoch + 1
      currentLearningRate = self.learningRate/(1+self.epoch*self.learningRateDecay)

      if self.maxEpoch > 0 and self.epoch > self.maxEpoch then
         print("<trainer> you have reached the maximum number of epochs")
         break
      end

      if dataset.infiniteSet then
         self.trainOffset = self.trainOffset + dataset:size()
      end
   end
end


function StochasticTrainer:test(dataset)
   print('<trainer> on testing Set:')

   local module = self.module
   local shuffledIndices = {}
   local criterion = self.criterion
   self.currentError = 0
   self.testset = dataset

   local shuffledIndices = {}
   if not self.shuffleIndices then
      for t = 1,dataset:size() do
         shuffledIndices[t] = t
      end
   else
      shuffledIndices = lab.randperm(dataset:size())
   end
   
   self.time = sys.clock()
   for t = 1,dataset:size() do
      -- disp progress
      if self.dispProgress then
         xlua.progress(t, dataset:size())
      end

      -- get new sample
      local sample = dataset[self.testOffset + shuffledIndices[t]]
      local input = sample[1]
      local target = sample[2]

      -- max target ?
      if self.maxTarget then
         target = torch.Tensor(target:nElement()):copy(target)
         _,target = lab.max(target)
         target = target[1]
      end
      
      -- test sample through current model
      if self.preprocessor then input = self.preprocessor:forward(input) end
      if criterion then
         self.currentError = self.currentError + 
	    criterion:forward(module:forward(input), target)
      else
         local _,error = module:forward(input, target)
         self.currentError = self.currentError + error
      end

      -- user hook
      if self.hookTestSample then
         self.hookTestSample(self, sample)
      end
   end

   self.currentError = self.currentError / dataset:size()
   print("<trainer> test current error = " .. self.currentError)

   self.time = sys.clock() - self.time
   self.time = self.time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (self.time*1000) .. 'ms')

   if self.hookTestEpoch then
      self.hookTestEpoch(self)
   end

   if dataset.infiniteSet then
      self.testOffset = self.testOffset + dataset:size()
   end

   return self.currentError
end

function StochasticTrainer:write(file)
   parent.write(self,file)
   file:writeObject(self.module)
   file:writeObject(self.criterion)
end

function StochasticTrainer:read(file)
   parent.read(self,file)
   self.module = file:readObject()
   self.criterion = file:readObject()
end
