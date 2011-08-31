local BatchTrainer, parent = torch.class('nn.BatchTrainer', 'nn.OnlineTrainer')

-- Essentially simialar to the OnlineTrainer but only used the parts
-- of the code which prepare the data and the tester. train() has been
-- replaced by nextBatch() which moves the trainer one batch further
-- in the data.  When the first epoch is finished then the batches are
-- reused.  Each call to optimizer.forward() in nextBatch() creates a
-- closure with the current batch as input.

function BatchTrainer:__init(...)
   -- parent.__init(self)
   -- unpack args
   xlua.unpack_class(self, {...},
      'BatchTrainer', 

      'A general-purpose online trainer class.\n'
         .. 'Provides 4 user hooks to perform extra work after each sample, or each epoch:\n'
         .. '> trainer = nn.BatchTrainer(...) \n'
         .. '> trainer.hookTrainSample = function(trainer, sample) ... end \n'
         .. '> trainer.hookTrainEpoch = function(trainer) ... end \n'
         .. '> trainer.hookTestSample = function(trainer, sample) ... end \n'
         .. '> trainer.hookTestEpoch = function(trainer) ... end \n'
         .. '> ',

      {arg='trainset', type='nn.DataList',
       help='dataset to split into batches for closures',req=true},
      {arg='module', type='nn.Module', help='a module to train', req=true},
      {arg='criterion', type='nn.Criterion', 
       help='a criterion to estimate the error'},
      {arg='preprocessor', type='nn.Module', 
       help='a preprocessor to prime the data before the module'},
      {arg='optimizer', type='nn.Optimization', 
       help='an optimization method'}, 
      {arg='batchSize', type='number', 
       help='[mini] batch size', default=1},
      {arg='maxEpoch', type='number', 
       help='maximum number of epochs', default=50},
      {arg='dispProgress', type='boolean', 
       help='display a progress bar during training/testing', default=true},
      {arg='save', type='string', 
       help='path to save networks and log training'},
      {arg='timestamp', type='boolean', 
       help='if true, appends a timestamp to each network saved', default=false}
   )
   -- private params
   self.trainOffset = -self.batchSize
   self.testOffset = 0

   -- counters
   self.epoch = 1
   self.batch = 0
end

-- update the counters
function BatchTrainer:next()
   self.batch = self.batch + 1
   self.trainOffset = self.trainOffset + self.batchSize
   if self.trainOffset > self.trainset:size()-1 then
      self.trainOffset = 1
      self.epoch = self.epoch + 1
      self.batch = 1
   end
   -- disp progress
   if self.dispProgress then
      xlua.progress(self.trainOffset, trainset:size())
   end

end

-- this function is called train() in the online trainer.  I seems to
-- make more sense to call it next_batch() here as the training is
-- done outside of this code.

function BatchTrainer:next_batch()
   self.next()
   local module = self.module
   local criterion = self.criterion
   local t = self.trainOffset
   local ds = self.trainset:size()
   local bs = self.batchSize
   
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. self.epoch .. ' batch # '..self.batch.. '[batchSize = ' .. self.batchSize .. ']')

   -- create mini batch
   self.inputs = self.inputs or {}
   self.targets = self.targets or {}
   local inputs = {}
   local targets = {}
   if not self.inputs[self.batch] then

      self.inputs[self.batch] = {}
      inputs = self.inputs[self.batch] 
      self.targets[self.batch] = {}
      targets = self.targets[self.batch]

      for i = t,math.min(t+bs-1,ds) do
	 -- load new sample
	 local sample = trainset[t + i]
	 local input = sample[1]
	 local target = sample[2]
      
	 -- optional preprocess (no learning is done for that guy)
	 if self.preprocessor then input = self.preprocessor:forward(input) end
	 
      -- store input/target
	 table.insert(inputs, input)
	 table.insert(targets, target)
      end
   else  
      -- get batch from cache
      inputs = self.inputs[self.batch] 
      targets = self.targets[self.batch]
   end   

   -- set up closure batch.evaluate() for optimizer
   local error = self.optimizer:forward(inputs, targets)
end


