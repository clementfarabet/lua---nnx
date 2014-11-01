------------------------------------------------------------------------
--[[ Recurrent ]]--
-- Ref. A.: http://goo.gl/vtVGkO (Mikolov et al.)
-- B. http://goo.gl/hu1Lqm
-- Truncated BPTT.
-- It processes the sequence one timestep at a time. 
-- A call to backward only backward propagates the output Module.
-- Back-Propagation Through Time (BPTT) is done when updateParameters
-- is called. The Module keeps a list and copy of all previous 
-- representations (Module.outputs), including intermediate ones 
-- for BPTT.
-- The use this module with batches, we suggest using different 
-- sequences of the same size within a batch and calling 
-- updateParameters() at the end of the Sequence. 
------------------------------------------------------------------------
local Recurrent, parent = torch.class('nn.Recurrent', 'nn.Module')

function Recurrent:__init(input, feedback, transfer, output, hiddenSize)
   parent.__init(self)

   self.hiddenSize = hiddenSize
   self.inputModule = input
   self.feedbackModule = feedback
   self.transferModule = transfer
   self.outputModule = output
   
   self.modules = {inputModule, feedbackModule, transferModule, outputModule} --add nn.Add()?
   
   -- used for forward propagations only
   local parallelModule = nn.ParallelTable()
   parallelModule:add(self.feedbackModule)
   parallelModule:add(self.inputModule)
   self._recurrentModule = nn.Sequential()
   self._recurrentModule:add(parallelModule)
   self._recurrentModule:add(nn.CAddTable())
   self._recurrentModule:add(self.transferModule)
   self.recurrentModule = nn.Sequential()
   self.recurrentModule:add(self._recurrentModule)
   self.recurrentModule:add(self.outputModule)
   
   -- used for the first step in sequence (replaces the recurrence)
   self._initialModule = nn.Sequential()
   self._initialModule:add(self.inputModule)
   self._initialModule:add(nn.Add(self.hiddenSize))
   self._initialModule:add(self.transferModule)
   self.initialModule = nn.Sequential()
   self.initialModule:add(self._initialModule)
   self.initialModule:add(self.outputModule)
   
   self.initialState = {}
   self.initialGradState = {}
   self.recurrentStates = {}
   self.recurrentGradStates = {}
   
   self.gradPrestates {}
   self.gradStates = {}
   
   self.step = 1
   
   self:reset()
end

function Recurrent:updateOutput(input)
   -- output(t) = transfer(feedback(output_(t-1)) + input(input_(t)))
   if self.step == 1 then
      -- set/save the output states
      local outputs = self._initialModule:representations()
      for i=1,#outputs do
         local output = outputs[i]
         local output_ = self.initialState[i]
         if not output_ then
            output_ = output:clone()
            self.initialState[i] = output_
         end
         output:set(output_)
      end
      local output = self.initialModule:updateOutput(input)
      self.output:set(output)
   else
      -- set/save the output states
      local outputs = self._recurrentModule:representations()
      for i=1,#outputs do
         local output = outputs[i]
         local recurrentState = self.recurrentStates[self.step]
         if not recurrentState then
            recurrentState = {}
            self.recurrentStates[self.step] = recurrentState
         end
         local output_ = recurrentState[i]
         if not output_ then
            output_ = output:clone()
            reccurentState[i] = output_
         end
         output:set(output_)
      end
      local output = self.recurrentModule:updateOutput(input)
      self.output:set(output)
   end
   return self.output
end

function Recurrent:updateGradInput(input, gradOutput)
   -- Back-Propagate Through Time (BPTT)
   return self.gradInput
end

function Recurrent:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
end

function Recurrent:size()
   return #self.modules
end

function Recurrent:get(index)
   return self.modules[index]
end

function Recurrent:zeroGradParameters()
  for i=1,#self.modules do
     self.modules[i]:zeroGradParameters()
  end
end

function Recurrent:updateParameters(learningRate)
   for i=1,#self.modules do
      self.modules[i]:updateParameters(learningRate)
   end
end

function Recurrent:training()
   for i=1,#self.modules do
      self.modules[i]:training()
   end
end

function Recurrent:evaluate()
   for i=1,#self.modules do
      self.modules[i]:evaluate()
   end
end

function Recurrent:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end

function Recurrent:reset(stdv)
   for i=1,#self.modules do
      self.modules[i]:reset(stdv)
   end
end

function Recurrent:parameters()
   local function tinsert(to, from)
      if type(from) == 'table' then
         for i=1,#from do
            tinsert(to,from[i])
         end
      else
         table.insert(to,from)
      end
   end
   local w = {}
   local gw = {}
   for i=1,#self.modules do
      local mw,mgw = self.modules[i]:parameters()
      if mw then
         tinsert(w,mw)
         tinsert(gw,mgw)
      end
   end
   return w,gw
end

function Recurrent:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'nn.Recurrent'
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self.modules do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self.modules do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end


--[[
Recurrent(input, recurrence, transfer, output)
local outputLayer = nn.Recurrent():add(nn.Linear()):add(nn.SoftMax())
r = nn.Recurrent(nn.LookupTable(), nn.Linear(), nn.Sigmoid(), outputLayer)

local i = 1
local wordIndice = torch.LongTensor{0,10000,200000,90000000}
while true do
   local input = text:index(1, wordIndice)
   local output = r:forward(input)
   increment(wordIndice)
   local target = text:index(1, wordIndice)
   local err = criterion:forward(output, target)
   local gradOutput = criterion:backward(output, target)
   -- only backpropagates through outputLayer
   -- and memorizes these gradOutputs
   r:backward(input, gradOutput)
   i = i + 1
   if i % rho then
      -- backpropagates through time (BPTT), 
      -- i.e. through recurrence and input layer
      r:updateParameters(lr)
   end
end
--]]
