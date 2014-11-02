------------------------------------------------------------------------
--[[ Recurrent ]]--
-- Ref. A.: http://goo.gl/vtVGkO (Mikolov et al.)
-- B. http://goo.gl/hu1Lqm
-- Processes the sequence one timestep (forward/backward) at a time. 
-- A call to backward only keeps a log of the gradOutputs and scales.
-- Back-Propagation Through Time (BPTT) is done when updateParameters
-- is called. The Module keeps a list of all previous representations 
-- (Module.outputs), including intermediate ones for BPTT.
-- To use this module with batches, we suggest using different 
-- sequences of the same size within a batch and calling 
-- updateParameters() at the end of the Sequence. 
-- TODO :
-- make this work with :representations()
------------------------------------------------------------------------
local Recurrent, parent = torch.class('nn.Recurrent', 'nn.Module')

function Recurrent:__init(start, input, feedback, transfer)
   parent.__init(self)

   local ts = torch.type(start)
   if ts == 'torch.LongTensor' or ts == 'number' then
      start = nn.Add(start)
   end
   self.startModule = start
   self.inputModule = input
   self.feedbackModule = feedback
   self.transferModule = transfer or nn.Sigmoid()
   
   -- used for the first step 
   self.initialModule = nn.Sequential()
   self.initialModule:add(self.inputModule)
   self.initialModule:add(self.startModule)
   self.initialModule:add(self.transferModule)
   
   -- used for the other steps (steps > 1)
   local parallelModule = nn.ParallelTable()
   parallelModule:add(self.inputModule)
   parallelModule:add(self.feedbackModule)
   self.recurrentModule = nn.Sequential()
   self.recurrentModule:add(parallelModule)
   self.recurrentModule:add(nn.CAddTable())
   self.recurrentModule:add(self.transferModule)
   
   self.modules = {self.startModule, self.inputModule, self.recurrentModule, self.transferModule}
   
   self.initialState = {}
   self.initialGradState = {}
   self.recurrentStates = {}
   self.recurrentGradStates = {}
   
   self.copyInputs = true
   self.inputs = {}
   self.outputs = {}
   self.gradOutputs = {}
   self.scales = {}
   
   self.step = 1
   
   self:reset()
end

function Recurrent:updateOutput(input)
   -- output(t) = transfer(feedback(output_(t-1)) + input(input_(t)))
   if self.step == 1 then
      -- set/save the output states
      local outputs = self.initialModule:representations()
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
      local outputs = self.recurrentModule:representations()
      local recurrentState = self.recurrentStates[self.step]
      if not recurrentState then
         recurrentState = {}
         self.recurrentStates[self.step] = recurrentState
      end
      for i=1,#outputs do
         local output = outputs[i]
         local output_ = recurrentState[i]
         if not output_ then
            output_ = output:clone()
            recurrentState[i] = output_
         end
         output:set(output_)
      end
      -- self.output is the previous output of this module
      local output = self.recurrentModule:updateOutput{input, self.output}
      self.output:set(output)
   end
   local input_ = self.inputs[self.step]
   if not input_ then
      input_ = input.new()
      self.inputs[self.step] = input_
   end
   if self.copyInputs then
      input_:resizeAs(input):copy(input)
   else
      input_:set(input)
   end
   self.outputs[self.step] = self.output
   self.step = self.step + 1
   return self.output
end

function Recurrent:updateGradInput(input, gradOutput)
   -- Back-Propagate Through Time (BPTT) happens in updateParameters()
   -- for now we just keep a list of the gradOutputs
   self.gradOutputs[self.step-1] = gradOutput 
   return self.gradInput
end

function Recurrent:accGradParameters(input, gradOutput, scale)
   -- Back-Propagate Through Time (BPTT) happens in updateParameters()
   -- for now we just keep a list of the scales
   self.scales[self.step-1] = scale
end

-- not to be confused with the hit movie Back to the Future
function Recurrent:backwardThroughTime()
   local gradInput
   for step=self.step-1,-1,2 do
      -- set the output/gradOutput states of current Module
      local outputs, gradInputs = self.recurrentModule:representations()
      local recurrentState = self.recurrentStates[step]
      local recurrentGradState = self.recurrentGradStates[step]
      if not recurrentGradState then
         recurrentGradState = {}
         self.recurrentGradStates[step] = recurrentGradState
      end
      for i=1,#outputs do
         local output, gradInput = outputs[i], gradInputs[i]
         local output_, gradInput_ = recurrentState[i], recurrentGradState[i]
         if not gradInput_ then
            gradInput_ = gradInput:clone()
            recurrentGradState[i] = gradInput_
         end
         output:set(output_)
         gradInput:set(gradInput_)
      end
      
      -- backward propagate through this step
      local input = self.inputs[step]
      local output = self.outputs[step-1]
      local gradOutput = self.gradOutputs[step]
      if gradInput then
         gradOutput:add(gradInput)
      end
      local scale = self.scales[step]
      gradInput = self.recurrentModule:backward({input, output}, gradOutput, scale/(self.step-1))
   end
   
   -- set the output/gradOutput states of initialModule
   local outputs, gradInputs = self.initialModule:representations()
   for i=1,#outputs do
      local output, gradInput = outputs[i], gradInputs[i]
      local output_, gradInput_ = self.initialState[i], self.initialGradState[i]
      if not gradInput_ then
         gradInput_ = gradInput:clone()
         self.initialGradState[i] = gradInput_
      end
      output:set(output_)
      gradInput:set(gradInput_)
   end
   
   -- backward propagate through first step
   local input = self.inputs[1]
   local gradOutput = self.gradOutputs[1]
   if gradInput then
      gradOutput:add(gradInput)
   end
   local scale = self.scales[1]
   gradInput = self.initialModule:backward(input, gradOutput, scale/(self.step-1))
   
   -- startModule's gradParams shouldn't be step-averaged
   -- as it is used only once. So un-step-average it
   local params, gradParams = self.startModule:parameters()
   if gradParams then
      for i,gradParam in ipairs(gradParams) do
         gradParams:mul(self.step-1)
      end
   end
   self.step = 1
   return gradInput
end

function Recurrent:updateParameters(learningRate)
   self:backwardThroughTime()
   parent.updateParameters(self, learningRate)
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
