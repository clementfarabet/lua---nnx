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
------------------------------------------------------------------------
local Recurrent, parent = torch.class('nn.Recurrent', 'nn.Module')

function Recurrent:__init(start, input, feedback, transfer, merge)
   parent.__init(self)

   local ts = torch.type(start)
   if ts == 'torch.LongTensor' or ts == 'number' then
      start = nn.Add(start)
   end
   self.startModule = start
   self.inputModule = input
   self.feedbackModule = feedback
   self.transferModule = transfer or nn.Sigmoid()
   self.mergeModule = merge or nn.CAddTable()
   
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
   self.recurrentModule:add(self.mergeModule)
   self.recurrentModule:add(self.transferModule)
   
   self.modules = {self.startModule, self.inputModule, self.feedbackModule, self.transferModule, self.mergeModule}
   
   self.initialOutputs = {}
   self.initialGradInputs = {}
   self.recurrentOutputs = {}
   self.recurrentGradInputs = {}
   
   self.copyInputs = true
   self.inputs = {}
   self.outputs = {}
   self.gradOutputs = {}
   self.scales = {}
   
   self.step = 1
   
   self:reset()
end

local function recursiveClone(t)
   local clone
   if torch.type(t) == 'table' then
      clone = {}
      for i = 1, #t do
         clone[i] = recursiveClone(t[i])
      end
   else
      if torch.typename(t) and 
        torch.typename(t):find('torch%..+Tensor') then
         clone = t:clone()
      end
   end
   return clone
end

function Recurrent:updateOutput(input)
   -- output(t) = transfer(feedback(output_(t-1)) + input(input_(t)))
   local output
   if self.step == 1 then
      -- set/save the output states
      local modules = self.initialModule:listModules()
      for i,modula in ipairs(modules) do
         local output_ = self.initialOutputs[i] or recursiveClone(modula.output)
         modula.output = output_
      end
      output = self.initialModule:updateOutput(input)
      for i,modula in ipairs(modules) do
         self.initialOutputs[i]  = modula.output
      end
   else
      -- set/save the output states
      local modules = self.recurrentModule:listModules()
      local recurrentOutputs = self.recurrentOutputs[self.step]
      if not recurrentOutputs then
         recurrentOutputs = {}
         self.recurrentOutputs[self.step] = recurrentOutputs
      end
      for i,modula in ipairs(modules) do
         local output_ = recurrentOutputs[i] or recursiveClone(modula.output)
         modula.output = output_
      end
      -- self.output is the previous output of this module
      output = self.recurrentModule:updateOutput{input, self.output}
      for i,modula in ipairs(modules) do
         recurrentOutputs[i]  = modula.output
      end
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
   self.outputs[self.step] = output
   self.output:set(output)
   self.step = self.step + 1
   return self.output
end

function Recurrent:updateGradInput(input, gradOutput)
   -- Back-Propagate Through Time (BPTT) happens in updateParameters()
   -- for now we just keep a list of the gradOutputs
   local gradOutput_ = self.gradOutputs[self.step-1] 
   if not gradOutput_ then
      gradOutput_ = recursiveClone(gradOutput)
      self.gradOutputs[self.step-1] = gradOutput_
   end
   gradOutput_:copy(gradOutput)
end

function Recurrent:accGradParameters(input, gradOutput, scale)
   -- Back-Propagate Through Time (BPTT) happens in updateParameters()
   -- for now we just keep a list of the scales
   self.scales[self.step-1] = scale
end

-- not to be confused with the hit movie Back to the Future
function Recurrent:backwardThroughTime()
   local gradInput
   for step=self.step-1,2,-1 do
      -- set the output/gradOutput states of current Module
      local modules = self.recurrentModule:listModules()
      local recurrentOutputs = self.recurrentOutputs[step]
      local recurrentGradInputs = self.recurrentGradInputs[step]
      if not recurrentGradInputs then
         recurrentGradInputs = {}
         self.recurrentGradInputs[step] = recurrentGradInputs
      end
      for i,modula in ipairs(modules) do
         local output, gradInput = modula.output, modula.gradInput
         local output_ = recurrentOutputs[i]
         local gradInput_ = recurrentGradInputs[i] or recursiveClone(gradInput)
         assert(output_, "backwardThroughTime requires previous updateOutput")
         modula.output = output_
         modula.gradInput = gradInput_
      end
      
      -- backward propagate through this step
      local input = self.inputs[step]
      local output = self.outputs[step-1]
      local gradOutput = self.gradOutputs[step]
      if gradInput then
         gradOutput:add(gradInput)
      end
      local scale = self.scales[step]
      gradInput = self.recurrentModule:backward({input, output}, gradOutput, scale/(self.step-1))[2]
      for i,modula in ipairs(modules) do
         recurrentGradInputs[i] = modula.gradInput
      end
   end
   
   -- set the output/gradOutput states of initialModule
   local modules = self.initialModule:listModules()
   for i,modula in ipairs(modules) do
      local output, gradInput = modula.output, modula.gradInput
      local output_ = self.initialOutputs[i]
      local gradInput_ = self.initialGradInputs[i] or recursiveClone(gradInput)
      modula.output = output_
      modula.gradInput = gradInput_
   end
   
   -- backward propagate through first step
   local input = self.inputs[1]
   local gradOutput = self.gradOutputs[1]
   if gradInput then
      gradOutput:add(gradInput)
   end
   local scale = self.scales[1]
   gradInput = self.initialModule:backward(input, gradOutput, scale/(self.step-1))
   
   for i,modula in ipairs(modules) do
      self.initialGradInputs[i] = modula.gradInput
   end
   
   -- startModule's gradParams shouldn't be step-averaged
   -- as it is used only once. So un-step-average it
   local params, gradParams = self.startModule:parameters()
   if gradParams then
      for i,gradParam in ipairs(gradParams) do
         gradParam:mul(self.step-1)
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
