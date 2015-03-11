------------------------------------------------------------------------
--[[ LSTM ]]--
-- Long Short Term Memory architecture.
-- Ref. A.: http://arxiv.org/pdf/1303.5778v1 (blueprint for this module)
-- B. http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf
-- Expects 1D or 2D input.
-- The first input in sequence uses zero value for cell and hidden state
------------------------------------------------------------------------
local LSTM, parent = torch.class('nn.LSTM', 'nn.AbstractRecurrent')

function LSTM:__init(inputSize, outputSize)
   parent.__init(self)
   self.inputSize = inputSize
   self.outputSize = outputSize   
   -- build the model
   self.model = self:buildModel()
   -- so it works with container
   self.modules[1] = self.model
end

-------------------------- factory methods -----------------------------
function LSTM:buildGate()
   -- Note : inputGate:forward expects an input table : {input, output, cell}
   local gate = nn.Sequential()
   local input2gate = nn.Linear(self.inputSize, self.outputSize)
   local cell2gate = nn.CMul(self.outputSize) -- diagonal cell to gate weight matrix
   local output2gate = nn.Linear(self.outputSize, self.outputSize)
   output2gate:noBias() --TODO
   local para = nn.ParallelTable()
   para:add(input2gate):add(cell2gate):add(output2gate)
   gate:add(para)
   gate:add(nn.CAddTable())
   gate:add(nn.Sigmoid())
   return gate
end

function LSTM:buildInputGate()
   return self:buildGate()
end

function LSTM:buildForgetGate()
   return self:buildGate()
end

function LSTM:buildHidden()
   local hidden = nn.Sequential()
   local input2hidden = nn.Linear(self.inputSize, self.outputSize)
   local output2hidden = nn.Linear(self.outputSize, self.outputSize) 
   local para = nn.ParallelTable()
   output2hidden:noBias()
   para:add(input2hidden):add(output2hidden)
   -- input is {input, output, cell}, but we only need {input, output}
   local concat = nn.ConcatTable()
   concat:add(nn.SelectTable(1):add(nn.SelectTable(2))
   hidden:add(concat)
   hidden:add(para)
   hidden:add(nn.CAddTable())
   hidden:add(nn.Tanh())
   return hidden
end

function LSTM:buildCell()
   -- build
   self.inputGate = self:buildInputGate() 
   self.forgetGate = self:buildForgetGate()
   self.hiddenLayer = self:buildHidden()
   -- forget = forgetGate{input, output, cell} * cell
   local forget = nn.Sequential()
   local concat = nn.ConcatTable()
   concat:add(self.forgetGate):add(self.SelectTable(3))
   forget:add(concat)
   forget:add(nn.CMulTable())
   -- input = inputGate{input, output, cell} * hiddenLayer{input, output, cell}
   local input = nn.Sequential()
   local concat2 = nn.ConcatTable()
   concat2:add(self.inputGate):add(self.hiddenLayer)
   input:add(concat2)
   input:add(nn.CMulTable())
   -- cell = forget + input
   local cell = nn.Sequential()
   local concat3 = nn.ConcatTable()
   concat3:add(forget):add(input)
   cell:add(concat3)
   cell:add(nn.CAddTable())
end   
   
function LSTM:buildOutputGate()
   return self:buildGate()
end

-- cell(t) = cellLayer{input, output, cell(t-1)}
-- output = outputGate{input, output, cell(t)}*tanh(cell(t))
function LSTM:buildModel()
   -- build components
   self.cellLayer = self:buildCell()
   self.outputGate = self:buildOutputGate()
   -- assemble
   local concat = nn.ConcatTable()
   local concat2 = nn.ConcatTable()
   concat2:add(nn.SelectTable(1):add(nn.SelectTable(2))
   concat:add(concat2):add(self.cellLayer)
   local model = nn.Sequential()
   model:add(concat2)
   -- output of concat is {{input, output}, cell(t)}, 
   -- so flatten to {input, output, cell(t)}
   model:add(nn.FlattenTable())
   local cellAct = nn.Sequential()
   cellAct:add(nn.Select(3))
   cellAct:add(nn.Tanh())
   local concat3 = nn.ConcatTable()
   concat3:add(self.outputGate):add(cellAct)
   model:add(concat3)
   model:add(nn.CMulTable())
   return model
end

function LSTM:updateOutput(input)
   local prevOutput, prevCell
   if self.step == 1 then
      prevOutput = self.startOutput
      prevCell = self.startCell
   else
      prevOutput = self.output
      prevCell = self.cell
   end
   if self.train ~= false then
      self.output = self.model:updateOutput{input, self.output, self.cell}
      self.cell = self.cellLayer.output
      
      -- set/save the output states
      local modules = self.recurrentModel:listModules()
      self:recycle()
      local recurrentOutputs = self.recurrentOutputs[self.step]
      if not recurrentOutputs then
         recurrentOutputs = {}
         self.recurrentOutputs[self.step] = recurrentOutputs
      end
      for i,modula in ipairs(modules) do
         local output_ = recursiveResizeAs(recurrentOutputs[i], modula.output)
         modula.output = output_
      end
       -- self.output is the previous output of this module
      output = self.recurrentModule:updateOutput{input, self.output}
      for i,modula in ipairs(modules) do
         recurrentOutputs[i]  = modula.output
      end
   else
      -- self.output is the previous output of this module
      output = self.recurrentModule:updateOutput{input, self.output}
   end
   return self.output
end

------------------------- forward backward -----------------------------
function LSTM:updateOutput(input)
   -- output(t) = transfer(feedback(output_(t-1)) + input(input_(t)))
   local output
   if self.train ~= false then
      -- set/save the output states
      local modules = self.recurrentModule:listModules()
      self:recycle()
      local recurrentOutputs = self.recurrentOutputs[self.step]
      if not recurrentOutputs then
         recurrentOutputs = {}
         self.recurrentOutputs[self.step] = recurrentOutputs
      end
      for i,modula in ipairs(modules) do
         local output_ = recursiveResizeAs(recurrentOutputs[i], modula.output)
         modula.output = output_
      end
       -- self.output is the previous output of this module
      output = self.recurrentModule:updateOutput{input, self.output}
      for i,modula in ipairs(modules) do
         recurrentOutputs[i]  = modula.output
      end
   else
      -- self.output is the previous output of this module
      output = self.recurrentModule:updateOutput{input, self.output}
   end
   
   if self.train ~= false then
      local input_ = self.inputs[self.step]
      self.inputs[self.step] = self.copyInputs 
         and self.recursiveCopy(input_, input) 
         or self.recursiveSet(input_, input)     
   end
   
   self.outputs[self.step] = output
   self.output = output
   self.step = self.step + 1
   self.gradParametersAccumulated = false
   return self.output
end

function LSTM:updateGradInput(input, gradOutput)
   -- Back-Propagate Through Time (BPTT) happens in updateParameters()
   -- for now we just keep a list of the gradOutputs
   self.gradOutputs[self.step-1] = self.recursiveCopy(self.gradOutputs[self.step-1] , gradOutput)
end

function LSTM:accGradParameters(input, gradOutput, scale)
   -- Back-Propagate Through Time (BPTT) happens in updateParameters()
   -- for now we just keep a list of the scales
   self.scales[self.step-1] = scale
end

function LSTM:backwardThroughTime()
   assert(self.step > 1, "expecting at least one updateOutput")
   local rho = math.min(self.rho, self.step-1)
   local stop = self.step - rho
   if self.fastBackward then
      local gradInput
      for step=self.step-1,math.max(stop,1),-1 do
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
            assert(gradInput, "missing gradInput")
            local output_ = recurrentOutputs[i]
            assert(output_, "backwardThroughTime should be preceded by updateOutput")
            modula.output = output_
            modula.gradInput = recursiveCopy(recurrentGradInputs[i], gradInput)
         end
         
         -- backward propagate through this step
         local input = self.inputs[step]
         local output = self.outputs[step-1]
         local gradOutput = self.gradOutputs[step] 
         if gradInput then
            recursiveAdd(gradOutput, gradInput)   
         end
         local scale = self.scales[step]
         
         gradInput = self.recurrentModule:backward({input, output}, gradOutput, scale/rho)[2]
         
         for i,modula in ipairs(modules) do
            recurrentGradInputs[i] = modula.gradInput
         end
      end
   else
      local gradInput = self:updateGradInputThroughTime()
      self:accGradParametersThroughTime()
      return gradInput
   end
end

function LSTM:backwardUpdateThroughTime(learningRate)
   local gradInput = self:updateGradInputThroughTime()
   self:accUpdateGradParametersThroughTime(learningRate)
   return gradInput
end

function LSTM:updateGradInputThroughTime()
   assert(self.step > 1, "expecting at least one updateOutput")
   local gradInput
   local rho = math.min(self.rho, self.step-1)
   local stop = self.step - rho
   for step=self.step-1,math.max(stop,1),-1 do
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
         assert(output_, "updateGradInputThroughTime should be preceded by updateOutput")
         modula.output = output_
         modula.gradInput = self.recursiveCopy(recurrentGradInputs[i], gradInput)
      end
      
      -- backward propagate through this step
      local input = self.inputs[step]
      local output = self.outputs[step-1]
      local gradOutput = self.gradOutputs[step]
      if gradInput then
         gradOutput:add(gradInput)
      end
      gradInput = self.recurrentModule:updateGradInput({input, output}, gradOutput)[2]
      for i,modula in ipairs(modules) do
         recurrentGradInputs[i] = modula.gradInput
      end
   end
   
   return gradInput
end

function LSTM:accGradParametersThroughTime()
   local rho = math.min(self.rho, self.step-1)
   local stop = self.step - rho
   for step=self.step-1,math.max(stop,1),-1 do
      -- set the output/gradOutput states of current Module
      local modules = self.recurrentModule:listModules()
      local recurrentOutputs = self.recurrentOutputs[step]
      local recurrentGradInputs = self.recurrentGradInputs[step]
      
      for i,modula in ipairs(modules) do
         local output, gradInput = modula.output, modula.gradInput
         local output_ = recurrentOutputs[i]
         local gradInput_ = recurrentGradInputs[i]
         assert(output_, "accGradParametersThroughTime should be preceded by updateOutput")
         assert(gradInput_, "accGradParametersThroughTime should be preceded by updateGradInputThroughTime")
         modula.output = output_
         modula.gradInput = gradInput_
      end
      
      -- backward propagate through this step
      local input = self.inputs[step]
      local output = self.outputs[step-1]
      local gradOutput = self.gradOutputs[step]

      local scale = self.scales[step]
      self.recurrentModule:accGradParameters({input, output}, gradOutput, scale/rho)
      
   end
   
   self.gradParametersAccumulated = true
   return gradInput
end

function LSTM:accUpdateGradParametersThroughTime(lr)
   local rho = math.min(self.rho, self.step-1)
   local stop = self.step - rho
   for step=self.step-1,math.max(stop,1),-1 do
      -- set the output/gradOutput states of current Module
      local modules = self.recurrentModule:listModules()
      local recurrentOutputs = self.recurrentOutputs[step]
      local recurrentGradInputs = self.recurrentGradInputs[step]
      
      for i,modula in ipairs(modules) do
         local output, gradInput = modula.output, modula.gradInput
         local output_ = recurrentOutputs[i]
         local gradInput_ = recurrentGradInputs[i]
         assert(output_, "accGradParametersThroughTime should be preceded by updateOutput")
         assert(gradInput_, "accGradParametersThroughTime should be preceded by updateGradInputThroughTime")
         modula.output = output_
         modula.gradInput = gradInput_
      end
      
      -- backward propagate through this step
      local input = self.inputs[step]
      local output = self.outputs[step-1]
      local gradOutput = self.gradOutputs[step]

      local scale = self.scales[step]
      self.recurrentModule:accUpdateGradParameters({input, output}, gradOutput, lr*scale/rho)
   end
   
   return gradInput
end

