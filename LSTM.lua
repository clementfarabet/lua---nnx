------------------------------------------------------------------------
--[[ LSTM ]]--
-- Long Short Term Memory architecture.
-- Ref. A.: http://arxiv.org/pdf/1303.5778v1 (blueprint for this module)
-- B. http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf
-- C. https://github.com/wojzaremba/lstm
-- Expects 1D or 2D input.
-- The first input in sequence uses zero value for cell and hidden state
------------------------------------------------------------------------
local LSTM, parent = torch.class('nn.LSTM', 'nn.AbstractRecurrent')

function LSTM:__init(inputSize, outputSize)
   parent.__init(self)
   self.inputSize = inputSize
   self.outputSize = outputSize   
   -- build the model
   self.recurrentModule = self:buildModel()
   -- make it work with nn.Container
   self.modules[1] = self.model
   
   self.startOutput = torch.Tensor()
   self.startCell = torch.Tensor()
   self.cells = {}
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
   -- forget = forgetGate{input, output(t-1), cell(t-1)} * cell(t-1)
   local forget = nn.Sequential()
   local concat = nn.ConcatTable()
   concat:add(self.forgetGate):add(self.SelectTable(3))
   forget:add(concat)
   forget:add(nn.CMulTable())
   -- input = inputGate{input, output(t-1), cell(t-1)} * hiddenLayer{input, output(t-1), cell(t-1)}
   local input = nn.Sequential()
   local concat2 = nn.ConcatTable()
   concat2:add(self.inputGate):add(self.hiddenLayer)
   input:add(concat2)
   input:add(nn.CMulTable())
   -- cell(t) = forget + input
   local cell = nn.Sequential()
   local concat3 = nn.ConcatTable()
   concat3:add(forget):add(input)
   cell:add(concat3)
   cell:add(nn.CAddTable())
end   
   
function LSTM:buildOutputGate()
   return self:buildGate()
end

-- cell(t) = cellLayer{input, output(t-1), cell(t-1)}
-- output = outputGate{input, output(t-1), cell(t)}*tanh(cell(t))
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
   -- output of concat2 is {{input, output}, cell(t)}, 
   -- so flatten to {input, output, cell(t)}
   model:add(nn.FlattenTable())
   local cellAct = nn.Sequential()
   cellAct:add(nn.Select(3))
   cellAct:add(nn.Tanh())
   local concat3 = nn.ConcatTable()
   concat3:add(self.outputGate):add(cellAct)
   -- we want the model to output : {output(t), cell(t)}
   local concat4 = nn.ConcatTable()
   local output = nn.Sequential()
   output:add(concat3)
   output:add(nn.CMulTable())
   concat4:add(output):add(nn.Identity())
   model:add(concat4)
   return model
end

------------------------- forward backward -----------------------------
function LSTM:updateOutput(input)
   local prevOutput, prevCell
   if self.step == 1 then
      prevOutput = self.startOutput
      prevCell = self.startCell
      if input:dim() == 2 then
         self.startOutput:resize(input:size(1), self.outputSize)
      else
         self.startOutput:resize(self.outputSize)
      end
      self.startCell:set(self.startOutput)
   else
      -- previous output and cell of this module
      prevOutput = self.output
      prevCell = self.cell
   end
      
   -- output(t), cell(t) = lstm{input(t), output(t-1), cell(t-1)}
   local output, cell
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
      -- the actual forward propagation
      output, cell = self.recurrentModule:updateOutput{input, prevOutput, prevCell}
      
      for i,modula in ipairs(modules) do
         recurrentOutputs[i]  = modula.output
      end
   else
      output, cell = self.recurrentModule:updateOutput{input, prevOutput, prevCell}
   end
   
   if self.train ~= false then
      local input_ = self.inputs[self.step]
      self.inputs[self.step] = self.copyInputs 
         and self.recursiveCopy(input_, input) 
         or self.recursiveSet(input_, input)     
   end
   
   self.outputs[self.step] = output
   self.cells[self.step] = cell
   
   self.output = output
   self.cell = cell
   
   self.step = self.step + 1
   self.gradParametersAccumulated = false
   -- note that we don't return the cell, just the output
   return self.output
end

function LSTM:backwardThroughTime()
   assert(self.step > 1, "expecting at least one updateOutput")
   local rho = math.min(self.rho, self.step-1)
   local stop = self.step - rho
   if self.fastBackward then
      local gradInput, gradCell
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
         local cell = self.cells[step-1]
         local gradOutput = self.gradOutputs[step] 
         if gradInput then
            recursiveAdd(gradOutput, gradInput)   
         end
         
         local scale = self.scales[step]
         local inputTable = {input, cell, output}
         local gradOutputTable = {gradOutput, gradCell}
         local gradInputTable = self.recurrentModule:backward(inputTable, gradOutputTable, scale/rho)
         gradInput, gradCell = unpack(gradInputTable)
         
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
         self.recursiveAdd(gradOutput, gradInput) 
      end
      
      local scale = self.scales[step]
      local inputTable = {input, cell, output}
      local gradOutputTable = {gradOutput, gradCell}
      local gradInputTable = self.recurrentModule:backward(inputTable, gradOutputTable, scale/rho)
      gradInput, gradCell = unpack(gradInputTable)
      
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
      -- TODO HERE
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

