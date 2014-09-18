local PushTable, parent = torch.class("nn.PushTable", "nn.Module")

function PushTable:__init(index)
   self._index = index
   self._pulls = {}
   self.output = {}
   self._gradInput = torch.Tensor()
   self.gradInput = {}
   self._nForward = 0
   self._nBackward = 0
end

function PushTable:pull(index)
   local pull = nn.PullTable(self, index)
   table.insert(self._pulls, pull)
   return pull
end

function PushTable:updateOutput(inputTable)
   for i, input in ipairs(inputTable) do
      if i < self._index then
         self.output[i] = input
      elseif i > self._index then
         self.output[i-1] = input
      end
   end
   
   local input = inputTable[self._index]
   for i,pull in ipairs(self._pulls) do
      pull:_updateOutput(input)
   end
   
   self._nBackward = 0
   return self.output
end

function PushTable:_updateGradInput(gradOutput)
   if self._nBackward == 0 then
      self._gradInput:copy(gradOutput)
   else
      self._gradInput:add(gradOutput)
   end
   self._nBackward = self._nBackward + 1
end

function PushTable:updateGradInput(inputTable, gradOutputTable)
   if self._nBackward ~= self._nForward then
      error("n Inputs forwarded (pushed) ~= n gradOutputs backwarded"..
            " (pulled) : "..self._nForward.." ~= "..self._nBackward) 
   end
   self._nForward = 0
   
   for i, gradOutput in ipairs(gradOutputTable) do
      if i < self._index then
         self.gradInput[i] = gradOutput
      elseif i > self._index then
         self.gradInput[i+1] = gradOutput
      end
   end
   self.gradInput[self._index] = self._gradInput
   assert(#inputTable == #self.gradInput, "tables size mismatch")
   return self.gradInput
end



