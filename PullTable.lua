local PullTable, parent = torch.class("nn.PullTable", "nn.Module")

function PullTable:__init(push, index)
   self._push = push
   self._index = index
   self.output = {}
end

function PullTable:push(output)
   self._output = output
end

function PullTable:updateOutput(inputTable)
   if torch.type(inputTable) == 'table' then
      for i, input in ipairs(inputTable) do
         if i < self._index then
            self.output[i] = input
         else
            self.output[i+1] = input
         end
      end
      self.output[self._index] = self._output
   else
      if self._index == 1 then
         self.output[2] = inputTable
         self.output[1] = self._output
      else
         self.output[1] = inputTable
         self.output[2] = self._output
      end
   end
   return self.output
end

function PullTable:updateGradInput(input, gradOutput)
   self._push:addGradOutput(input, gradOutput)
   return self.gradInput
end
