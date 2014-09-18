local PushTable, parent = torch.class("nn.PushTable", "nn.Module")

function PushTable:__init(index)
   self._index = index
   self._pulls = {}
   self.output = {}
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
      pull:push(input)
   end
   return self.output
end

function PushTable:updateGradInput(inputTable, gradOutputTable)
   
   
end



