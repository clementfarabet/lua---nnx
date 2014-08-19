------------------------------------------------------------------------
--[[ NarrowLookupTable ]]--
-- Concatenates embeddings with descending narrowed sizes 
-- (ascDelta = true).
-- Useful for language models, where most recent words in context 
-- are more useful in predicting next word than older ones.
-- If input is ordered furthest to nearest word, use ascDelta = false.
------------------------------------------------------------------------
local NarrowLookupTable, parent = torch.class('nn.NarrowLookupTable', 'nn.LookupTable')

function NarrowLookupTable:__init(deltaSize, nIndex, embedSize, ascDelta)
   nn.Module.__init(self)
   self.deltaSize = deltaSize
   self.deltaSizes = torch.LongTensor()
   self.embedSize = embedSize
   self.ascDelta = (ascDelta == nil) and true or ascDelta
   
   self.weight = torch.Tensor(nIndex, embedSize)
   self.gradWeight = torch.Tensor(nIndex, embedSize):zero()
   self.inputs = {}
   
   self.accUpdate = false
   self.nIndex = 0

   self.nBackward = 0
   self:reset()
end

-- this could be overrided in a subclass :
function NarrowLookupTable:buildSizes(nIndex)
   if self.nIndex == nIndex then
      return
   end
   self.deltaSizes:resize(nIndex)
   local deltaSize = 0
   if self.ascDelta then
      for i=1,self.deltaSizes:size(1),1 do
         self.deltaSizes[i] = deltaSize
         deltaSize = deltaSize + self.deltaSize
      end
   else
      for i=self.deltaSizes:size(1),1,-1 do
         self.deltaSizes[i] = deltaSize
         deltaSize = deltaSize + self.deltaSize
      end
   end
   self.outputSize = nIndex*self.embedSize - self.deltaSizes:sum()
   self.nIndex = nIndex
end

function NarrowLookupTable:updateOutput(input)
   if input:dim() == 1 then
      local nIndex = input:size(1)
      self:buildSizes(nIndex)
      self.output:resize(self.outputSize)
      local embedIdx = 1
      for i=1,nIndex do
         local embedSize = self.embedSize - self.deltaSizes[i]
         local embed = self.weight[input[i]]:narrow(1, 1, embedSize)
         self.output:narrow(1, embedIdx, embedSize):copy(embed)
         embedIdx = embedIdx + embedSize
      end
   elseif input:dim() == 2 then
      local nExample = input:size(1)
      local nIndex = input:size(2)
      self:buildSizes(nIndex)
      self.output:resize(nExample, self.outputSize)
      for i=1,nExample do
         local output = self.output:select(1, i)
         local input = input:select(1, i)
         local embedIdx = 1
         for j=1,nIndex do
            local embedSize = self.embedSize - self.deltaSizes[j]
            local embed = self.weight[input[j]]:narrow(1, 1, embedSize)
            output:narrow(1, embedIdx, embedSize):copy(embed)
            embedIdx = embedIdx + embedSize
         end
      end
   end

   return self.output
end

function NarrowLookupTable:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.nBackward = self.nBackward + 1
      local embedIdx = 1
      for i=1,input:size(1) do
         local k = input[i]
         self.inputs[k] = (self.inputs[k] or 0) + 1
         local embedSize = self.embedSize - self.deltaSizes[i]
         local gradEmbed = gradOutput:narrow(1, embedIdx, embedSize)
         self.gradWeight[input[i]]:narrow(1, 1, embedSize):add(gradEmbed)
         embedIdx = embedIdx + embedSize
      end
   elseif input:dim() == 2 then
      self.nBackward = self.nBackward + input:size(1)
      for i=1,input:size(1) do
         local input = input:select(1, i)
         local gradOutput = gradOutput:select(1, i)
         local embedIdx = 1
         for j=1,input:size(1) do
            local k = input[j]
            self.inputs[k] = (self.inputs[k] or 0) + 1
            local embedSize = self.embedSize - self.deltaSizes[j]
            local gradEmbed = gradOutput:narrow(1, embedIdx, embedSize)
            self.gradWeight[input[j]]:narrow(1, 1, embedSize):add(gradEmbed)
            embedIdx = embedIdx + embedSize
         end
      end
   end
end

function NarrowLookupTable:accUpdateGradParameters(input, gradOutput, lr)
   if input:dim() == 1 then
      local embedIdx = 1
      for i=1,input:size(1) do
         local k = input[j]
         local kscale = self:scaleUpdateByKey(k)
         local embedSize = self.embedSize - self.deltaSizes[i]
         local gradEmbed = gradOutput:narrow(1, embedIdx, embedSize)
         self.weight[input[i]]:narrow(1, 1, embedSize):add(-lr*kscale, gradEmbed)
         embedIdx = embedIdx + embedSize
      end
   elseif input:dim() == 2 then 
      for i=1,input:size(1) do
         local input = input:select(1, i)
         local gradOutput = gradOutput:select(1, i)
         local embedIdx = 1
         for j=1,input:size(1) do
            local k = input[j]
            local kscale = self:scaleUpdateByKey(k)
            local embedSize = self.embedSize - self.deltaSizes[j]
            local gradEmbed = gradOutput:narrow(1, embedIdx, embedSize)
            self.weight[input[j]]:narrow(1, 1, embedSize):add(-lr*kscale, gradEmbed)
            embedIdx = embedIdx + embedSize
         end
      end
   end
end

function NarrowLookupTable:type(type)
   self.gradInput = self.gradInput:type(type)
   self.output = self.output:type(type)
   self.weight = self.weight:type(type)
   self.gradWeight = self.gradWeight:type(type)
end
