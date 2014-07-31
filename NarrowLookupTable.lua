local NarrowLookupTable, parent = torch.class('nn.NarrowLookupTable', 'nn.LookupTable')

function NarrowLookupTable:__init(deltaSize, nIndex, embedSize)
   nn.Module.__init(self)
   self.deltaSize = deltaSize
   self.deltaSizes = torch.LongTensor()
   self.embedSize = embedSize
   
   self.weight = torch.Tensor(nIndex, embedSize)
   self.gradWeight = torch.Tensor(nIndex, embedSize):zero()
   self.inputs = {}
   
   self.accUpdate = false
   self.nIndex = 0

   self.nBackward = 0
   self:reset()
end

function NarrowLookupTable:buildSizes(nIndex)
   if self.nIndex == nIndex then
      return
   end
   self.deltaSizes:resize(nIndex)
   local deltaSize = 0
   for i=1,self.deltaSizes:size(1) do
      self.deltaSizes[i] = deltaSize
      deltaSize = deltaSize + self.deltaSize
   end
   self.outputSize = self.deltaSizes:sum()
   assert(nIndex*self.embedSize + self.deltaSize - (nIndex*self.deltaSize*(nIndex+1)/2) == self.outputSize)
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
      for i=1,input:size(1) do
         local k = input[i]
         self.inputs[k] = (self.inputs[k] or 0) + 1
         self.gradWeight:select(1, k):add(scale, gradOutput:select(1, i))
      end
   elseif input:dim() == 2 then
      self.nBackward = self.nBackward + input:size(1)
      for i=1,input:size(1) do
         local input = input:select(1, i)
         local gradOutput = gradOutput:select(1, i)
         for j=1,input:size(1) do
            local k = input[j]
            self.inputs[k] = (self.inputs[k] or 0) + 1
            self.gradWeight:select(1, k):add(scale, gradOutput:select(1, j))
         end
      end
   end
end

function NarrowLookupTable:accUpdateGradParameters(input, gradOutput, lr)
   if input:dim() == 1 then
      for i=1,input:size(1) do
         local k = input[j]
         local kscale = self:scaleUpdateByKey(k)
         self.weight:select(1, input[i]):add(-lr*kscale, gradOutput:select(1, i))
      end
   elseif input:dim() == 2 then 
      for i=1,input:size(1) do
         local input = input:select(1, i)
         local gradOutput = gradOutput:select(1, i)
         for j=1,input:size(1) do
            local k = input[j]
            local kscale = self:scaleUpdateByKey(k)
            self.weight:select(1, k):add(-lr*kscale, gradOutput:select(1, j))
         end
      end
   end
end

function NarrowLookupTable:updateParameters(learningRate)
   assert(not self.accUpdate, "use accUpdateGradParameters instead")
   for k,nBackward in pairs(self.inputs) do
      local kscale = self:scaleUpdateByKey(k)
      self.weight:select(1, k):add(-learningRate*kscale, self.gradWeight:select(1, k))
   end
end

-- scale the update for each key
function NarrowLookupTable:scaleUpdateByKey(inputKey)
   -- default is to perform no key-based scalling
   return 1
end

-- we do not need to accumulate parameters when sharing
NarrowLookupTable.sharedAccUpdateGradParameters = NarrowLookupTable.accUpdateGradParameters
