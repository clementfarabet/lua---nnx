local SoftMaxTree, parent = torch.class('nn.SoftMaxTree', 'nn.Module')
------------------------------------------------------------------------
--[[ SoftMaxTree ]]--
-- Computes the log of a product of softmaxes in a path
-- Returns an output tensor of size 1D
-- Only works with a tree (one parent per child)

-- TODO: 
-- a shareClone method to make clones without wasting memory
-- which may require differing setup after initialization
------------------------------------------------------------------------

function SoftMaxTree:__init(inputSize, hierarchy, rootId, verbose)
   parent.__init(self)
   self.rootId = rootId or 0
   self.inputSize = inputSize
   assert(type(hierarchy) == 'table', "Expecting table at arg 2")
   -- get the total amount of children (non-root nodes)
   local nChildNode = 0
   local nParentNode = 0
   local maxNodeId = -99999999
   local minNodeId = 999999999
   local maxParentId = -99999999
   local maxChildId = -9999999
   local parentIds = {}
   for parentId, children in pairs(hierarchy) do
      assert(children:dim() == 1, "Expecting table of 1D tensors at arg 2")
      nChildNode = nChildNode + children:size(1)
      nParentNode = nParentNode + 1
      maxParentId = math.max(parentId, maxParentId)
      local maxChildrenId = children:max()
      maxChildId = math.max(maxChildrenId, maxChildId)
      maxNodeId = math.max(parentId, maxNodeId, maxChildrenId)
      minNodeId = math.min(parentId, minNodeId, children:min())
      table.insert(parentIds, parentId)
   end
   if minNodeId < 0 then
      error("nodeIds must must be positive: "..minNodeId, 2) 
   end
   if verbose then
      print("Hierachy has :")
      print(nParentNode.." parent nodes")
      print(nChildNode.." child nodes")
      print((nChildNode - nParentNode).." leaf nodes")
      print("node index will contain "..maxNodeId.." slots")
      if maxNodeId ~= (nChildNode + 1) then
         print("Warning: Hierarchy has more nodes than Ids")
         print("Consider making your nodeIds a contiguous sequence ")
         print("in order to waste less memory on indexes.")
      end
   end
   
   self.nChildNode = nChildNode
   self.nParentNode = nParentNode
   self.minNodeId = minNodeId
   self.maxNodeId = maxNodeId
   self.maxParentId = maxParentId
   self.maxChildId = maxChildId
   
   -- initialize weights and biases
   self.weight = torch.Tensor(self.nChildNode, self.inputSize)
   self.bias = torch.Tensor(self.nChildNode)
   self.gradWeight = torch.Tensor(self.nChildNode, self.inputSize)
   self.gradBias = torch.Tensor(self.nChildNode)
   
   -- contains all childIds
   self.childIds = torch.IntTensor(self.nChildNode)
   -- contains all parentIds
   self.parentIds = torch.IntTensor(parentIds)
   
   -- index of children by parentId
   self.parentChildren = torch.IntTensor(self.maxParentId, 2):fill(-1)
   local start = 1
   for parentId, children in pairs(hierarchy) do
      local node = self.parentChildren:select(1, parentId)
      node[1] = start
      local nChildren = children:size(1)
      node[2] = nChildren
      self.childIds:narrow(1, start, nChildren):copy(children)
      start = start + nChildren
   end
   
   -- index of parent by childId
   self.childParent = torch.IntTensor(self.maxChildId, 2):fill(-1)
   for parentIdx=1,self.parentIds:size(1) do
      local parentId = self.parentIds[parentIdx]
      local node = self.parentChildren:select(1, parentId)
      local start = node[1]
      local nChildren = node[2]
      local children = self.childIds:narrow(1, start, nChildren)
      for childIdx=1,children:size(1) do
         local childId = children[childIdx]
         local child = self.childParent:select(1, childId)
         child[1] = parentId
         child[2] = childIdx
      end
   end
   
   -- stores the parentIds of nodes that have been accGradParameters
   self.updates = {}
   
   -- used internally to store intermediate outputs or gradOutputs
   self._linearOutput = torch.Tensor()
   self._linearGradOutput = torch.Tensor()
   self._logSoftMaxOutput = torch.Tensor()
   
   self:reset()
end

function SoftMaxTree:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.nChildNode*self.inputSize)
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
end

function SoftMaxTree:updateOutput(inputTable)
   local input, target = unpack(inputTable)
   return input.nn.SoftMaxTree_updateOutput(self, input, target)
end

function SoftMaxTree:updateGradInput(inputTable, gradOutput)
   local input, target = unpack(inputTable)
   if self.gradInput then
      return input.nn.SoftMaxTree_updateGradInput(self, input, gradOutput, target)
   end
end

function SoftMaxTree:accGradParameters(inputTable, gradOutput, scale)
   local input, target = unpack(inputTable)
   scale = scale or 1
   input.nn.SoftMaxTree_accGradParameters(self, input, gradOutput, target, scale)
end

-- when static is true, return parameters with static keys
-- i.e. keys that don't change from batch to batch
function SoftMaxTree:parameters(static)
   local params, grads = {}, {}
   local updated = false
   for parentId, scale in pairs(self.updates) do
      local node = self.parentChildren:select(1, parentId)
      local parentIdx = node[1]
      local nChildren = node[2]
      if static then
         params[parentId] = self.weight:narrow(1, parentIdx, nChildren)
         grads[parentId] = self.gradWeight:narrow(1, parentIdx, nChildren)
         local biasId = parentId+self.maxParentId
         params[biasId] = self.bias:narrow(1, parentIdx, nChildren)
         grads[biasId] = self.gradBias:narrow(1, parentIdx, nChildren)
      else
         table.insert(params, self.weight:narrow(1, parentIdx, nChildren))
         table.insert(params, self.bias:narrow(1, parentIdx, nChildren))
         table.insert(grads, self.gradWeight:narrow(1, parentIdx, nChildren))
         table.insert(grads, self.gradBias:narrow(1, parentIdx, nChildren))
      end
      updated = true
   end
   if not updated then
      return {self.weight, self.bias}, {self.gradWeight, self.gradBias}
   end
   return params, grads
end

function SoftMaxTree:getNodeParameters(parentId)
   local node = self.parentChildren:select(1,parentId)
   local start = node[1]
   local nChildren = node[2]
   local weight = self.weight:narrow(1, start, nChildren)
   local bias = self.bias:narrow(1, start, nChildren)
   local gradWeight = self.gradWeight:narrow(1, start, nChildren)
   local gradBias = self.gradBias:narrow(1, start, nChildren)
   return {weight, bias}, {gradWeight, gradBias}
end

function SoftMaxTree:zeroGradParameters(partial)
   local _,gradParams = self:parameters(partial)
   for k,gradParam in pairs(gradParams) do
      gradParam:zero()
   end
   self.updates = {}
end

function SoftMaxTree:type(type)
   if type and (type == 'torch.FloatTensor' or type == 'torch.DoubleTensor') then
      self.weight = self.weight:type(type)
      self.bias = self.bias:type(type)
      self.gradWeight = self.gradWeight:type(type)
      self.gradBias = self.gradBias:type(type)
      self._linearOutput = self._linearOutput:type(type)
      self._linearGradOutput = self._linearGradOutput:type(type)
      self._logSoftMaxOutput = self._logSoftMaxOutput:type(type)
      self.output = self.output:type(type)
      self.gradInput = self.gradInput:type(type)
   end
   return self
end

-- we do not need to accumulate parameters when sharing
SoftMaxTree.sharedAccUpdateGradParameters = SoftMaxTree.accUpdateGradParameters
