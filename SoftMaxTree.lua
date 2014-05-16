local SoftMaxTree, parent = torch.class('nn.SoftMaxTree', 'nn.Module')
------------------------------------------------------------------------
--[[ SoftMaxTree ]]--
-- Generates an output tensor of size 1D
-- Computes the log of a product of softmaxes in a path
-- One parent per child

-- TODO: 
-- a shareClone method to make speedier clones
-- differ setup after init
-- verify that each parent has a parent (except root)
-- nodeIds - 1?
-- types
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
      minNodeId = math.max(parentId, minNodeId, children:min())
      table.insert(parentIds, parentId)
   end
   assert(minNodeId >= 0, "nodeIds must must be non-negative")
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
   
   -- used internally to store intermediate outputs
   self._linearOutput = torch.Tensor()
   self._logSoftMaxOutput = torch.Tensor()
   self._narrowOutput = torch.Tensor()
   
   print("parentIds", self.parentIds)
   print("parentChildren", self.parentChildren)
   print("children", self.childIds)
   print("childParent", self.childParent)
   self:reset()
end

function SoftMaxTree:getNodeParameters(parentId)
   local node = self.parentChildren:select(1,parentId)
   local start = node[1]
   local nChildren = node[2]
   local weight = self.weight:narrow(1, start, nChildren)
   local bias = self.bias:narrow(1, start, nChildren)
   return weight, bias
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

function SoftMaxTree:updateGradInput(inputTable, gradOutputTable)
   local input, target = unpack(inputTable)
   local gradOutput, target = unpack(gradOutputTable)
   if self.gradInput then
      return input.nn.SoftMaxTree_updateGradInput(self, input, gradOutput, target)
   end
end

function SoftMaxTree:accGradParameters(inputTable, gradOutputTable, scale)
   local input, target = unpack(inputTable)
   local gradOutput, target = unpack(gradOutputTable)
   scale = scale or 1
   input.nn.SoftMaxTree_accGradParameters(self, input, gradOutput, scale)
end

-- we do not need to accumulate parameters when sharing
SoftMaxTree.sharedAccUpdateGradParameters = SoftMaxTree.accUpdateGradParameters
