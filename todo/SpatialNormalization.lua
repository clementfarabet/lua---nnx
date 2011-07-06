local SpatialNormalization, parent = torch.class('nn.SpatialNormalization','nn.Module')

local help_desc = 
[[a spatial (2D) contrast normalizer
? computes the local mean and local std deviation
  across all input features, using the given 2D kernel
? the local mean is then removed from all maps, and the std dev
  used to divide the inputs, with a threshold
? if no threshold is given, the global std dev is used
? weight replication is used to preserve sizes (this is
  better than zero-padding, but more costly to compute, use
  nn.ContrastNormalization to use zero-padding)
? two 1D kernels can be used instead of a single 2D kernel. This
  is beneficial to integrate information over large neiborhoods.
]]

local help_example = 
[[EX:
-- create a contrast normalizer, with a 9x9 gaussian kernel
-- works on 8 input feature maps, therefore the mean+dev will
-- be estimated on 9x9x8 cubes
stimulus = lab.randn(500,500,8)
gaussian = image.gaussian{width=9}
mod = nn.SpatialNormalization(gaussian, 8)
result = mod:forward(stimulus)]]

function SpatialNormalization:__init(...) -- kernel for weighted mean | nb of features
   parent.__init(self)

   -- get args
   local args, ker, nf, thres, kers1D
      = xlua.unpack(
      {...},
      'nn.SpatialNormalization',
      help_desc .. '\n' .. help_example,
      {arg='kernel', type='torch.Tensor', help='a KxK filtering kernel'},
      {arg='nInputPlane', type='number', help='number of input maps', req=true},
      {arg='threshold', type='number', help='threshold, for division [default = adaptive]'},
      {arg='kernels', type='table', help='two 1D filtering kernels (1xK and Kx1)'}
   )

   -- check args
   if not ker and not kers1D then
      xerror('please provide kernel(s)', 'nn.SpatialNormalization', args.usage)
   end
   self.kernel = ker or kers1D
   local ker2
   if kers1D then
      ker = kers1D[1]
      ker2 = kers1D[2]
   end
   self.nfeatures = nf
   self.fixedThres = thres -- optional, if not provided, the global std is used

   -- padding values
   self.padW = math.floor(ker:size(1)/2)
   self.padH = math.floor(ker:size(2)/2)
   self.kerWisPair = 0
   self.kerHisPair = 0

   -- padding values for 2nd kernel
   if ker2 then
      self.pad2W = math.floor(ker2:size(1)/2)
      self.pad2H = math.floor(ker2:size(2)/2)
   else
      self.pad2W = 0
      self.pad2H = 0
   end
   self.ker2WisPair = 0
   self.ker2HisPair = 0

   -- normalize kernel
   ker:div(ker:sum())
   if ker2 then ker2:div(ker2:sum()) end

   -- manage the case where ker is even size (for padding issue)
   if (ker:size(1)/2 == math.floor(ker:size(1)/2)) then
      print ('Warning, kernel width is even -> not symetric padding')
      self.kerWisPair = 1
   end
   if (ker:size(2)/2 == math.floor(ker:size(2)/2)) then
      print ('Warning, kernel height is even -> not symetric padding')
      self.kerHisPair = 1
   end
   if (ker2 and ker2:size(1)/2 == math.floor(ker2:size(1)/2)) then
      print ('Warning, kernel width is even -> not symetric padding')
      self.ker2WisPair = 1
   end
   if (ker2 and ker2:size(2)/2 == math.floor(ker2:size(2)/2)) then
      print ('Warning, kernel height is even -> not symetric padding')
      self.ker2HisPair = 1
   end
   
   -- create convolution for computing the mean
   convo1 = nn.Sequential()
   convo1:add(nn.SpatialPadding(self.padW,self.padW-self.kerWisPair,
                                    self.padH,self.padH-self.kerHisPair))
   local ctable = nn.SpatialConvolutionTable:OneToOneTable(nf)
   convo1:add(nn.SpatialConvolutionTable(ctable,ker:size(1),ker:size(2)))
   convo1:add(nn.Sum(3))
   convo1:add(nn.Replicate(nf))
   -- set kernel
   local fb = convo1.modules[2].weight
   for i=1,fb:size(3) do fb:select(3,i):copy(ker) end
   -- set bias to 0
   convo1.modules[2].bias:zero()

   -- 2nd ker ?
   if ker2 then
      local convo2 = nn.Sequential()
      convo2:add(nn.SpatialPadding(self.pad2W,self.pad2W-self.ker2WisPair,
                                   self.pad2H,self.pad2H-self.ker2HisPair))
      local ctable = nn.SpatialConvolutionTable:OneToOneTable(nf)
      convo2:add(nn.SpatialConvolutionTable(ctable,ker2:size(1),ker2:size(2)))
      convo2:add(nn.Sum(3))
      convo2:add(nn.Replicate(nf))
      -- set kernel
      local fb = convo2.modules[2].weight
      for i=1,fb:size(3) do fb:select(3,i):copy(ker2) end
      -- set bias to 0
      convo2.modules[2].bias:zero()
      -- convo is a double convo now:
      local convopack = nn.Sequential()
      convopack:add(convo1)
      convopack:add(convo2)
      self.convo = convopack
   else
      self.convo = convo1
   end

   -- create convolution for computing the meanstd
   convostd1 = nn.Sequential()
   convostd1:add(nn.SpatialPadding(self.padW,self.padW-self.kerWisPair,
                                      self.padH,self.padH-self.kerHisPair))
   convostd1:add(nn.SpatialConvolutionTable(ctable,ker:size(1),ker:size(2)))
   convostd1:add(nn.Sum(3))
   convostd1:add(nn.Replicate(nf))
   -- set kernel
   local fb = convostd1.modules[2].weight
   for i=1,fb:size(3) do fb:select(3,i):copy(ker) end
   -- set bias to 0
   convostd1.modules[2].bias:zero()

   -- 2nd ker ?
   if ker2 then
      local convostd2 = nn.Sequential()
      convostd2:add(nn.SpatialPadding(self.pad2W,self.pad2W-self.ker2WisPair,
                                   self.pad2H,self.pad2H-self.ker2HisPair))
      convostd2:add(nn.SpatialConvolutionTable(ctable,ker2:size(1),ker2:size(2)))
      convostd2:add(nn.Sum(3))
      convostd2:add(nn.Replicate(nf))
      -- set kernel
      local fb = convostd2.modules[2].weight
      for i=1,fb:size(3) do fb:select(3,i):copy(ker2) end
      -- set bias to 0
      convostd2.modules[2].bias:zero()
      -- convo is a double convo now:
      local convopack = nn.Sequential()
      convopack:add(convostd1)
      convopack:add(convostd2)
      self.convostd = convopack
   else
      self.convostd = convostd1
   end

   -- other operation
   self.squareMod = nn.Square()
   self.sqrtMod = nn.Sqrt()
   self.substractMod = nn.CSubTable()
   self.meanDiviseMod = nn.CDivTable()
   self.stdDiviseMod = nn.CDivTable()
   self.diviseMod = nn.CDivTable()
   self.thresMod = nn.Threshold()
   -- some tempo states
   self.coef = torch.Tensor(1,1)
   self.inConvo = torch.Tensor()
   self.inMean = torch.Tensor()
   self.inputZeroMean = torch.Tensor()
   self.inputZeroMeanSq = torch.Tensor()
   self.inConvoVar = torch.Tensor()
   self.inVar = torch.Tensor()
   self.inStdDev = torch.Tensor()
   self.thstd = torch.Tensor()
end

--------------------------------------------------------------------------------
-- pad with 0s the 2 first dims of the input
-- first dim padded with padW on both side
-- second dim padded with padH on both side
-- the function is deprecated
--------------------------------------------------------------------------------
function SpatialNormalization:pad(input, padH, padW)
   if (input:nDimension()<2) then
      error('input has to have at least 2 Dimensions')
   end
   local outSize = input:size()
   -- handle kernel even size
   outSize[1] = outSize[1] + 2*padH - self.kerHisPair
   outSize[2] = outSize[2] + 2*padW - self.kerWisPair
   local out = torch.Tensor(outSize):zero()
   out:sub(padH+1,padH+input:size(1),padW+1,padW+input:size(2)):copy(input)
   return out
end

function SpatialNormalization:forward(input)
   -- init coef if necessary
   if (input:size(1) ~= self.coef:size(1)) 
   or (input:size(2) ~= self.coef:size(2)) then
      -- recompute coef only if necessary
      local intVals = torch.Tensor(input:size(1),input:size(2),self.nfeatures):fill(1)
      self.coef = self.convo:forward(intVals)
      self.coef = torch.Tensor():resizeAs(self.coef):copy(self.coef)
   end
   self.input = input
   if (input:nDimension() == 2) then
      self.input = torch.Tensor():resize(input:size(1),input:size(2),1):copy(input)
   end
   -- compute mean
   self.inConvo = self.convo:forward(self.input)
   self.inMean = self.meanDiviseMod:forward({self.inConvo,self.coef})
   self.inputZeroMean = self.substractMod:forward({self.input,self.inMean})
   -- compute std dev
   self.inputZeroMeanSq = self.squareMod:forward(self.inputZeroMean)
   self.inConvoVar = self.convostd:forward(self.inputZeroMeanSq)
   -- mine it's not workinf with jaco test
   self.inStdDevNotUnit = self.sqrtMod:forward(self.inConvoVar)
   self.inStdDev = self.stdDiviseMod:forward({self.inStdDevNotUnit,self.coef})
   -- koray working way
   -- self.inStdDev = self.sqrtMod:forward(self.inConvoVar)
   -- threshold to avoid zero division
   -- koray's way
   -- btw need to set this to a constant in order to pass jacobian test
   local meanstd = self.inStdDev:mean()
   self.thresMod.threshold = self.fixedThres or math.max(meanstd,1e-3)
   self.thresMod.val = self.fixedThres or math.max(meanstd,1e-3)
   self.stdDev = self.thresMod:forward(self.inStdDev)
   -- my way
   -- local epsilon = math.max(1e-8,self.inStdDev:mean()/8 - 1e-6)
   -- self.thresMod.threshold = epsilon
   -- self.thresMod.val = 10e8
   -- self.stdDev = self.thresMod:forward(self.inStdDev)
   
   --remove std dev
   self.diviseMod:forward({self.inputZeroMean,self.stdDev})
   self.output = self.diviseMod.output
   return self.output
end

function SpatialNormalization:backward(input, gradOutput)
   self.input = input
   if (input:nDimension() == 2) then
      self.input = torch.Tensor():resize(input:size(1),input:size(2),1):copy(input)
   end
   self.gradInput:resizeAs(self.input):zero()
   local gradDiv = self.diviseMod:backward({self.inputZeroMean,self.stdDev},gradOutput)
   local gradThres = gradDiv[2]
   local gradZeroMean = gradDiv[1]
   local gradinStdDev = self.thresMod:backward(self.inStdDev,gradThres)

   local gradstdDiv = self.stdDiviseMod:backward({self.inStdDevNotUnit,self.coef},gradinStdDev)
   local gradinStdDevNotUnit = gradstdDiv[1]

   local gradinConvoVar  = self.sqrtMod:backward(self.inConvoVar,gradinStdDevNotUnit)
   local gradinputZeroMeanSq = self.convostd:backward(self.inputZeroMeanSq,gradinConvoVar)

   gradZeroMean:add(self.squareMod:backward(self.inputZeroMean,gradinputZeroMeanSq))
   local gradDiff = self.substractMod:backward({self.input,self.inMean},gradZeroMean)
   local gradinMean = gradDiff[2]
   local gradinConvoNotUnit = self.meanDiviseMod:backward({self.inConvo,self.coef},gradinMean)
   local gradinConvo = gradinConvoNotUnit[1]
   -- first part of the gradInput
   self.gradInput:add(gradDiff[1])
   -- second part of the gradInput
   self.gradInput:add(self.convo:backward(self.input,gradinConvo))
   return self.gradInput
end

function SpatialNormalization:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.convo:empty()
   self.convostd:empty()
   self.squareMod:empty()
   self.sqrtMod:empty()
   self.squareMod:empty()
   self.sqrtMod:empty()
   self.substractMod:empty()
   self.meanDiviseMod:empty()
   self.stdDiviseMod:empty()
   self.thresMod:empty()
   self.diviseMod:empty()
   self.coef:resize(1,1)
   self.coef:storage():resize(1)
end

function SpatialNormalization:write(file)
   parent.write(self,file)
   file:writeObject(self.kernel)
   file:writeInt(self.nfeatures)
   file:writeInt(self.padW)
   file:writeInt(self.padH)
   file:writeInt(self.kerWisPair)
   file:writeInt(self.kerHisPair)
   file:writeObject(self.convo)
   file:writeObject(self.convostd)
   file:writeObject(self.squareMod)
   file:writeObject(self.sqrtMod)
   file:writeObject(self.substractMod)
   file:writeObject(self.meanDiviseMod)
   file:writeObject(self.stdDiviseMod)
   file:writeObject(self.thresMod)
   file:writeObject(self.diviseMod)
   file:writeObject(self.coef)
   if type(self.kernel) == 'table' then
      file:writeInt(self.pad2W)
      file:writeInt(self.pad2H)
      file:writeInt(self.ker2WisPair)
      file:writeInt(self.ker2HisPair)
   end
end

function SpatialNormalization:read(file)
   parent.read(self,file)
   self.kernel = file:readObject()
   self.nfeatures = file:readInt()
   self.padW = file:readInt()
   self.padH = file:readInt()
   self.kerWisPair = file:readInt()
   self.kerHisPair = file:readInt()
   self.convo = file:readObject()
   self.convostd = file:readObject()
   self.squareMod = file:readObject()
   self.sqrtMod = file:readObject()
   self.substractMod = file:readObject()
   self.meanDiviseMod = file:readObject()
   self.stdDiviseMod = file:readObject()
   self.thresMod = file:readObject()
   self.diviseMod = file:readObject()
   self.coef = file:readObject()
   if type(self.kernel) == 'table' then
      self.pad2W = file:readInt()
      self.pad2H = file:readInt()
      self.ker2WisPair = file:readInt()
      self.ker2HisPair = file:readInt()
   end
end
