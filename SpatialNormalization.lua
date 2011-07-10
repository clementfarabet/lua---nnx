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
-- create a spatial normalizer, with a 9x9 gaussian kernel
-- works on 8 input feature maps, therefore the mean+dev will
-- be estimated on 8x9x9 cubes
stimulus = lab.randn(8,500,500)
gaussian = image.gaussian(9)
mod = nn.SpatialNormalization(gaussian, 8)
result = mod:forward(stimulus)]]

function SpatialNormalization:__init(...) -- kernel for weighted mean | nb of features
   parent.__init(self)

   -- get args
   local args, nf, ker, thres
      = xlua.unpack(
      {...},
      'nn.SpatialNormalization',
      help_desc .. '\n' .. help_example,
      {arg='nInputPlane', type='number', help='number of input maps', req=true},
      {arg='kernel', type='torch.Tensor | table', help='a KxK filtering kernel or two {1xK, Kx1} 1D kernels'},
      {arg='threshold', type='number', help='threshold, for division [default = adaptive]'}
   )

   -- check args
   if not ker then
      xerror('please provide kernel(s)', 'nn.SpatialNormalization', args.usage)
   end
   self.kernel = ker
   local ker2
   if type(ker) == 'table' then
      ker2 = ker[2]
      ker = ker[1]
   end
   self.nfeatures = nf
   self.fixedThres = thres

   -- padding values
   self.padW = math.floor(ker:size(2)/2)
   self.padH = math.floor(ker:size(1)/2)
   self.kerWisPair = 0
   self.kerHisPair = 0

   -- padding values for 2nd kernel
   if ker2 then
      self.pad2W = math.floor(ker2:size(2)/2)
      self.pad2H = math.floor(ker2:size(1)/2)
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
   if (ker:size(2)/2 == math.floor(ker:size(2)/2)) then
      print ('Warning, kernel width is even -> not symetric padding')
      self.kerWisPair = 1
   end
   if (ker:size(1)/2 == math.floor(ker:size(1)/2)) then
      print ('Warning, kernel height is even -> not symetric padding')
      self.kerHisPair = 1
   end
   if (ker2 and ker2:size(2)/2 == math.floor(ker2:size(2)/2)) then
      print ('Warning, kernel width is even -> not symetric padding')
      self.ker2WisPair = 1
   end
   if (ker2 and ker2:size(1)/2 == math.floor(ker2:size(1)/2)) then
      print ('Warning, kernel height is even -> not symetric padding')
      self.ker2HisPair = 1
   end
   
   -- create convolution for computing the mean
   local convo1 = nn.Sequential()
   convo1:add(nn.SpatialPadding(self.padW,self.padW-self.kerWisPair,
                                self.padH,self.padH-self.kerHisPair))
   local ctable = nn.tables.oneToOne(nf)
   convo1:add(nn.SpatialConvolutionSparse(ctable,ker:size(2),ker:size(1)))
   convo1:add(nn.Sum(1))
   convo1:add(nn.Replicate(nf))
   -- set kernel
   local fb = convo1.modules[2].weight
   for i=1,fb:size(1) do fb[i]:copy(ker) end
   -- set bias to 0
   convo1.modules[2].bias:zero()

   -- 2nd ker ?
   if ker2 then
      local convo2 = nn.Sequential()
      convo2:add(nn.SpatialPadding(self.pad2W,self.pad2W-self.ker2WisPair,
                                   self.pad2H,self.pad2H-self.ker2HisPair))
      local ctable = nn.tables.oneToOne(nf)
      convo2:add(nn.SpatialConvolutionSparse(ctable,ker2:size(2),ker2:size(1)))
      convo2:add(nn.Sum(1))
      convo2:add(nn.Replicate(nf))
      -- set kernel
      local fb = convo2.modules[2].weight
      for i=1,fb:size(1) do fb[i]:copy(ker2) end
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
   local convostd1 = nn.Sequential()
   convostd1:add(nn.SpatialPadding(self.padW,self.padW-self.kerWisPair,
                                   self.padH,self.padH-self.kerHisPair))
   convostd1:add(nn.SpatialConvolutionSparse(ctable,ker:size(2),ker:size(1)))
   convostd1:add(nn.Sum(1))
   convostd1:add(nn.Replicate(nf))
   -- set kernel
   local fb = convostd1.modules[2].weight
   for i=1,fb:size(1) do fb[i]:copy(ker) end
   -- set bias to 0
   convostd1.modules[2].bias:zero()

   -- 2nd ker ?
   if ker2 then
      local convostd2 = nn.Sequential()
      convostd2:add(nn.SpatialPadding(self.pad2W,self.pad2W-self.ker2WisPair,
                                      self.pad2H,self.pad2H-self.ker2HisPair))
      convostd2:add(nn.SpatialConvolutionSparse(ctable,ker2:size(2),ker2:size(1)))
      convostd2:add(nn.Sum(1))
      convostd2:add(nn.Replicate(nf))
      -- set kernel
      local fb = convostd2.modules[2].weight
      for i=1,fb:size(1) do fb[i]:copy(ker2) end
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
   self.subtractMod = nn.CSubTable()
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

function SpatialNormalization:forward(input)
   -- auto switch to 3-channel
   self.input = input
   if (input:nDimension() == 2) then
      self.input = torch.Tensor(1,input:size(1),input:size(2)):copy(input)
   end

   -- recompute coef only if necessary
   if (self.input:size(3) ~= self.coef:size(2)) or (self.input:size(2) ~= self.coef:size(1)) then
      local intVals = torch.Tensor(self.nfeatures,self.input:size(2),self.input:size(3)):fill(1)
      self.coef = self.convo:forward(intVals)
      self.coef = torch.Tensor():resizeAs(self.coef):copy(self.coef)
   end

   -- compute mean
   self.inConvo = self.convo:forward(self.input)
   self.inMean = self.meanDiviseMod:forward{self.inConvo,self.coef}
   self.inputZeroMean = self.subtractMod:forward{self.input,self.inMean}

   -- compute std dev
   self.inputZeroMeanSq = self.squareMod:forward(self.inputZeroMean)
   self.inConvoVar = self.convostd:forward(self.inputZeroMeanSq)
   self.inStdDevNotUnit = self.sqrtMod:forward(self.inConvoVar)
   self.inStdDev = self.stdDiviseMod:forward({self.inStdDevNotUnit,self.coef})
   local meanstd = self.inStdDev:mean()
   self.thresMod.threshold = self.fixedThres or math.max(meanstd,1e-3)
   self.thresMod.val = self.fixedThres or math.max(meanstd,1e-3)
   self.stdDev = self.thresMod:forward(self.inStdDev)

   --remove std dev
   self.diviseMod:forward{self.inputZeroMean,self.stdDev}
   self.output = self.diviseMod.output
   return self.output
end

function SpatialNormalization:backward(input, gradOutput)
   -- auto switch to 3-channel
   self.input = input
   if (input:nDimension() == 2) then
      self.input = torch.Tensor(1,input:size(1),input:size(2)):copy(input)
   end
   self.gradInput:resizeAs(self.input):zero()

   -- backprop all
   local gradDiv = self.diviseMod:backward({self.inputZeroMean,self.stdDev},gradOutput)
   local gradThres = gradDiv[2]
   local gradZeroMean = gradDiv[1]
   local gradinStdDev = self.thresMod:backward(self.inStdDev,gradThres)
   local gradstdDiv = self.stdDiviseMod:backward({self.inStdDevNotUnit,self.coef},gradinStdDev)
   local gradinStdDevNotUnit = gradstdDiv[1]
   local gradinConvoVar  = self.sqrtMod:backward(self.inConvoVar,gradinStdDevNotUnit)
   local gradinputZeroMeanSq = self.convostd:backward(self.inputZeroMeanSq,gradinConvoVar)
   gradZeroMean:add(self.squareMod:backward(self.inputZeroMean,gradinputZeroMeanSq))
   local gradDiff = self.subtractMod:backward({self.input,self.inMean},gradZeroMean)
   local gradinMean = gradDiff[2]
   local gradinConvoNotUnit = self.meanDiviseMod:backward({self.inConvo,self.coef},gradinMean)
   local gradinConvo = gradinConvoNotUnit[1]
   -- first part of the gradInput
   self.gradInput:add(gradDiff[1])
   -- second part of the gradInput
   self.gradInput:add(self.convo:backward(self.input,gradinConvo))
   return self.gradInput
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
   file:writeObject(self.subtractMod)
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
   file:writeInt(self.fixedThres or 0)
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
   self.subtractMod = file:readObject()
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
   self.fixedThres = file:readInt()
   if self.fixedThres == 0 then self.fixedThres = nil end
end
