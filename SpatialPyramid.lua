local SpatialPyramid, parent = torch.class('nn.SpatialPyramid', 'nn.Module')

local help_desc = [[
Simplified (and more flexible regarding sizes) fovea:
From a given image, generates a pyramid of scales, and process each scale
with the given list of processors. 
The result of each module/scale is then
upsampled to produce a homogenous list of 3D feature maps grouping the different scales.

There are two operating modes: focused [mostly training], and global [inference]. 

In global mode,
the entire input is processed.

In focused mode, the fovea is first focused on a particular (x,y) point.
This function has two additional parameters, w and h, that represent the size
of the OUTPUT of the processors.
To focus the fovea, simply call fovea:focus(x,y,w,h) before doing a forward.
A call to fovea:focus(nil) makes it unfocus (go back to global mode). ]]

function SpatialPyramid:__init(ratios, processors, kW, kH, dW, dH)
   parent.__init(self)
   assert(#ratios == #processors)
   
   self.ratios = ratios
   self.kH = kH
   self.kW = kW
   self.dH = dH
   self.dW = dW
   self.focused = false
   self.x = 0
   self.y = 0
   self.wFocus = 0
   self.hFocus = 0
   self.processors = processors

   local wPad = kW-dW
   local hPad = kH-dH
   self.padLeft   = math.floor(wPad/2)
   self.padRight  = math.ceil (wPad/2)
   self.padTop    = math.floor(hPad/2)
   self.padBottom = math.ceil (hPad/2)

   -- focused
   self.focused_pipeline = nn.Sequential()
   self.focused_pipeline:add(nn.Replicate(#self.ratios))
   self.focused_pipeline:add(nn.SplitTable(1))
   local focused_parallel = nn.ParallelTable()
   self.focused_pipeline:add(focused_parallel)
   self.focused_pipeline:add(nn.JoinTable(1))
   for i = 1,#self.ratios do
      local seq = nn.Sequential()
      seq:add(nn.SpatialPadding(0,0,0,0))
      seq:add(nn.SpatialDownSampling(self.ratios[i], self.ratios[i]))
      seq:add(processors[i])
      focused_parallel:add(seq)
   end

   -- unfocused
   self.unfocused_pipeline = nn.Sequential()
   self.unfocused_pipeline:add(nn.Replicate(#self.ratios))
   self.unfocused_pipeline:add(nn.SplitTable(1))
   local unfocused_parallel = nn.ParallelTable()
   self.unfocused_pipeline:add(unfocused_parallel)
   self.unfocused_pipeline:add(nn.JoinTable(1))
   for i = 1,#self.ratios do
      local seq = nn.Sequential()
      seq:add(nn.SpatialDownSampling(self.ratios[i], self.ratios[i]))
      seq:add(nn.SpatialPadding(self.padLeft, self.padRight, self.padTop, self.padBottom))
      seq:add(processors[i])
      seq:add(nn.SpatialUpSampling(self.ratios[i], self.ratios[i]))
      unfocused_parallel:add(seq)
   end
end

function SpatialPyramid:focus(x, y, w, h)
   w = w or 1
   h = h or 1
   if x and y then
      self.x = x
      self.y = y
      self.focused = true
      self.winWidth = {}
      self.winHeight = {}
      for i = 1,#self.ratios do
	 self.winWidth[i]  = self.ratios[i] * ((w-1) * self.dW + self.kW)
	 self.winHeight[i] = self.ratios[i] * ((h-1) * self.dH + self.kH)
      end
   else
      self.focused = false
   end
end

function SpatialPyramid:configureFocus(wImg, hImg)
   for i = 1,#self.ratios do
      local focused_parallel = self.focused_pipeline.modules[3].modules[i]
      focused_parallel.modules[1].pad_l = -self.x + math.ceil (self.winWidth[i] /2)
      focused_parallel.modules[1].pad_r =  self.x + math.floor(self.winWidth[i] /2) - wImg
      focused_parallel.modules[1].pad_t = -self.y + math.ceil (self.winHeight[i]/2)
      focused_parallel.modules[1].pad_b =  self.y + math.floor(self.winHeight[i]/2) - hImg
   end
end   

function SpatialPyramid:checkSize(input)
   for i = 1,#self.ratios do
      if (math.mod(input:size(2), self.ratios[i]) ~= 0) or
         (math.mod(input:size(3), self.ratios[i]) ~= 0) then
         print('SpatialPyramid: input sizes must be multiple of ratios')
	 assert(false)
      end
   end
end
 
function SpatialPyramid:updateOutput(input)
   self:checkSize(input)
   if self.focused then
      self:configureFocus(input:size(3), input:size(2))
      self.output = self.focused_pipeline:updateOutput(input)
   else
      self.output = self.unfocused_pipeline:updateOutput(input)
   end
   return self.output
end

function SpatialPyramid:updateGradInput(input, gradOutput)
   if self.focused then
      self.gradInput = self.focused_pipeline:updateGradInput(input, gradOutput)
   else
      self.gradInput = self.unfocused_pipeline:updateGradInput(input, gradOutput)
   end
   return self.gradInput
end

function SpatialPyramid:zeroGradParameters()
   self.focused_pipeline:zeroGradParameters()
   self.unfocused_pipeline:zeroGradParameters()
end

function SpatialPyramid:accGradParameters(input, gradOutput, scale)
   if self.focused then
      self.focused_pipeline:accGradParameters(input, gradOutput, scale)
   else
      self.unfocused_pipeline:accGradParameters(input, gradOutput, scale)
   end
end

function SpatialPyramid:updateParameters(learningRate)
   if self.focused then
      self.focused_pipeline:updateParameters(learningRate)
   else
      self.unfocused_pipeline:updateParameters(learningRate)
   end
end

function SpatialPyramid:type(type)
   parent.type(self, type)
   self.focused_pipeline:type(type)
   self.unfocused_pipeline:type(type)
   return self
end

function SpatialPyramid:parameters()
   if self.focused then
      return self.focused_pipeline:parameters()
   else
      return self.unfocused_pipeline:parameters()
   end
end

function SpatialPyramid:__tostring__()
   if self.focused then
      local dscr = tostring(self.focused_pipeline):gsub('\n', '\n    |    ')
      return 'SpatialPyramid (focused)\n' .. dscr
   else
      local dscr = tostring(self.unfocused_pipeline):gsub('\n', '\n    |    ')
      return 'SpatialPyramid (unfocused)\n' .. dscr
   end
end