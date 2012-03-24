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
   self.focused_narrows1 = {}
   self.focused_narrows2 = {}
   self.focused_paddings = {}
   for i = 1,#self.ratios do
      local seq = nn.Sequential()
      seq:add(nn.Identity())
      seq:add(nn.Identity())
      seq:add(nn.Identity())
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
      self.focused = true
      self.x1 = {}
      self.x2 = {}
      self.y1 = {}
      self.y2 = {}
      for i = 1,#self.ratios do
	 local winWidth  = self.ratios[i] * ((w-1) * self.dW + self.kW)
	 local winHeight = self.ratios[i] * ((h-1) * self.dH + self.kH)
	 self.x1[i] = x-math.ceil (winWidth /2)+1
	 self.x2[i] = x+math.floor(winWidth /2)
	 self.y1[i] = y-math.ceil (winHeight/2)+1
	 self.y2[i] = y+math.floor(winHeight/2)
      end
   else
      self.focused = false
   end
end

function SpatialPyramid:configureFocus(wImg, hImg)
   for i = 1,#self.ratios do
      local x1 = self.x1[i]
      local x2 = self.x2[i]
      local y1 = self.y1[i]
      local y2 = self.y2[i]
      local px1 = 0
      local px2 = 0
      local py1 = 0
      local py2 = 0
      if x1 < 1 then
	 px1 = 1-x1
	 x1 = 1
      end
      if y1 < 1 then
	 py1 = 1-y1
	 y1 = 1
      end
      if x2 > wImg then
	 px2 = x2-wImg
	 x2 = wImg
      end
      if y2 > hImg then
	 py2 = y2-hImg
	 y2 = hImg
      end
      local focused_parallel = self.focused_pipeline.modules[3].modules[i]
      focused_parallel.modules[1] = nn.Narrow(2, y1, y2-y1+1)
      focused_parallel.modules[2] = nn.Narrow(3, x1, x2-x1+1)
      focused_parallel.modules[3] = nn.SpatialPadding(px1, px2, py1, py2)
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