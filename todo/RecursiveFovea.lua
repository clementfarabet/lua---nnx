local RecursiveFovea, parent = torch.class('nn.RecursiveFovea', 'nn.Module')

local help_desc =
[[
From a given image, generates a pyramid of scales, and process each scale
with the given list of preprocessors and processors. 
The result of each module/scale is upsampled to feed the next stage of recursion.

The pipeline is the following:
input  ->  pyramid{ratios}  ->  preProcessors  ->  padding  ->  processors[1]  -> output_1
|,-- <----------------- <----'
`----->  processors[2]  -> output_2
|,-- <----------------- <----'
`----->       ...       -> 
|,-- <----------------- <----'
`----->  processors[N]  -> output_N -> output

There are two operating modes: focused [training?], and global [inference].

In inference mode, the entire input pyramid is processed, to produce
an answer that is has large as the original input.

In sampling mode, the fovea is first focused on a particular
(x,y) point.  To focus the fovea, simply call
fovea:focus(x,y,winSize) before doing a forward.  A call to
fovea:focus(nil) makes it unfocus (go back to global mode).

Optionally, a list of criterions can be provided, to estimate
the current error with respect to some target vectors. If
criterions are provided, then target vectors can be provided as
well to the forward and backward functions.]]

function RecursiveFovea:__init(...)
   parent.__init(self)
   -- check args
   xlua.unpack_class(
      self,
      {...},
      'nn.RecursiveFovea',
      help_desc,
      {arg='nInputPlane', type='number',
       help='number of input planes', 
       req=true},
      {arg='nRecursivePlane', type='number', 
       help='number of recursive planes (e.g. nb of output planes used by next input stage', req=true},
      {arg='ratios', type='table', 
       help='list of downsampling ratios, in decreasing order', 
       req=true},
      {arg='processors', type='table', 
       help='list of processors (each processor sees a single scale)', 
       req=true},
      {arg='preProcessors', type='table', 
       help='list of preprocessors (applied before padding)'},
      {arg='postProcessors', type='table', 
       help='list of postprocessors (applied before criterions, and on last stage)'},
      {arg='criterions', type='table', 
       help='list of criterions (applied to each intermediate output)'},
      {arg='fov', type='number', 
       help='field of view (== processors\' receptive field)', default=1},
      {arg='batchSize', type='number', 
       help='size of mini-batch used when computing gradient [default = fov/sub]'},
      {arg='sub', type='number', 
       help='global subsampling (== processors\' subsampling ratio)', default=1},
      {arg='scale_target_values', type='boolean', 
       help='scale the target values as well as dimension',default=false},
      {arg='verbose', type='boolean', 
       help='prints a lot of information', default=true}
   )

   -- batchSize ?
   self.batchSize = self.batchSize or self.fov / self.sub

   -- internal modules:
   self.downsamplers = {}
   self.padders = {}
   self.processors = self.processors or {}
   self.upsamplers = {}
   self.upsampledPadders = {}
   self.preProcessors = self.preProcessors or {}
   self.postProcessors = self.postProcessors or {} -- todo
   self.criterions = self.criterions or {}

   -- temporary results:
   self.pyramid = {}
   self.preProcessed = {}
   self.padded = {}
   self.narrowed = {}
   self.concatenated = {}
   self.processed = {}
   self.upsampled = {}
   self.upsampledPadded = {}
   self.upsampledNarrowed = {}
   self.postProcessed = {}
   self.predicted = {}

   self.gradPostProcessed = {}
   self.gradUpsampledNarrowed = {}
   self.gradUpsampledPadded = {}
   self.gradUpsampled = {}
   self.gradProcessed = {}
   self.gradConcatenated = {}
   self.gradNarrowed = {}
   self.gradPadded = {}
   self.gradPreProcessed = {}
   self.gradPyramid = {}
   self.gradPredicted = {}

   -- hold current targets, if criterions are in use
   self.targets_scaled = {}
   self.targets = {}

   -- check preprocessors/processors/criterions
   if #self.processors ~= #self.ratios then
      xerror('the number of processors provided should == the number of ratios (scales): ' .. #self.ratios,
             'nn.RecursiveFovea')
   end
   if self.preProcessors[1] and #self.preProcessors ~= #self.ratios then
      xerror('the number of preProcessors provided should == the number of ratios (scales): ' .. #self.ratios,
             'nn.RecursiveFovea')
   end
   if self.criterions[1] and #self.criterions ~= #self.ratios then
      xerror('the number of criterions provided should == the number of ratios (scales): ' .. #self.ratios,
             'nn.RecursiveFovea')
   end

   -- sort scales, in decreasing order
   table.sort(self.ratios, function(a,b) return a>b end)

   -- info
   if self.verbose then
      print(self)
   end
end

function RecursiveFovea:configure(fov, sub, input_w, input_h)
   -- don't reconfigure if params have not changed
   if fov == self.fov and input_w == self.input_w and input_h == self.input_h then
      return
   end
   self.fov = fov
   self.sub = sub
   self.input_w = input_w
   self.input_h = input_h

   local ratios = self.ratios
   local nscales = #ratios
   local focused = self.focused

   -- generate lists of all sizes
   local pyramid = {w={},h={}}
   local padded = {w={},h={}}
   local narrowed = {w={},h={}}
   local processed = {w={},h={}}

   -- using resampling if ratios are not integer
   self.bilinear = false
   for idx = 1,nscales do
      if ratios[idx] < 1 then
         xerror('ratios should be >= 1','nn.RecursiveFovea')
      elseif ratios[idx] ~= math.floor(ratios[idx]) then
         self.bilinear = true
      end
   end

   -- compute intermediate sizes
   for idx = nscales,1,-1 do
      -- check order
      if idx > 1 and self.ratios[idx] > self.ratios[idx-1] then
         xerror('downsampling ratios should be provided in decreasing order, for proper coarse-to-fine recursion',
                'nn.RecursiveFovea')
      end
      -- pyramid size
      pyramid[idx] = {w = math.floor(input_w / ratios[idx]), h = math.floor(input_h / ratios[idx])}
      if idx == nscales then
         -- infer processed size
         processed[idx] = {w = math.floor(input_w / ratios[idx] / sub), h = math.floor(input_h / ratios[idx] / sub)}
         -- infer narrowed size
         narrowed[idx] = {w = processed[idx].w*sub + fov - sub, h = processed[idx].h*sub + fov - sub}
         -- and padded size
         padded[idx] = narrowed[idx] --{w = pyramid[idx].w + fov - 1, h = pyramid[idx].h + fov - 1}
      else
         -- infer processed size from next stage in recursion
         processed[idx] = {w = math.ceil(math.ceil(pyramid[idx+1].w * ratios[idx+1]/ratios[idx] / sub)),
                           h = math.ceil(math.ceil(pyramid[idx+1].h * ratios[idx+1]/ratios[idx] / sub))}
         -- infer narrowed size
         narrowed[idx] = {w = processed[idx].w*sub + fov - sub, h = processed[idx].h*sub + fov - sub}
         -- and padded size
         padded[idx] = narrowed[idx] --{w = pyramid[idx].w + fov - 1, h = pyramid[idx].h + fov - 1}
      end
   end

   -- configure downsamplers, padders and upsamplers
   for idx = 1,nscales do
      -- downsamplers (for pyramid)
      local r = ratios[idx]
      if self.bilinear then
         self.downsamplers[idx] = nn.SpatialReSampling(pyramid[idx].w, pyramid[idx].h)
      else
         self.downsamplers[idx] = nn.SpatialSubSampling(self.nInputPlane, r, r, r, r)
         self.downsamplers[idx].weight:fill(1/(r^2))
         self.downsamplers[idx].bias:zero()
      end

      -- padders
      local padw = (padded[idx].w - pyramid[idx].w)
      local padh = (padded[idx].h - pyramid[idx].h)
      local padl = math.floor(padw/2)
      local padr = padw - padl + 1
      local padt = math.floor(padh/2)
      local padb = padh - padt + 1
      self.padders[idx] = nn.SpatialPadding(padl, padr, padt, padb)
      self.upsampledPadders[idx] = nn.SpatialPadding(padl, padr, padt, padb)

      -- upsamplers
      if idx < nscales then
         local upw = (ratios[idx] / ratios[idx+1]) * sub
         local uph = (ratios[idx] / ratios[idx+1]) * sub
         if self.bilinear then
            self.upsamplers[idx] = nn.SpatialReSampling(pyramid[idx+1].w, pyramid[idx+1].h)
         else
            self.upsamplers[idx] = nn.SpatialUpSampling(self.nInputPlane, upw, uph)
         end
      end
   end

   -- store results
   self.pyramid_size = pyramid
   self.padded_size = padded
   self.narrowed_size = narrowed
   self.processed_size = processed
   
   -- info
   if self.debug then
      print('')
      xprint('reconfig complete:','nn.RecursiveFovea')
      for idx = 1,nscales do
         print('scale ' .. idx .. ' :')
         print('  + pyramid   > ' .. pyramid[idx].w .. 'x' .. pyramid[idx].h)
         print('  + padded    > ' .. padded[idx].w .. 'x' .. padded[idx].h)
         print('  + narrowed  > ' .. narrowed[idx].w .. 'x' .. narrowed[idx].h)
         print('  + processed > ' .. processed[idx].w .. 'x' .. processed[idx].h)
      end
      print('')
   end
end

function RecursiveFovea:__tostring__()
   local str = 'nn.RecursiveFovea:\n'
   str = str .. '  + number of recursion stages : '..(#self.ratios) .. '\n'
   str = str .. '  + downsampling ratios (scales) :'
   for idx = 1,#self.ratios do
      str = str .. '   ' .. self.ratios[idx]
   end
   str = str .. '\n'
   str = str .. '  + processors\' field of view : '..(self.fov)..'x'..(self.fov)..'\n'
   str = str .. '  + processors\' downsampling ratio : '..(self.sub)..'x'..(self.sub)..'\n'
   if self.criterions[1] then
      str = str .. '  + using training criterions : ' .. torch.typename(self.criterions[1]) .. '\n'
   end
   str = str .. '  + verbose : ' .. tostring(self.verbose)
   return str
end

function RecursiveFovea:focus(x,y)
   -- fprop and bprop sizes must be different 
   -- * frop must create an output which will be upsampled to batchsize+fov
   -- * bprop creates a batchsize+fov sized input around center of focus 
   -- fov keeps track of the padding
   -- batchSize keeps track of the neighboring samples which will be bproped together
   -- in the simple case everything is fproped
   self.x = x
   self.y = y
   if self.x and self.y then 
      self.focused = true
   else 
      self.focused = false
      return
   end
   local corners = {}
   for idx = 1,#self.ratios do
      -- compute the center of focus at each scale, taking into account the ratios/downsampling effect
      local ox = math.floor(math.floor((self.x-1) / self.ratios[idx]) / self.sub) * self.sub + 1
      local oy = math.floor(math.floor((self.y-1) / self.ratios[idx]) / self.sub) * self.sub + 1
      -- remap these centers to become corners
      ox = ox - math.ceil(self.batchSize/2) + 1
      oy = oy - math.ceil(self.batchSize/2) + 1
      -- append
      table.insert(corners, {x=ox,y=oy})
   end
   self.corners = corners
end

function RecursiveFovea:forward(input,target,x,y)
   -- input must be 3D
   if input:nDimension() ~= 3 or input:size(3) ~= self.nInputPlane then
      xerror('input must be 3d and have ' .. self.nInputPlane .. ' input planes','nn.RecursiveFovea')
   end

   -- focus ?
   if x and y then
      self:focus(x,y)
   end

   -- configure fovea for given input and current parameters
   local width = input:size(1)
   local height = input:size(2)
   local nmaps = input:size(3)
   local nscales = #self.ratios
   local fov = self.fov
   local sub = self.sub
   local corners = self.corners
   self:configure(fov, sub, width, height)

   -- (1-2) create preprocessed pyramid
   for idx = 1,nscales do
      -- (1) generate pyramid
      self.pyramid[idx] = self.downsamplers[idx]:forward(input)

      -- (2) preprocess
      if self.preProcessors[idx] then
         self.preProcessed[idx] = self.preProcessors[idx]:forward(self.pyramid[idx])
      else
         self.preProcessed[idx] = self.pyramid[idx]
      end
   end

   -- (3-7) walk through recursion
   for idx = 1,nscales do
      -- (3) pad inputs
      self.padded[idx] = self.padders[idx]:forward(self.preProcessed[idx])

      -- (4) is fovea focused ?
      self.narrowed[idx] 
         = self.padded[idx]:narrow(1,1,self.narrowed_size[idx].w):narrow(2,1,self.narrowed_size[idx].h)

      -- (5) concatenate current input and upsampled result from previous stage in the recursion
      self.concatenated[idx] = self.concatenated[idx] or torch.Tensor()
      self.concatenated[idx]:resize(self.narrowed[idx]:size(1), self.narrowed[idx]:size(2),
                                    self.narrowed[idx]:size(3) + self.nRecursivePlane)
      self.concatenated[idx]:narrow(3,1,self.narrowed[idx]:size(3)):copy(self.narrowed[idx])
      if idx > 1 then
         local p = self.concatenated[idx]:narrow(3,self.narrowed[idx]:size(3)+1,
                                                 self.nRecursivePlane)
         p:copy(self.upsampledNarrowed[idx-1])
         if self.scale_target_values then
            local r = self.ratios[idx-1]/self.ratios[idx]
            p:mul(r)
         end
         
      else
         self.concatenated[idx]:narrow(3,self.narrowed[idx]:size(3)+1,self.nRecursivePlane):zero()
      end

      -- (6) apply processors to pyramid
      self.processed[idx] = self.processors[idx]:forward(self.concatenated[idx])

      -- (7) upsample, pad and narrow, for next stage
      if idx < nscales then
         -- (7.a)
         self.upsampled[idx] = self.upsamplers[idx]:forward(self.processed[idx])

         -- (7.b)
         self.upsampledPadded[idx] = self.upsampledPadders[idx]:forward(self.upsampled[idx])

         -- (7.c)
         self.upsampledNarrowed[idx]
            = self.upsampledPadded[idx]:narrow(1,1,self.narrowed_size[idx+1].w):narrow(2,1,self.narrowed_size[idx+1].h)
      end
   end

   -- (8) optional post processors
   for idx = 1,nscales do
      if self.postProcessors[idx] then
         self.postProcessed[idx] = self.postProcessors[idx]:forward(self.processed[idx])
      else
         self.postProcessed[idx] = self.processed[idx]
      end
   end

   -- DEBUG
   if self.debug then
      for idx = 1,nscales do
         print('')
         print('scale ' .. idx .. ' :')
         print('  + pyramid   > ' .. self.pyramid[idx]:size(1) .. 'x' .. self.pyramid[idx]:size(2))
         print('  + padded    > ' .. self.padded[idx]:size(1) .. 'x' .. self.padded[idx]:size(2))
         print('  + narrowed  > ' .. self.narrowed[idx]:size(1) .. 'x' .. self.narrowed[idx]:size(2))
         print('  + processed > ' .. self.processed[idx]:size(1) .. 'x' .. self.processed[idx]:size(2))
      end
   end

   -- (9) if criterions are provided, compute their errors
   local error = 0
   if self.criterions[1] and target then
      for idx = 1,nscales do
         -- generate the target vector for each scale
         local ratio = self.ratios[idx]
         local target_w = self.postProcessed[idx]:size(1)
         local target_h = self.postProcessed[idx]:size(2)
         self.targets_scaled[idx] = self.targets_scaled[idx] or torch.Tensor()
         if target:nDimension() == 3 then
            self.targets_scaled[idx]:resize(self.processed[idx]:size(1), self.processed[idx]:size(2), target:size(3))
         else
            self.targets_scaled[idx]:resize(self.processed[idx]:size(1), self.processed[idx]:size(2))
         end
         image.scale(target, self.targets_scaled[idx], 'simple')
         -- in the case of flow the value of the target is absolute a
         -- 20px shift at scale 1 is a 10px shift at scale 2, this
         -- changes the dimension of the target.  Need some sensibly
         -- named tag for this
         if self.scale_target_values then
            local ts = self.targets_scaled[idx]
            ts:mul(1/self.ratios[idx])
         end
         if self.focused then
            local bs = self.batchSize
            -- adjust focus point for each scale
            corners[idx].x = math.min(math.max(corners[idx].x, 1), self.postProcessed[idx]:size(1)-bs+1)
            corners[idx].y = math.min(math.max(corners[idx].y, 1), self.postProcessed[idx]:size(2)-bs+1)
            
            -- then crop/extract mini batch patch on both targets and postProcessed vectors
            self.predicted[idx] = self.postProcessed[idx]:narrow(1,corners[idx].x,bs):narrow(2,corners[idx].y,bs)
            self.targets[idx] = self.targets_scaled[idx]:narrow(1,corners[idx].x,bs):narrow(2,corners[idx].y,bs)
         else
            self.predicted[idx] = self.postProcessed[idx]
            self.targets[idx] = self.targets_scaled[idx]
         end
         -- then evaluate the criterion's error
         error = error + self.criterions[idx]:forward(self.predicted[idx], self.targets[idx])
      end

      -- normalize error
      error = error / nscales

      -- DEBUG
      if self.debug then
         print('')
         xprint('adjusted focus points','nn.RecursiveFovea')
         for idx = 1,nscales do
            if self.focused then
               print('  + at scale ' .. idx .. ', focused on corner: ' .. corners[idx].x .. ',' .. corners[idx].y ..
                     ' with a batch size: '..self.batchSize)
            end
         end
      end
   end

   -- (10) return output (last stage in the recursion)
   self.output = self.postProcessed[nscales]
   return self.output, error
end

function RecursiveFovea:backward(input)
   -- local params
   local nscales = #self.ratios
   local fov = self.fov
   local sub = self.sub
   local corners = self.corners

   -- (9) backprop through criterions using generated targets (from prev forward call)
   for idx = 1,nscales do
      -- bprop through criterion
      self.gradPredicted[idx] = self.criterions[idx]:backward(self.predicted[idx], self.targets[idx])

      -- then remap partial grad vector 
      self.gradPostProcessed[idx] = self.gradPostProcessed[idx] or torch.Tensor()
      self.gradPostProcessed[idx]:resizeAs(self.postProcessed[idx]):zero()
      if self.focused then
         local bs = self.batchSize
         self.gradPostProcessed[idx]:narrow(1,corners[idx].x,bs):narrow(2,corners[idx].y,bs):copy(self.gradPredicted[idx])
      else
         self.gradPostProcessed[idx]:copy(self.gradPredicted[idx])
      end
   end

   -- (8) backprop through post processors
   for idx = 1,nscales do
      if self.postProcessors[idx] then
         self.gradProcessed[idx] = self.postProcessors[idx]:backward(self.processed[idx], self.gradPostProcessed[idx])
      else
         self.gradProcessed[idx] = self.gradPostProcessed[idx]
      end
   end

   -- (7) recursive gradient: not done for now (needs to see if it's really worth it)
   --

   -- (6) backprop through processors
   for idx = 1,nscales do
      self.gradConcatenated[idx] = self.processors[idx]:backward(self.concatenated[idx], self.gradProcessed[idx])
   end

   -- (5) bprop through concatenators
   for idx = 1,nscales do
      self.gradNarrowed[idx] = self.gradConcatenated[idx]:narrow(3, 1, self.narrowed[idx]:size(3))
   end

   -- (4) bprop through narrow
   for idx = 1,nscales do
      self.gradPadded[idx] = self.gradPadded[idx] or torch.Tensor()
      self.gradPadded[idx]:resizeAs(self.padded[idx]):zero()
      self.gradPadded[idx]:narrow(1,1,self.narrowed_size[idx].w):narrow(2,1,self.narrowed_size[idx].h):copy(self.gradNarrowed[idx])
   end

   -- (3) bprop through padders
   for idx = 1,nscales do
      self.gradPreProcessed[idx] = self.padders[idx]:backward(self.preProcessed[idx], self.gradPadded[idx])
   end

   -- (2) bprop through preProcessors
   for idx = 1,nscales do
      if self.preProcessors[idx] then
         self.gradPyramid[idx] = self.preProcessors[idx]:backward(self.pyramid[idx], self.gradPreProcessed[idx])
      else
         self.gradPyramid[idx] = self.gradPreProcessed[idx]
      end
   end

   -- (1) bprop through pyramid
   self.gradInput:resizeAs(input):zero()
   for idx = 1,nscales do
      local partialGrad = self.downsamplers[idx]:backward(input, self.gradPyramid[idx])
      self.gradInput:add(partialGrad)
   end
   return self.gradInput
end

function RecursiveFovea:reset(stdv)
   for idx = 1,#self.processors do
      self.processors[idx]:reset(stdv)
   end
end

function RecursiveFovea:zeroGradParameters(momentum)
   for idx = 1,#self.processors do
      self.processors[idx]:zeroGradParameters(momentum)
   end
end

function RecursiveFovea:updateParameters(learningRate)
   for idx = 1,#self.processors do
      self.processors[idx]:updateParameters(learningRate)
   end
end

function RecursiveFovea:decayParameters(decay)
   for idx = 1,#self.processors do
      self.processors[idx]:decayParameters(decay)
   end
end

function RecursiveFovea:write(file)
   parent.write(self, file)
   -- params
   file:writeInt(self.nInputPlane)
   file:writeInt(self.nRecursivePlane)
   file:writeInt(self.fov)
   file:writeInt(self.sub)
   file:writeObject(self.ratios)
   -- modules
   file:writeObject(self.downsamplers)
   file:writeObject(self.padders)
   file:writeObject(self.upsamplers)
   file:writeObject(self.upsampledPadders)
   file:writeObject(self.processors)
   file:writeObject(self.preProcessors)
   file:writeObject(self.postProcessors)
   file:writeObject(self.criterions)
   -- states
   file:writeObject(self.pyramid)
   file:writeObject(self.preProcessed)
   file:writeObject(self.padded)
   file:writeObject(self.narrowed)
   file:writeObject(self.concatenated)
   file:writeObject(self.processed)
   file:writeObject(self.upsampled)
   file:writeObject(self.upsampledPadded)
   file:writeObject(self.upsampledNarrowed)
   file:writeObject(self.postProcessed)
   file:writeObject(self.predicted)
end

function RecursiveFovea:read(file)
   parent.read(self, file)
   -- params
   self.nInputPlane = file:readInt()
   self.nRecursivePlane = file:readInt()
   self.fov = file:readInt()
   self.sub = file:readInt()
   self.ratios = file:readObject()
   self.batchSize = self.fov
   -- modules
   self.downsamplers = file:readObject()
   self.padders = file:readObject()
   self.upsamplers = file:readObject()
   self.upsampledPadders = file:readObject()
   self.processors = file:readObject()
   self.preProcessors = file:readObject()
   self.postProcessors = file:readObject()
   self.criterions = file:readObject()
   -- states
   self.pyramid = file:readObject()
   self.preProcessed = file:readObject()
   self.padded = file:readObject()
   self.narrowed = file:readObject()
   self.concatenated = file:readObject()
   self.processed = file:readObject()
   self.upsampled = file:readObject()
   self.upsampledPadded = file:readObject()
   self.upsampledNarrowed = file:readObject()
   self.postProcessed = file:readObject()
   self.predicted = file:readObject()
   -- grad states
   self.gradPostProcessed = {}
   self.gradUpsampledNarrowed = {}
   self.gradUpsampledPadded = {}
   self.gradUpsampled = {}
   self.gradProcessed = {}
   self.gradConcatenated = {}
   self.gradNarrowed = {}
   self.gradPadded = {}
   self.gradPreProcessed = {}
   self.gradPyramid = {}
   self.gradPredicted = {}
   self.targets_scaled = {}
   self.targets = {}
end
