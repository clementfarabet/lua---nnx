local SpatialNLLCriterion, parent = torch.class('nn.SpatialNLLCriterion', 'nn.Criterion')

function SpatialNLLCriterion:__init()
   parent.__init(self)
   self.fullOutput = torch.Tensor()
   self.resampleTarget = 1
   self.nbUpdates = -1     -- if -1, then the whole grad is used, if 1, then the center is used, else, 
                           -- N random points are sampled
end

function SpatialNLLCriterion:adjustTarget(input, target)
   -- preprocess target ?
   local s = self.resampleTarget
   if (target:size(1)*self.resampleTarget) ~= input:size(1) then
      local w = input:size(1)/s
      local x = math.floor((target:size(1) - (input:size(1)-1)*1/s)/2) + 1
      local h = input:size(2)/s
      local y = math.floor((target:size(2) - (input:size(1)-1)*1/s)/2) + 1
      target = target:narrow(1,x,w):narrow(2,y,h)
   end
   if s ~= 1 then
      local targets = torch.Tensor(input:size(1), input:size(2))
      image.scale(target, targets, 'simple')
      target = targets
   end
   self.target = target
   return target
end

SpatialNLLCriterion.forward_c = inline.load [[
      const void* torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
      THTensor *input  = luaT_checkudata(L, 2, torch_Tensor_id);
      THTensor *target  = luaT_checkudata(L, 3, torch_Tensor_id);
      THTensor *output = luaT_getfieldcheckudata(L, 1, "fullOutput", torch_Tensor_id);
      int width = target->size[0];
      int height = target->size[1];
      int x,y;
      for (y=0; y<height; y++) {
         for (x=0; x<width; x++) {
            THTensor_set2d(output, x, y, -THTensor_get3d(input, x, y, THTensor_get2d(target, x, y)-1) );
         }
      }
      return 1;
]]

function SpatialNLLCriterion:forward(input,target)
   target = self:adjustTarget(input, target)
   self.fullOutput:resizeAs(target)
   self:forward_c(input,target)
   self.output = self.fullOutput:sum()
   return self.output
end

SpatialNLLCriterion.backward_c = inline.load [[
      const void* torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
      THTensor *input  = luaT_checkudata(L, 2, torch_Tensor_id);
      THTensor *target  = luaT_checkudata(L, 3, torch_Tensor_id);
      THTensor *gradInput  = luaT_checkudata(L, 4, torch_Tensor_id);
      int width = target->size[0];
      int height = target->size[1];
      int x,y;
      for (y=0; y<height; y++) {
         for (x=0; x<width; x++) {
            THTensor_set3d(gradInput, x, y, THTensor_get2d(target, x, y)-1, -1);
         }
      }
      return 1;
]]

function SpatialNLLCriterion:backward(input,target)
   target = self:adjustTarget(input, target)
   self.gradInput:resizeAs(input):zero()
   if self.nbUpdates == -1 then
      self:backward_c(input,target,self.gradInput)
   elseif self.nbUpdates == 1 then
      self.fullGradInput = torch.Tensor() or self.fullGradInput
      self.fullGradInput:resizeAs(input):zero()
      self:backward_c(input,target,self.fullGradInput)
      local x = math.ceil(self.gradInput:size(1)/2)
      local y = math.ceil(self.gradInput:size(2)/2)
      self.gradInput[x][y]:copy( self.fullGradInput[x][y] )
   else
      self.fullGradInput = torch.Tensor() or self.fullGradInput
      self.fullGradInput:resizeAs(input):zero()
      self:backward_c(input,target,self.fullGradInput)
      for i = 1,self.nbUpdates do
         local x = math.random(1,self.gradInput:size(1))
         local y = math.random(1,self.gradInput:size(2))
         self.gradInput[x][y]:copy( self.fullGradInput[x][y] )
      end
   end
   return self.gradInput
end

function SpatialNLLCriterion.testme()
   local lsm = nn.ClassNLLCriterion()
   local slsm = nn.SpatialNLLCriterion()
   local inp = lab.rand(4,4,3)
   local target = torch.Tensor(4,4)
   target:apply(function(x) return math.random(1,3) end)
   print('input:',inp)
   print('target:',target)
   local outp = slsm:forward(inp,target)
   print('output:',outp)
   local outp_r = torch.Tensor(inp:size(1),inp:size(2))
   for i = 1,4 do
      for j = 1,4 do
         outp_r[i][j] = lsm:forward(inp[i][j],target[i][j])
      end
   end
   print('output (groundtruth):',outp_r:sum())
   local gradin = slsm:backward(inp,target)
   print('gradInput:',gradin)
   local gradin_r = torch.Tensor():resizeAs(gradin)
   for i = 1,4 do
      for j = 1,4 do
         gradin_r[i][j]:copy( lsm:backward(inp[i][j],target[i][j]) )
      end
   end
   print('gradInput (groundtruth):',gradin)
   print('error on output',math.abs(outp-outp_r:sum()))
   print('error on gradInput',(gradin-gradin_r):abs():sum())
end

function SpatialNLLCriterion:write(file)
   parent.write(self, file)
   file:writeDouble(self.resampleTarget)
   file:writeInt(self.nbUpdates)
end

function SpatialNLLCriterion:read(file)
   parent.read(self, file)
   self.resampleTarget= file:readDouble()
   self.nbUpdates = file:readInt()
   self.fullOutput = torch.Tensor()
end
