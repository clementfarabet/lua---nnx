local SpatialMSECriterion, parent = torch.class('nn.SpatialMSECriterion', 'nn.Criterion')

function SpatialMSECriterion:__init()
   parent.__init(self)
   self.fullOutput = torch.Tensor()
   self.resampleTarget = 1
   self.nbUpdates = -1     -- if -1, then the whole grad is used, if 1, then the center is used, else, 
                           -- N random points are sampled
   self.distribution = false -- if true, then targets are integrated over the input, to produce distributions
   self.ignoreClass = false -- if set, then this class is ignored (its target is always pulled down to -1)
end

SpatialMSECriterion.createTarget_c = inline.load [[
      const void* torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
      THTensor *new  = luaT_checkudata(L, 1, torch_Tensor_id);
      THTensor *old  = luaT_checkudata(L, 2, torch_Tensor_id);
      int width = new->size[0];
      int height = new->size[1];
      int k,x,y;
      for (y=0; y<height; y++) {
         for (x=0; x<width; x++) {
            THTensor_set3d(new, x, y, THTensor_get2d(old, x, y)-1, 1);
         }
      }
      return 1;
]]

SpatialMSECriterion.createDistribution_c = inline.load [[
      const void* torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
      THTensor *distribution  = luaT_checkudata(L, 1, torch_Tensor_id);
      THTensor *raw  = luaT_checkudata(L, 2, torch_Tensor_id);
      int fov_w = lua_tonumber(L, 3);
      int fov_h = lua_tonumber(L, 4);
      int width = distribution->size[0];
      int height = distribution->size[1];
      int x,y,i,j;
      for (y=0; y<height; y++) {
         for (x=0; x<width; x++) {
            for (i=0; i<fov_h; i++) {
               for (j=0; j<fov_w; j++) {
                  int k = THTensor_get2d(raw, x+j, y+i)-1;
                  THTensor_set3d(distribution, x, y, k, THTensor_get3d(distribution, x, y, k) + 1);
               }
            }
         }
      }
      return 1;
]]

function SpatialMSECriterion:adjustTarget(input, target)
   -- preprocess target ?
   local rawtarget = target
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
   if target:nDimension() == 2 then
      -- missing 3rd dimension: assuming indexed targets
      -- we create fake vectors to emulate classical mean-square regression problem
      self.newtarget = self.newtarget or torch.Tensor()
      self.newtarget:resizeAs(input):fill(-1)
      self.createTarget_c(self.newtarget, target)
      target = self.newtarget
   end
   if self.distribution then
      -- replace targets by distributions of targets in the given receptive field
      local fov_w = (rawtarget:size(1) - target:size(1))
      local fov_h = (rawtarget:size(2) - target:size(2))
      target:zero()
      self.createDistribution_c(target,rawtarget,fov_w,fov_h)
      if self.ignoreClass then
         target:select(3, self.ignoreClass):zero()
      end
      target:div(fov_h*fov_w/2):add(-1)
   elseif self.ignoreClass then
      target:select(3, self.ignoreClass):fill(-1)
   end
   self.target = target
   return target
end

SpatialMSECriterion.forward_c = inline.load [[
      const void* torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
      THTensor *input  = luaT_checkudata(L, 2, torch_Tensor_id);
      THTensor *target  = luaT_checkudata(L, 3, torch_Tensor_id);
      THTensor *output = luaT_getfieldcheckudata(L, 1, "fullOutput", torch_Tensor_id);
      double z = 0;
      TH_TENSOR_APPLY3(double, output, double, input, double, target,
                       z = (*input_p - *target_p);
                       *output_p =  0.5*z*z;)
      return 1;
]]

function SpatialMSECriterion:forward(input,target)
   target = self:adjustTarget(input, target)
   self.fullOutput:resizeAs(input)
   self:forward_c(input,target)
   if self.meanError then
      self.output = self.fullOutput:mean()
   else
      self.output = self.fullOutput:sum()
   end
   return self.output
end

SpatialMSECriterion.backward_c = inline.load [[
      const void* torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
      THTensor *input  = luaT_checkudata(L, 2, torch_Tensor_id);
      THTensor *target  = luaT_checkudata(L, 3, torch_Tensor_id);
      THTensor *gradInput  = luaT_checkudata(L, 4, torch_Tensor_id);
      TH_TENSOR_APPLY3(double, gradInput, double, input, double, target,
                       *gradInput_p = (*input_p - *target_p);)
      return 1;
]]

function SpatialMSECriterion:backward(input,target)
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

function SpatialMSECriterion.testme()
   local inp = lab.rand(4,4,1)
   local tgt = torch.Tensor(4,4,1):copy(inp)
   tgt[2][3][1] = tgt[2][3][1] + 0.1
   local crt = nn.SpatialMSECriterion()
   print("input:")
   print(inp)
   print("target:")
   print(tgt)
   local out = crt:forward(inp,tgt)
   print("forward: output")
   print(out)
   local gIn = crt:backward(inp,tgt)
   print("backward: gradInput")
   print(gIn)
end

function SpatialMSECriterion:write(file)
   parent.write(self, file)
   file:writeDouble(self.resampleTarget)
   file:writeInt(self.nbUpdates)
   file:writeBool(self.distribution)
   file:writeInt(self.ignoreClass)
   file:writeBool(self.meanError)
end

function SpatialMSECriterion:read(file)
   parent.read(self, file)
   self.resampleTarget= file:readDouble()
   self.nbUpdates = file:readInt()
   self.fullOutput = torch.Tensor()
   self.distribution = file:readBool()
   self.ignoreClass = file:readInt()
   self.meanError = file:readBool()
end
