local SpatialPadding, parent = torch.class('nn.SpatialPadding', 'nn.Module')

function SpatialPadding:__init(pad_l, pad_r, pad_t, pad_b)
   parent.__init(self)

   -- usage
   if not pad_l then
      error(xlua.usage('nn.SpatialPadding',
                          'a 2D padder module for images, zero-padding', nil,
                          {type='number', help='left padding', req=true},
                          {type='number', help='right padding'},
                          {type='number', help='top padding'},
                          {type='number', help='bottom padding'}))
   end

   self.pad_l = pad_l
   self.pad_r = pad_r or self.pad_l
   self.pad_t = pad_t or self.pad_l
   self.pad_b = pad_b or self.pad_l
end

function SpatialPadding:setPadding(pad_l, pad_r, pad_t, pad_b)
   self.pad_l = pad_l or 0
   self.pad_r = pad_r or self.pad_l
   self.pad_t = pad_t or self.pad_l
   self.pad_b = pad_b or self.pad_l
end

function SpatialPadding:forward(input)
   local h = input:size(2) + self.pad_t + self.pad_b
   local w = input:size(3) + self.pad_l + self.pad_r
   if w < 1 or h < 1 then error("Input too small") end
   self.output:resize(input:size(1), h, w)
   self.output:zero()
   -- crop input if necessary
   local c_input = input
   if self.pad_t < 0 then c_input = c_input:narrow(2, 1 - self.pad_t, c_input:size(2) + self.pad_t) end
   if self.pad_b < 0 then c_input = c_input:narrow(2, 1, c_input:size(2) + self.pad_b) end
   if self.pad_l < 0 then c_input = c_input:narrow(3, 1 - self.pad_l, c_input:size(3) + self.pad_l) end
   if self.pad_r < 0 then c_input = c_input:narrow(3, 1, c_input:size(3) + self.pad_r) end
   -- crop outout if necessary
   local c_output = self.output
   if self.pad_t > 0 then c_output = c_output:narrow(2, 1 + self.pad_t, c_output:size(2) - self.pad_t) end
   if self.pad_b > 0 then c_output = c_output:narrow(2, 1, c_output:size(2) - self.pad_b) end
   if self.pad_l > 0 then c_output = c_output:narrow(3, 1 + self.pad_l, c_output:size(3) - self.pad_l) end
   if self.pad_r > 0 then c_output = c_output:narrow(3, 1, c_output:size(3) - self.pad_r) end
   -- copy input to output
   c_output:copy(c_input)
   return self.output
end

function SpatialPadding:backward(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   -- crop gradInput if necessary
   local cg_input = self.gradInput
   if self.pad_t < 0 then cg_input = cg_input:narrow(2, 1 - self.pad_t, cg_input:size(2) + self.pad_t) end
   if self.pad_b < 0 then cg_input = cg_input:narrow(2, 1, cg_input:size(2) + self.pad_b) end
   if self.pad_l < 0 then cg_input = cg_input:narrow(3, 1 - self.pad_l, cg_input:size(3) + self.pad_l) end
   if self.pad_r < 0 then cg_input = cg_input:narrow(3, 1, cg_input:size(3) + self.pad_r) end
   -- crop gradOutout if necessary
   local cg_output = gradOutput
   if self.pad_t > 0 then cg_output = cg_output:narrow(2, 1 + self.pad_t, cg_output:size(2) - self.pad_t) end
   if self.pad_b > 0 then cg_output = cg_output:narrow(2, 1, cg_output:size(2) - self.pad_b) end
   if self.pad_l > 0 then cg_output = cg_output:narrow(3, 1 + self.pad_l, cg_output:size(3) - self.pad_l) end
   if self.pad_r > 0 then cg_output = cg_output:narrow(3, 1, cg_output:size(3) - self.pad_r) end
   -- copy gradOuput to gradInput
   cg_input:copy(cg_output)
   return self.gradInput
end


function SpatialPadding:write(file)
   parent.write(self, file)
   file:writeInt(self.pad_l)
   file:writeInt(self.pad_r)
   file:writeInt(self.pad_t)
   file:writeInt(self.pad_b)
end

function SpatialPadding:read(file)
   parent.read(self, file)
   self.pad_l = file:readInt()  
   self.pad_r = file:readInt()  
   self.pad_t = file:readInt()  
   self.pad_b = file:readInt()  
end

