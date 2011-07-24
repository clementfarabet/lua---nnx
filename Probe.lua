local Probe, parent = torch.class('nn.Probe', 'nn.Module')

function Probe:__init(...)
   parent.__init(self)
   xlua.unpack_class(self, {...}, 'nn.Probe', 
                     'print/display input/gradients of a network',
                     {arg='name', type='string', help='unique name to identify probe', req=true},
                     {arg='print', type='boolean', help='print full tensor', default=false},
                     {arg='display', type='boolean', help='display tensor', default=false},
                     {arg='size', type='boolean', help='print tensor size', default=false},
                     {arg='backw', type='boolean', help='activates probe for backward()', default=false})
end

function Probe:forward(input)
   self.output = input
   if self.size or self.content then
      print('')
      print('<probe::' .. self.name .. '> forward()')
      if self.content then print(input)
      elseif self.size then print(#input)
      end
   end
   if self.display then
      self.winf = image.display{image=input, win=self.winf}
   end
   return self.output
end

function Probe:backward(input, gradOutput)
   self.gradInput = gradOutput
   if self.backw then
      if self.size or self.content then
         print('')
         print('<probe::' .. self.name .. '> backward()')
         if self.content then print(gradOutput)
         elseif self.size then print(#gradOutput)
         end
      end
      if self.display then
         self.winb = image.display{image=gradOutput, win=self.winb}
      end
   end
   return self.gradInput
end

function Probe:write(file)
   parent.write(self, file)
   file:writeObject(self.name)
   file:writeBool(self.content)
   file:writeBool(self.display)
   file:writeBool(self.size)
   file:writeBool(self.backw)
end

function Probe:read(file)
   parent.read(self, file)
   self.name = file:readObject()
   self.content = file:readBool()
   self.display = file:readBool()
   self.size = file:readBool()
   self.backw = file:readBool()
end
