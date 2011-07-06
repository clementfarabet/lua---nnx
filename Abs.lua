local Abs, parent = torch.class('nn.Abs', 'nn.Module')

function Abs:__init(args)
   parent.__init(self)
   if args then
      error(xlua.usage('nn.Abs',
                          'a simple component-wise mapping: abs()',
                          'abs = nn.Abs()\n'..
                             'rectified = abs:forward(sometensor)',
                          {type='nil', help='no arg required'}))
   end
end

function Abs:forward(input)
   input.nn.Abs_forward(self, input)
   return self.output
end

function Abs:backward(input, gradOutput)
   input.nn.Abs_backward(self, input, gradOutput)
   return self.gradInput
end

function Abs:write(file)
   parent.write(self, file)
end

function Abs:read(file)
   parent.read(self, file)
end
