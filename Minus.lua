local Minus, parent = torch.class('nn.Minus', 'nn.Module')

function Minus:forward(input)
   self.output:resizeAs(input):copy(input):mul(-1)
   return self.output
end

function Minus:backward(input, gradOutput)
   self.gradInput:resizeAs(input):copy(gradOutput):mul(-1)
   return self.gradInput
end
