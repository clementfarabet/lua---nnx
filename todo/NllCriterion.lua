local NllCriterion, parent = torch.class('nn.NllCriterion', 'nn.Criterion')

function NllCriterion:__init()
   parent.__init(self)
end

function NllCriterion:forward(input, target)
   local linput = torch.Tensor()
   linput:set(input:storage(),1,input:nElement())
   local ltarget = torch.Tensor()
   ltarget:set(target:storage(),1,target:nElement())
   local _,idTarget = lab.max(ltarget)
   self.output = -linput[idTarget[1]]
   return self.output
end

function NllCriterion:backward(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()
   -- set the target in a 1d fashion
   local lgradInput = torch.Tensor()
   lgradInput:set(self.gradInput:storage(),1,self.gradInput:nElement())
   local ltarget = torch.Tensor()
   ltarget:set(target:storage(),1,target:nElement())
   local _,idTarget = lab.max(ltarget)
   lgradInput[idTarget[1]] = -1
   return self.gradInput
end
