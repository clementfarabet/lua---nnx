
local ConfusionMatrix = torch.class('nn.ConfusionMatrix')

function ConfusionMatrix:__init(nclasses, classes)
   self.mat = lab.zeros(nclasses,nclasses)
   self.valids = lab.zeros(nclasses)
   self.totalValid = 0
   self.averageValid = 0
   self.classes = classes
end

function ConfusionMatrix:add(prediction, target)
   if type(prediction) == 'number' then
      -- comparing numbers
      self.mat[target][prediction] = self.mat[target][prediction] + 1
   else
      -- comparing vectors
      local _,prediction = lab.max(prediction)
      local _,target = lab.max(target)
      self.mat[target[1]][prediction[1]] = self.mat[target[1]][prediction[1]] + 1
   end
end

function ConfusionMatrix:zero()
   self.mat:zero()
   self.valids:zero()
   self.totalValid = 0
   self.averageValid = 0
end

function ConfusionMatrix:updateValids()
   local total = 0
   for t = 1,self.mat:size(1) do
      self.valids[t] = self.mat[t][t] / self.mat:select(1,t):sum()
      total = total + self.mat[t][t]
   end
   self.totalValid = total / self.mat:sum()
   self.averageValid = 0
   local nvalids = 0
   for t = 1,self.mat:size(1) do
      if not sys.isNaN(self.valids[t]) then
         self.averageValid = self.averageValid + self.valids[t]
         nvalids = nvalids + 1
      end
   end
   self.averageValid = self.averageValid / nvalids
end

function ConfusionMatrix:__tostring__()
   self:updateValids()
   local str = 'ConfusionMatrix:\n'
   local nclasses = self.mat:size(1)
   str = str .. '['
   for t = 1,nclasses do
      local pclass = self.valids[t] * 100
      if t == 1 then
         str = str .. '['
      else
         str = str .. ' ['
      end
      for p = 1,nclasses do
         str = str .. '' .. string.format('%8d\t', self.mat[t][p])
      end
      if self.classes then
         if t == nclasses then
            str = str .. ']]  ' .. pclass .. '% \t[class: ' .. (self.classes[t] or '') .. ']\n'
         else
            str = str .. ']   ' .. pclass .. '% \t[class: ' .. (self.classes[t] or '') .. ']\n'
         end
      else
         if t == nclasses then
            str = str .. ']]  ' .. pclass .. '% \n'
         else
            str = str .. ']   ' .. pclass .. '% \n'
         end
      end
   end
   str = str .. ' + average row correct: ' .. (self.averageValid*100) .. '% \n'
   str = str .. ' + global correct: ' .. (self.totalValid*100) .. '%'
   return str
end
