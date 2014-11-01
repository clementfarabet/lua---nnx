local Module = nn.Module

-- returns a table of outputs and the commensurate module's gradInputs
-- this shouldn't return any duplicates
-- can't be used to set units
function Module:representations()
   local function tinsert(to, from)
      if type(from) == 'table' then
         for i=1,#from do
            tinsert(to,from[i])
         end
      else
         table.insert(to,from)
      end
   end
   local outputs = {}
   local gradInputs = {}
   if self.modules then
      for i=1,#self.modules do
         local output,gradInput = self.modules[i]:representations()
         if output then
            tinsert(outputs,output)
            tinsert(gradInputs,gradInput)
         end
      end
   else
      table.insert(outputs, self.output)
      table.insert(gradInputs, self.gradInput)
   end
   return outputs, gradInputs
end

