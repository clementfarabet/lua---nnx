
local Logger = torch.class('nn.Logger')

function Logger:__init(filename)
   if filename then
      self.file = io.open(filename,'w')
   else
      self.file = io.stdout
      print('<Logger> warning: no file name provided, logging to std out') 
   end
   self.empty = true
   self.symbols = {}
   self.figures = {}
end

function Logger:add(symbols)
   -- (1) first time ? print symbols' names on first row
   if self.empty then
      self.empty = false
      self.nsymbols = #symbols
      for k,val in pairs(symbols) do
         self.file:write(k .. '\t')
         self.symbols[k] = {}
      end
      self.file:write('\n')
   end
   -- (2) print all symbols on one row
   for k,val in pairs(symbols) do
      if type(val) == 'number' then
         self.file:write(string.format('%11.4e',val) .. '\t')
      elseif type(val) == 'string' then
         self.file:write(val .. '\t')
      else
         xlua.error('can only log numbers and strings', 'Logger')
      end
   end
   self.file:write('\n')
   self.file:flush()
   -- (3) save symbols in internal table
   for k,val in pairs(symbols) do
      table.insert(self.symbols[k], val)
   end
end

function Logger:plot(...)
   if not lab.plot then
      if not self.warned then 
         print('<Logger> warning: cannot plot with this version of Torch') 
      end
      return
   end
   for name,list in pairs(self.symbols) do
      local nelts = #list
      local plot_x = lab.range(1,nelts)
      local plot_y = torch.Tensor(nelts)
      for i = 1,nelts do
         plot_y[i] = list[i]
      end
      self.figures[name] = lab.figure(self.figures[name])
      lab.plot(name, plot_x, plot_y, '-')
      lab.title(name)
   end
end
