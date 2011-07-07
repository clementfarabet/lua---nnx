--------------------------------------------------------------------------------
-- DataList: a container for plain DataSets.
-- Each sub dataset represents one class.
--
-- Authors: Corda, Farabet
--------------------------------------------------------------------------------

local DataList, parent = torch.class('nn.DataList','nn.DataSet')

function DataList:__init()
   parent.__init(self)
   self.datasets = {}
   self.nbClass = 0
   self.ClassName = {}
   self.nbSamples = 0
end

function DataList:__tostring__()
   str = 'DataList'
   str = str .. ' + nb samples : '..self.nbSamples
   str = str .. ' + nb classes : '..self.nbClass
   return str
end

function DataList:__index__(key)
   if type(key)=='number' and self.nbClass>0 and key <= self.nbSamples then
      local class = ((key-1) % self.nbClass) + 1
      local classSize = self.datasets[class]:size()
      local elmt = math.floor((key-1)/self.nbClass) + 1
      elmt = ((elmt-1) % classSize) + 1

      -- create target vector on the fly
      self.datasets[class][elmt][2] = torch.Tensor(1,1,self.nbClass):fill(-1)
      self.datasets[class][elmt][2][1][1][class] = 1

      -- apply hook on sample
      local sample = self.datasets[class][elmt]
      if self.hookOnSample then
         sample = self.hookOnSample(self,sample)
      end

      return sample,true
   end
   -- if key is not a number this should return nil
   return rawget(self, key)
end

function DataList:appendDataSet(dataSet,className)
   table.insert(self.datasets,dataSet)
   if self.nbSamples == 0 then
      self.nbSamples = dataSet:size()
   else
      self.nbSamples = math.floor(math.max(self.nbSamples/self.nbClass,dataSet:size()))
   end
   self.nbClass = self.nbClass + 1
   self.nbSamples = self.nbSamples * self.nbClass
   table.insert(self.ClassName,self.nbClass,className)
end
