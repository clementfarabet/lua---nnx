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
   self.ClassRev = {}
   self.nbSamples = 0
   self.targetIsProbability = false
   self.spatialTarget = false
end

function DataList:__tostring__()
   str = 'DataList:\n'
   str = str .. ' + nb samples : '..self.nbSamples..'\n'
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
      if self.spatialTarget then
         if self.targetIsProbability then
            self.datasets[class][elmt][2] = torch.Tensor(self.nbClass,1,1):zero()
         else
            self.datasets[class][elmt][2] = torch.Tensor(self.nbClass,1,1):fill(-1)
         end
         self.datasets[class][elmt][2][class][1][1] = 1
      else
         if self.targetIsProbability then
            self.datasets[class][elmt][2] = torch.Tensor(self.nbClass):zero()
         else
            self.datasets[class][elmt][2] = torch.Tensor(self.nbClass):fill(-1)
         end
         self.datasets[class][elmt][2][class] = 1
      end

      -- apply hook on sample
      local sample = self.datasets[class][elmt]
      if self.hookOnSample then
         sample = self.hookOnSample(self,sample)
      end

      -- auto conversion to CUDA
      if torch.getdefaulttensortype() == 'torch.CudaTensor' then
         sample[1] = torch.Tensor(sample[1]:size()):copy(sample[1])
      end

      return sample,true
   end
   -- if key is not a number this should return nil
   return rawget(self, key)
end

function DataList:appendDataSet(dataSet,className)
   table.insert(self.datasets,dataSet)
   -- if self.nbSamples == 0 then
   --    self.nbSamples = dataSet:size()
   -- else
   --    print(self.nbSamples/self.nbClass,dataSet:size())
   --    self.nbSamples = math.floor(math.max(self.nbSamples/self.nbClass,
   --                                         dataSet:size()))
   -- end
   if not self.ClassRev[className] then
      self.ClassRev[className] = true
      self.nbClass = self.nbClass + 1
      table.insert(self.ClassName,self.nbClass,className)
   end
   self.nbSamples = self.nbSamples + dataSet:size()
end
