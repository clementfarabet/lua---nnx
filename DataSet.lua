--------------------------------------------------------------------------------
-- DataSet: a class to handle standard datasets.
--
-- Authors: Clement Farabet, Benoit Corda
--------------------------------------------------------------------------------

local lDataSet = torch.class('nn.DataSet')

function lDataSet:__init(...)
   xlua.require('image',true)
   self.nbSamples = 0
   if select('#',...) > 0 then
      self:load(...)
   end
end

function lDataSet:size()
   return self.nbSamples
end

function lDataSet:__tostring__()
   str = 'DataSet:\n'
   if self.nbSamples then
      str = str .. ' + nb samples : '..self.nbSamples
   else
      str = str .. ' + empty set...'
   end
   return str
end

function lDataSet:load(...)
   -- parse args
   local args, dataSetFolder, nbSamplesRequired, cacheFile, channels, sampleSize
      = xlua.unpack(
      {...},
      'DataSet.load', nil,
      {arg='dataSetFolder', type='string', help='path to dataset', req=true},
      {arg='nbSamplesRequired', type='number', help='number of patches to load', default='all'},
      {arg='cacheFile', type='string', help='path to file to cache files'},
      {arg='channels', type='number', help='nb of channels', default=1},
      {arg='sampleSize', type='table', help='resize all sample: {w,h}'}
   )
   self.cacheFileName = cacheFile or self.cacheFileName

   -- Clear current dataset
   self:emptySet()

   -- Then try to find if cache file exists
   -- the base name of this file can be provided by useCacheFile()
   -- and the suffixe is the nb of samples needed, 'all' if not specified
   local fileName
   local datasetLoadedFromFile = false
   if (self.cacheFileName ~= nil) then
      fileName = self.cacheFileName .. '-' .. nbSamplesRequired
      if sys.filep(fileName) then
         -- File found
         print('<DataSet> Loading samples from cached file ' .. fileName)
         f = torch.DiskFile(fileName, 'rw')
         self:read(f)
         f.close(f)
         datasetLoadedFromFile = true
      end
   end

   -- If dataset couldn't be loaded from cache, load it
   if (datasetLoadedFromFile == false) then
      self:append{dataSetFolder=dataSetFolder, channels=channels,
                  nbSamplesRequired=nbSamplesRequired,
                  sampleSize=sampleSize}
      -- if cache name given, create it now
      if (fileName ~= nil) then
         print('<DataSet> Dumping dataset to cache file ' .. fileName .. ' for fast retrieval')
         f = torch.DiskFile(fileName, 'rw')
         self:write(f)
         f.close(f)
      end
   end
end

function lDataSet:emptySet(dataSetFolder)
   for i = 1,table.getn(self) do
      self[i] = nil
   end
   self.nbSamples = 0
end

function lDataSet:apply(toapply)
   print('<DataSet> Applying function to dataset')
   for i=1,self.nbSamples do
      xlua.progress(i, self.nbSamples)
      self[i][1] = toapply(self[i][1])
   end
end

function lDataSet:cropAndResize(side)
   for i=1,self.nbSamples do
      local newSample = torch.Tensor(1, side, side)
      local initSide = math.min(self[i][1]:size()[1], self[i][1]:size()[2])
      local x1 = math.floor((self[i][1]:size(3) - initSide) / 2)
      local y1 = math.floor((self[i][1]:size(2) - initSide) / 2)
      local x2 = x1 + initSide
      local y2 = y1 + initSide
      image.crop(newSample,self[i][1],x1,y1,x2,y2)
      self[i][1] = newSample
   end
end

function lDataSet:add(args)
   local input = args.input
   local output = args.output
   self.nbSamples = self.nbSamples + 1
   self[self.nbSamples] = {input, output}
end

function lDataSet:append(...)
   -- parse args
   local args, dataSetFolder, channels, nbSamplesRequired, useLabelPiped,
   useDirAsLabel, nbLabels, sampleSize
      = xlua.unpack(
      {...},
      'DataSet:append', 'append a folder to the dataset object',
      {arg='dataSetFolder', type='string', help='path to dataset', req=true},
      {arg='channels', type='number', help='number of channels for the image to load', default=3},
      {arg='nbSamplesRequired', type='number', help='max number of samples to load'},
      {arg='useLabelPiped', type='boolean', help='flag to use the filename as output value',default=false},
      {arg='useDirAsLabel', type='boolean', help='flag to use the directory as label',default=false},
      {arg='nbLabels', type='number', help='how many classes (goes with useDirAsLabel)', default=1},
      {arg='sampleSize', type='table', help='resize all sample: {w,h}'}
   )
   -- parse args
   local files = sys.dir(dataSetFolder)

   print('<DataSet> Loading samples from ' .. args.dataSetFolder .. '/')

   -- nb of samples to load:
   local toLoad = table.getn(files)
   if (nbSamplesRequired ~= nil and nbSamplesRequired ~= 'all') then
      toLoad = math.min(toLoad, nbSamplesRequired)
   end
   local loaded = 0

   for k,file in pairs(files) do
      local input, inputs, rawOutput

      -- disp progress
      xlua.progress(k, toLoad)

      if (string.find(file,'.png')) then
         -- load the PNG into a new Tensor
         pathToPng = sys.concat(dataSetFolder, file)
         input = image.loadPNG(pathToPng,channels)

         -- parse the file name and set the ouput from it
         rawOutput = sys.split(string.gsub(file, ".png", ""),'|')

      elseif (string.find(file,'.p[pgn]m')) then
         -- load the PPM into a new Tensor
         pathToPpm = sys.concat(dataSetFolder, file)
         input = image.loadPPM(pathToPpm,channels)

         -- parse the file name and set the ouput from it
         rawOutput = sys.split(string.gsub(file, ".p[pgn]m", ""),'|')

      elseif (string.find(file,'.jpg')) then
         -- load the JPG into a new Tensor
         pathToPpm = sys.concat(dataSetFolder, file)
         input = image.load(pathToPpm,channels)

         -- parse the file name and set the ouput from it
         rawOutput = sys.split(string.gsub(file, ".jpg", ""),'|')
      end

      -- if image loaded then add into the set
      if (input and rawOutput) then
         table.remove(rawOutput,1) --remove file ID

         -- put input in 3D tensor
         input:resize(channels, input:size(2), input:size(3))

         -- rescale ?
         if sampleSize then
            inputs = torch.Tensor(channels, sampleSize[2], sampleSize[3])
            image.scale(input, inputs, 'bilinear')
         else
            inputs = input
         end

         -- and generate output
         local output = torch.Tensor(table.getn(rawOutput), 1)
         for i,v in ipairs(rawOutput) do
            output[i][1]=v
         end

         -- add input/output in the set
         self.nbSamples = self.nbSamples + 1
         self[self.nbSamples] = {inputs, output}

         loaded = loaded + 1
         if (loaded == toLoad) then
            break
         end
      end

      -- some cleanup, for memory
      collectgarbage()
   end
end

function lDataSet:appendDataSet(dataset)
   print("<DataSet> Merging dataset of size = "..dataset:size()..
      " into dataset of size = "..self:size())
   for i = 1,dataset:size() do
      self.nbSamples = self.nbSamples + 1
      self[self.nbSamples] = {}
      self[self.nbSamples][1] = torch.Tensor(dataset[i][1]):copy(dataset[i][1])
      if (dataset[i][2] ~= nil) then
         self[self.nbSamples][2] = torch.Tensor(dataset[i][2]):copy(dataset[i][2])
      end
   end
end

function lDataSet:popSubset(args)
   -- parse args
   local nElement = args.nElement
   local ratio = args.ratio or 0.1
   local subset = args.outputSet or nn.DataSet()

   -- get nb of samples to pop
   local start_index
   if (nElement ~= nil) then
      start_index = self:size() - nElement + 1
   else
      start_index = math.floor((1-ratio)*self:size()) + 1
   end

   -- info
   print('<DataSet> Popping ' .. self:size() - start_index + 1 .. ' samples dataset')

   -- extract samples
   for i = self:size(), start_index, -1 do
      subset.nbSamples = subset.nbSamples + 1
      subset[subset.nbSamples] = {}
      subset[subset.nbSamples][1] = torch.Tensor(self[i][1]):copy(self[i][1])
      subset[subset.nbSamples][2] = torch.Tensor(self[i][2]):copy(self[i][2])
      self[i] = nil
      self.nbSamples = self.nbSamples - 1
   end

   -- return network
   return subset
end

function lDataSet:resize(w,h)
   self.resized = true
   xlua.error('not implemented yet', 'DataSet')
end

function lDataSet:shuffle()
   if (self.nbSamples == 0) then
      print('Warning, trying to shuffle empty Dataset, no effect...')
      return
   end
   local n = self.nbSamples

   while n > 2 do
      local k = math.random(n)
      -- swap elements
      self[n], self[k] = self[k], self[n]
      n = n - 1
   end
end

function lDataSet:display(args) -- opt args : scale, nbSamples
   -- arg list:
   local min, max, nbSamples, scale, w
   local title = 'DataSet'
   local resX = 800
   local resY = 600
   -- parse args:
   args = args or {}
   min = args.min
   max = args.max
   nbSamples = args.nbSamples or self.nbSamples
   nbSamples = math.min(nbSamples,self.nbSamples)
   scale = args.scale
   title = args.title or title
   w = window or gfx.Window(resX, resY, title)
   resX = args.resX or resX
   resY = args.resY or resY
   print('<DataSet> displaying ' .. nbSamples .. ' samples')

   local step_x = 0
   local step_y = 0
   self.window = w

   if (scale == nil) then
      --get the best scale to feet all data
      local sizeX = self[1][1]:size()[1]
      local sizeY = self[1][1]:size()[2]
      scale = math.sqrt(resX*resY/ (sizeX*sizeY*nbSamples))
   end

   for i=1,nbSamples do
      if (step_x >= resX) then
         step_x = 0
         step_y = step_y + self[i][1]:size()[2]*scale
         if (step_y >= resY) then
            break
         end
      end
      local tmp  = image.scaleForDisplay{tensor=self[i][1], min=min, max=max}
      w:blit(tmp, scale, step_x, step_y, title)
      step_x = step_x + self[i][1]:size()[1]*scale
   end
end

function lDataSet:useCacheFile(fileName)
   self.cacheFileName = fileName
end


function lDataSet:save(fileName)
   local fileName = fileName or self.fileName
   self.fileName = fileName
   print('<DataSet> Saving DataSet to:',fileName)
   local file = torch.DiskFile(fileName, 'w')
   self:write(file)
   file:close()
end

function lDataSet:open(fileName)
   local fileName = fileName or self.fileName
   self.fileName = fileName
   print('<DataSet> Loading DataSet from File:',fileName)
   local file = torch.DiskFile(fileName, 'r')
   self:read(file)
   file:close()
   print('<DataSet> '..self.nbSamples..' samples loaded')
end

function lDataSet:write(file)
   file:writeBool(self.resized)
   file:writeInt(self.nbSamples)
   -- write all the samples
   for i = 1,self.nbSamples do
      file:writeObject(self[i])
   end
end

function lDataSet:read(file)
   self.resized = file:readBool()
   self.nbSamples = file:readInt()
   -- read all the samples
   for i = 1,self.nbSamples do
      self[i] = file:readObject()
   end
end
