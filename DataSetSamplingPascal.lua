

--------------------------------------------------------------------------------
-- DataSetSamplingPascal: A class to handle datasets from Pascal 
--
-- Provides options to cache (on disk) dataset, precompute
-- segmentation masks, shuffle samples, filter class frequency ...
--
-- Authors: Clement Farabet, Benoit Corda, Camille Couprie
--------------------------------------------------------------------------------

local DataSetSamplingPascal = torch.class('DataSetSamplingPascal')

local path_images = 'Images'
local path_annotations = 'Annotations'
local path_masks = 'Masks'


--$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

function DataSetSamplingPascal:__init(...)
   -- check args
   xlua.unpack_class(
      self,
      {...},
      'DataSetSamplingPascal',
      'Creates a DataSet from standard Pascal directories (Images+Annotations)',
      {arg='path', type='string', help='path to Pascal directory', req=true},
      {arg='nbClasses', type='number', help='number of classes in dataset', default=1},
      {arg='nbSegments', type='number', help='number of segment per image in dataset', default=10},
      {arg='classNames', type='table', help='list of class names', default={'no name'}},
      {arg='nbRawSamples', type='number', help='number of images'},
      {arg='rawSampleMaxSize', type='number', help='resize all images to fit in a MxM window'},
      {arg='rawSampleSize', type='table', help='resize all images precisely: {w=,h=}}'},
      {arg='rawMaskRescale',type='boolean',help='does are the N classes spread between 0->255 in the PNG and need to be rescaled',default=true},
      {arg='samplingMode', type='string', help='segment sampling method: random | equal', default='random'},
      {arg='labelType', type='string', help='type of label returned: center | pixelwise', default='center'},
      {arg='infiniteSet', type='boolean', help='if true, the set can be indexed to infinity, looping around samples', default=false},
      {arg='classToSkip', type='number', help='index of class to skip during sampling', default=1},
      {arg='ScClassToSkip', type='number', help='index of class to skip during sampling', default=2},
      {arg='preloadSamples', type='boolean', help='if true, all samples are preloaded in memory', default=false},
      {arg='cacheFile', type='string', help='path to cache file (once cached, loading is much faster)'},
      {arg='verbose', type='boolean', help='dumps information', default=false}
   )

   -- fixed parameters
   self.colorMap = image.colormap(self.nbClasses)
   self.rawdata = {}
   self.currentIndex = -1
   self.realIndex = -1

   self.ctr_segment_index = 0
   self.dummy = 0

   -- parse dir structure
   print('<DataSetSamplingPascal> loading Pascal dataset from '..self.path)
   for folder in paths.files(paths.concat(self.path,path_images)) do
      if folder ~= '.' and folder ~= '..' then
    -- allowing for less nesting in the data set preparation [MS]
    if sys.filep(paths.concat(self.path,path_images,folder)) then 
       self:getsizes('./',folder)
    else
       -- loop though nested folders
       for file in paths.files(paths.concat(self.path,path_images,folder)) do

          if file ~= '.' and file ~= '..' then
             self:getsizes(folder,file)
          end
       end
    end
      end
   end


   -- nb samples: user defined or max
   self.nbRawSamples = self.nbRawSamples or #self.rawdata
   -- extract some info (max sizes)
   self.maxY = self.rawdata[1].size[2]
   self.maxX = self.rawdata[1].size[3]
   for i = 2,self.nbRawSamples do
      if self.maxX < self.rawdata[i].size[3] then
         self.maxX = self.rawdata[i].size[3]
      end
      if self.maxY < self.rawdata[i].size[2] then
         self.maxY = self.rawdata[i].size[2]
      end
   end
   self.nbSamples = self.nbRawSamples

   -- max size ?
   local maxXY = math.max(self.maxX, self.maxY)
   if not self.rawSampleMaxSize then
      if self.rawSampleSize then
    self.rawSampleMaxSize = 
       math.max(self.rawSampleSize.w,self.rawSampleSize.h)
      else
    self.rawSampleMaxSize = maxXY
      end
   end
   if maxXY < self.rawSampleMaxSize then
      self.rawSampleMaxSize = maxXY
   end

   -- some info
   if self.verbose then
      print(self)
   end

   -- sampling mode
   if self.samplingMode == 'equal' or self.samplingMode == 'random' then
      self:parseAllMasks()
      if self.samplingMode == 'random' then
         -- get the number of usable segments
         self.nbRandomSegments = 0
         for i,v in ipairs(self.tags) do
            if i ~= self.classToSkip and i ~= self.ScClassToSkip then
              self.nbRandomSegments = self.nbRandomSegments + v.size
            end
         end
         -- create shuffle table
         self.randomLookup = torch.ByteTensor(self.nbRandomSegments)
         local idx = 1
         for i,v in ipairs(self.tags) do
            if i ~= self.classToSkip and i ~= self.ScClassToSkip and v.size > 0 then
               self.randomLookup:narrow(1,idx,v.size):fill(i)
               idx = idx + v.size
            end
         end
      end
   else
      error('ERROR <DataSetSamplingPascal> unknown sampling mode')
   end

   -- preload ?
   if self.preloadSamples then
      self:preload()
   end
end

--$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

function DataSetSamplingPascal:getsizes(folder,file)
   local filepng = file:gsub('jpg$','png')
   local filexml = file:gsub('jpg$','xml')
   local imgf = paths.concat(self.path,path_images,folder,file)
   local maskf = paths.concat(self.path,path_masks,folder,filepng)
   local annotf = paths.concat(self.path,path_annotations,folder,filexml)
   local size_c, size_y, size_x
   if file:find('.jpg$') then
      size_c, size_y, size_x = image.getJPGsize(imgf)
   elseif file:find('.png$') then
      size_c, size_y, size_x = image.getPNGsize(imgf)
   elseif file:find('.mat$') then
      if not xrequire 'mattorch' then 
    xerror('<DataSetSamplingPascal> mattorch package required to handle MAT files')
      end
      local loaded = mattorch.load(imgf)
      for _,matrix in pairs(loaded) do loaded = matrix; break end
      size_c = loaded:size(1)
      size_y = loaded:size(2)
      size_x = loaded:size(3)
      loaded = nil
      collectgarbage()
   else
      xerror('images must either be JPG, PNG or MAT files', 'DataSetSamplingPascal')
   end

   table.insert(self.rawdata, {imgfile=imgf,
                maskfile=maskf,
                annotfile=annotf,
                size={size_c, size_y, size_x}})
end

--$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

function DataSetSamplingPascal:size()
   return self.nbSamples
end

--$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

function DataSetSamplingPascal:__tostring__()
   local str = 'DataSetSamplingPascal:\n'
   str = str .. '  + path : '..self.path..'\n'
   if self.cacheFile then
      str = str .. '  + cache files : [path]/'..self.cacheFile..'-[tags|samples]\n'
   end
   str = str .. '  + nb samples : '..self.nbRawSamples..'\n'
   if self.infiniteSet then
      str = str .. '  + infinite set (actual nb of samples >> set:size())\n'
   end
   if self.rawSampleMaxSize then
      str = str .. '  + samples are resized to fit in a '
      str = str .. self.rawSampleMaxSize .. 'x' .. self.rawSampleMaxSize .. ' tensor'
      str = str .. ' [max raw size = ' .. self.maxX .. 'x' .. self.maxY .. ']\n'
      if self.rawSampleSize then
         str = str .. '  + imposed ratio of ' .. self.rawSampleSize.w .. 'x' .. self.rawSampleSize.h .. '\n'
      end
   end
   if self.classToSkip ~= 0 then
      str = str .. '  + unused class : ' .. self.classNames[self.classToSkip] .. '\n'
   end
   if self.ScClassToSkip ~= 0 then
      str = str .. '  + unused class : ' .. self.classNames[self.ScClassToSkip] .. '\n'
   end

   str = str .. '  + sampling mode : ' .. self.samplingMode .. '\n'
   str = str .. '  + label type : ' .. self.labelType .. '\n'
   str = str .. '  + '..self.nbClasses..' categories : '
   for i = 1,#self.classNames-1 do
      str = str .. self.classNames[i] .. ' | '
   end
   str = str .. self.classNames[#self.classNames]
   return str
end

--$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

function DataSetSamplingPascal:__index__(key)

   -- generate sample + target at index 'key':
   if type(key)=='number' then

      -- select sample, according to samplingMode
    
      local ctr_target, tag_idx
      if self.samplingMode == 'random' then
         -- get indexes from random table
         ctr_target = self.randomLookup[math.random(1,self.nbRandomPatches)]
         tag_idx = math.floor(math.random(0,self.tags[ctr_target].size-1)/3)*3+1
      elseif self.samplingMode == 'equal' then
         -- equally sample each category:
         ctr_target = ((key-1) % (self.nbClasses)) + 1
         while self.tags[ctr_target].size == 0 or ctr_target == self.classToSkip or ctr_target == self.ScClassToSkip do
            -- no sample in that class, replacing with random patch
            ctr_target = math.floor(torch.uniform(1,self.nbClasses))
         end
         local nbSamplesPerClass = math.ceil(self.nbSamples / self.nbClasses)
         if self.infiniteSet then
            tag_idx = math.random(1,self.tags[ctr_target].size/3)
         else
            tag_idx = math.floor((key-1)/self.nbClasses) + 1
         end
         tag_idx = ((tag_idx-1) % (self.tags[ctr_target].size/3))*3 + 1
      end

      -- generate patch
      self:loadSample(self.tags[ctr_target].data[tag_idx+2])
      local sample = self.currentSample
      local mask = self.currentMask
      self.ctr_segment_index = self.tags[ctr_target].data[tag_idx]
      self.dummy = self.tags[ctr_target].data[tag_idx+1]
      -- Ce serait bien de rajouter le vecteur overlap pour ne pas le recalculer
      return {sample,mask,self.ctr_segment_index}, true
    end
   return rawget(self,key)
end

--$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

function DataSetSamplingPascal:loadSample(index)

   if self.preloadedDone then
      if index ~= self.currentIndex then
         -- load new sample
         self.currentSample = self.preloaded.samples[index]
         self.currentMask = self.preloaded.masks[index]
         -- remember index
         self.currentIndex = index
      end
   elseif index ~= self.currentIndex then

      self.realIndex = self.rawdata[index].imgfile:gsub('.jpg$','')
      -- clean up
      self.currentSample = nil
      self.currentMask = nil
      collectgarbage()
      -- matlab or regular images ?
      local matlab = false
      if self.rawdata[index].imgfile:find('.mat$') then
         if not xrequire 'mattorch' then 
            xerror('<DataSetSamplingPascal> mattorch package required to handle MAT files')
         end
         matlab = true
      end
      -- load image
      local img_loaded, mask_loaded
      if matlab then
         img_loaded = mattorch.load(self.rawdata[index].imgfile)
         mask_loaded = mattorch.load(self.rawdata[index].maskfile)
         for _,matrix in pairs(img_loaded) do
            img_loaded = matrix
            break
         end
         for _,matrix in pairs(mask_loaded) do
            mask_loaded = matrix
            break
         end
         img_loaded = img_loaded:transpose(2,3)
         mask_loaded = mask_loaded:transpose(1,2)
      else
         img_loaded = image.load(self.rawdata[index].imgfile)
         mask_loaded = image.load(self.rawdata[index].maskfile)[1]
      end
      -- resize ?
      if self.rawSampleSize then
         -- resize precisely
         local w = self.rawSampleSize.w
         local h = self.rawSampleSize.h
         self.currentSample = torch.Tensor(img_loaded:size(1),h,w)
         image.scale(img_loaded, self.currentSample, 'bilinear')
         self.currentMask = torch.Tensor(h,w)
         image.scale(mask_loaded, self.currentMask, 'simple')

      elseif self.rawSampleMaxSize and (self.rawSampleMaxSize < img_loaded:size(3)
                                     or self.rawSampleMaxSize < img_loaded:size(2)) then
         -- resize to fit in bounding box
         local w,h
         if img_loaded:size(3) >= img_loaded:size(2) then
            w = self.rawSampleMaxSize
            h = math.floor((w*img_loaded:size(2))/img_loaded:size(3))
         else
            h = self.rawSampleMaxSize
            w = math.floor((h*img_loaded:size(3))/img_loaded:size(2))
         end
         self.currentSample = torch.Tensor(img_loaded:size(1),h,w)
         image.scale(img_loaded, self.currentSample, 'bilinear')
         self.currentMask = torch.Tensor(h,w)
         image.scale(mask_loaded, self.currentMask, 'simple')
      else
         self.currentSample = img_loaded
         self.currentMask = mask_loaded
      end
      -- process mask
      if matlab then
         if self.currentMask:min() == 0 then
            self.currentMask:add(1)
         end
      elseif self.rawMaskRescale then
         -- stanford dataset style (png contains 0 and 255)
         self.currentMask:mul(self.nbClasses-1):add(0.5):floor():add(1)
      else
         -- PNG already stores values at the correct classes
         -- only holds values from 0 to nclasses
         self.currentMask:mul(255):add(1):add(0.5):floor()
      end
      self.currentIndex = index
   end
end

--$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

function DataSetSamplingPascal:preload(saveFile)
   -- if cache file exists, just retrieve images from it
   if self.cacheFile
      and paths.filep(paths.concat(self.path,self.cacheFile..'-samples')) then
      print('<DataSetSamplingPascal> retrieving saved samples from :'
            .. paths.concat(self.path,self.cacheFile..'-samples')
         .. ' [delete file to force new scan]')
      local file = torch.DiskFile(paths.concat(self.path,self.cacheFile..'-samples'), 'r')
      file:binary()
      self.preloaded = file:readObject()
      file:close()
      self.preloadedDone = true
      return
   end
   print('<DataSetSamplingPascal> preloading all images')
   self.preloaded = {samples={}, masks={}}
   for i = 1,self.nbRawSamples do
      xlua.progress(i,self.nbRawSamples)
      -- load samples, and store them in raw byte tensors (min memory footprint)
      self:loadSample(i)
      local rawTensor = torch.Tensor(self.currentSample:size()):copy(self.currentSample)
      local rawMask = torch.Tensor(self.currentMask:size()):copy(self.currentMask)
      -- insert them in our list
      table.insert(self.preloaded.samples, rawTensor)
      table.insert(self.preloaded.masks, rawMask)
   end
   self.preloadedDone = true
   -- optional cache file
   if saveFile then
      self.cacheFile = saveFile
   end
   -- if cache file given, serialize list of tags to it
   if self.cacheFile then
      print('<DataSetSamplingPascal> saving samples to cache file: '
            .. paths.concat(self.path,self.cacheFile..'-samples'))
      local file = torch.DiskFile(paths.concat(self.path,self.cacheFile..'-samples'), 'w')
      file:binary()
      file:writeObject(self.preloaded)
      file:close()
   end
end

--$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

function DataSetSamplingPascal:parseMask(existing_tags)
   local tags
   if not existing_tags then
      tags = {}
      local storage
      for i = 1,self.nbClasses do
         storage = torch.ShortStorage(self.rawSampleMaxSize*self.rawSampleMaxSize*3)
         tags[i] = {data=storage, size=0}
      end

   else
      tags = existing_tags
      -- make sure each tag list is large enough to hold the incoming data
      for i = 1,self.nbClasses do
      	    --print('current size'..tags[i].size)
         if ((tags[i].size + (self.rawSampleMaxSize*self.rawSampleMaxSize*3)) >
          tags[i].data:size()) then
            tags[i].data:resize(tags[i].size+(self.rawSampleMaxSize*self.rawSampleMaxSize*3),true)
         end
      end
   end
   -- extract labels
   local mask = self.currentMask
  
   -- (1) Compute overlap score for each segment given the image

   local file = sys.concat(self.realIndex:gsub('Images','Segments'),'.mat')
   local mat_path = file:gsub('/.mat$','.mat')
   local loaded = mattorch.load(mat_path)
   loaded = loaded.top_masks:float()
   local segment1, segmenttmp
   nb_segments = self.nbSegments
   if self.nbSegments > loaded:size(1) then nb_segments = loaded:size(1) end
   for k=1,nb_segments do

         -- (a) load one segment mask 
 	 segment1 = loaded[k]:t()	 
	
	 -- (b) resize the segment mask
	 segmenttmp = image.scale(segment1, self.rawSampleSize.w,self.rawSampleSize.h)

	 -- (c) compute overlap
	 local overlap = imgraph.overlap(segmenttmp, mask,#classes_pascal)

   	 -- (2) If overlap score ok, add the segment index to tags
      	 for i = 1,self.nbClasses do
      	     if overlap[i]>0.5 then
       	     	tags[i].data[tags[i].size+1] = k
	     	tags[i].data[tags[i].size+2] = 0 -- useless 
		tags[i].data[tags[i].size+3] = self.currentIndex
		tags[i].size = tags[i].size+3
	       -- print('insert '..k..'; '..tags[i].size )
    	     end
      	 end



   end
   return tags
end

--$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

function DataSetSamplingPascal:parseAllMasks(saveFile)
   -- if cache file exists, just retrieve tags from it
   if self.cacheFile and paths.filep(paths.concat(self.path,self.cacheFile..'-tags')) then
      print('<DataSetSamplingPascal> retrieving saved tags from :' .. paths.concat(self.path,self.cacheFile..'-tags')
         .. ' [delete file to force new scan]')
      local file = torch.DiskFile(paths.concat(self.path,self.cacheFile..'-tags'), 'r')
      file:binary()
      self.tags = file:readObject()
      file:close()
      return
   end
   -- parse tags, long operation
   print('<DataSetSamplingPascal> parsing all masks to generate list of tags')
   print('<DataSetSamplingPascal> WARNING: this operation could allocate up to '..
         math.ceil(self.nbRawSamples*self.rawSampleMaxSize*self.rawSampleMaxSize*
                   3*2/1024/1024)..'MB')
   self.tags = nil
   for i = 1,self.nbRawSamples do
      xlua.progress(i,self.nbRawSamples)
      self:loadSample(i)
      self.tags = self:parseMask(self.tags)
   end
   -- report
   print('<DataSetSamplingPascal> nb of segment extracted per category:')
   for i = 1,self.nbClasses do
      print('  ' .. i .. ' - ' .. self.tags[i].size/3)
   end
   -- optional cache file
   if saveFile then
      self.cacheFile = saveFile
   end
   -- if cache file exists, serialize list of tags to it
   if self.cacheFile then
      print('<DataSetSamplingPascal> saving tags to cache file: ' .. paths.concat(self.path,self.cacheFile..'-tags'))
      local file = torch.DiskFile(paths.concat(self.path,self.cacheFile..'-tags'), 'w')
      file:binary()
      file:writeObject(self.tags)
      file:close()
   end
end

--$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

function DataSetSamplingPascal:display(...)
   -- check args
   local _, title, samples, zoom = xlua.unpack(
      {...},
      'DataSetSamplingPascal.display',
      'display masks, overlayed on dataset images',
      {arg='title', type='string', help='window title', default='DataSetSamplingPascal'},
      {arg='samples', type='number', help='number of samples to display', default=50},
      {arg='zoom', type='number', help='zoom', default=0.5}
   )

   -- require imgraph package to handle segmentation colors
   require 'imgraph'

   -- load the samples and display them
   local allimgs = {}
   for i=1,samples do
      self:loadSample(i)
      local dispTensor = self.currentSample:clone()
      local dispMask = self.currentMask:clone()
      if dispTensor:size(1) > 3 and dispTensor:nDimension() == 3 then
         dispTensor = dispTensor:narrow(1,1,3)
      end
      dispTensor:div(dispTensor:max())
      dispMask, self.colormap = imgraph.colorize(dispMask, self.colormap)
      dispTensor:add(dispMask)
      allimgs[i] = dispTensor
   end

   -- display
   image.display{win=painter, image=allimgs, legend=title, zoom=0.5}
end








