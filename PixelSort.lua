local PixelSort, parent = torch.class("nn.PixelSort", "nn.Module")

-- Reverse pixel shuffle, based on the torch nn.PixelShuffle module (i'd attribute code, but not sure who wrote that)
-- Converts a [batch x channel x m x p] tensor to [batch x channel*r^2 x m/r x p/r]
-- tensor, where r is the downscaling factor.
-- Useful as an alternative to pooling & strided convolutions, as it doesn't discard information
-- if used with bottleneck convolution, you can discard half of the information, as opposed to 3/4 in pooling
-- also avoids the 'checkerboard' sampling issues found with strided convolutions.
-- @param downscaleFactor - the downscaling factor to use
function PixelSort:__init(downscaleFactor)
   parent.__init(self)
   self.downscaleFactor = downscaleFactor
   self.downscaleFactorSquared = self.downscaleFactor * self.downscaleFactor
end

-- Computes the forward pass of the layer i.e. Converts a
-- [batch x channel x m x p] tensor to [batch x channel*r^2 x m/r x p/r] tensor.
-- @param input - the input tensor to be sorted of size [b x c x m x p]
-- @return output - the sorted tensor of size [b x c*r^2 x m/r x p/r]
function PixelSort:updateOutput(input)
   self._intermediateShape = self._intermediateShape or torch.LongStorage(6)
   self._outShape = self.outShape or torch.LongStorage()
   self._shuffleOut = self._shuffleOut or input.new()

   local batched = false
   local batchSize = 1
   local inputStartIdx = 1
   local outShapeIdx = 1
   if input:nDimension() == 4 then
      batched = true
      batchSize = input:size(1)
      inputStartIdx = 2
      outShapeIdx = 2
      self._outShape:resize(4)
      self._outShape[1] = batchSize
   else
      self._outShape:resize(3)
   end

   local channels = input:size(inputStartIdx)
   local inHeight = input:size(inputStartIdx + 1)
   local inWidth = input:size(inputStartIdx + 2)

   self._intermediateShape[1] = batchSize
   self._intermediateShape[2] = channels
   self._intermediateShape[3] = inHeight / self.downscaleFactor
   self._intermediateShape[4] = self.downscaleFactor
   self._intermediateShape[5] = inWidth / self.downscaleFactor
   self._intermediateShape[6] = self.downscaleFactor

   self._outShape[outShapeIdx] = channels * self.downscaleFactorSquared
   self._outShape[outShapeIdx + 1] = inHeight / self.downscaleFactor
   self._outShape[outShapeIdx + 2] = inWidth / self.downscaleFactor

   local inputView = torch.view(input, self._intermediateShape)

   self._shuffleOut:resize(inputView:size(1), inputView:size(2), inputView:size(4),
                           inputView:size(6), inputView:size(3), inputView:size(5))
   self._shuffleOut:copy(inputView:permute(1, 2, 4, 6, 3, 5))

   self.output = torch.view(self._shuffleOut, self._outShape)

   return self.output
end

-- Computes the backward pass of the layer, given the gradient w.r.t. the output
-- this function computes the gradient w.r.t. the input.
-- @param input - the input tensor of shape [b x c x m x p]
-- @param gradOutput - the tensor with the gradients w.r.t. output of shape [b x c*r^2 x m/r x p/r]
-- @return gradInput - a tensor of the same shape as input, representing the gradient w.r.t. input.
function PixelSort:updateGradInput(input, gradOutput)
   self._intermediateShape = self._intermediateShape or torch.LongStorage(6)
   self._shuffleIn = self._shuffleIn or input.new()

   local batchSize = 1
   local inputStartIdx = 1
   if input:nDimension() == 4 then
      batchSize = input:size(1)
      inputStartIdx = 2
   end
   local channels = input:size(inputStartIdx)
   local height = input:size(inputStartIdx + 1)
   local width = input:size(inputStartIdx + 2)
   
   self._intermediateShape[1] = batchSize
   self._intermediateShape[2] = channels
   self._intermediateShape[3] = self.downscaleFactor
   self._intermediateShape[4] = self.downscaleFactor
   self._intermediateShape[5] = height /self.downscaleFactor
   self._intermediateShape[6] = width /self.downscaleFactor

   local gradOutputView = torch.view(gradOutput, self._intermediateShape)

   self._shuffleIn:resize(gradOutputView:size(1), gradOutputView:size(2), gradOutputView:size(5),
                          gradOutputView:size(4), gradOutputView:size(6), gradOutputView:size(3))
   self._shuffleIn:copy(gradOutputView:permute(1, 2, 5, 3, 6, 4))

   self.gradInput = torch.view(self._shuffleIn, input:size())

   return self.gradInput
end


function PixelSort:clearState()
   nn.utils.clear(self, {
      "_intermediateShape",
      "_outShape",
      "_shuffleIn",
      "_shuffleOut",
   })
   return parent.clearState(self)
end
