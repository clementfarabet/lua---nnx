local SpatialColorTransform, parent = torch.class('nn.SpatialColorTransform', 'nn.Module')

local help_desc = 
[[Provides a set of widely used/known color space transforms,
for images: RGB->YUV, YUV->RGB, RGB->Y transforms, and 
more exotic transforms such as RGB->Normed-RGB]]

local help_example = 
[[-- transforms an RGB image into a YUV image:
converter = nn.SpatialColorTransform('rgb2yuv')
rgb = image.lena()
yuv = converter:forward(rgb) 
image.display(yuv) ]]

function SpatialColorTransform:__init(type)
   -- parent init
   parent.__init(self)

   -- require the image package
   xlua.require('image',true)

   -- usage
   self.usage = xlua.usage(
      'nn.SpatialColorTransform', help_desc, help_example,
      {type='string', req=true,
       help='transform = yuv2rgb | rgb2yuv | rgb2y | hsl2rgb | hsv2rgb | rgb2hsl | rgb2hsv | rgb2nrgb | rgb2y+nrgb'}
   )

   -- transform type
   self.type = type
   if type == 'yuv2rgb' then
      self.islinear = true
      self.linear = nn.SpatialLinear(3,3)
      -- R
      self.linear.weight[1][1] = 1
      self.linear.weight[1][2] = 0
      self.linear.weight[1][3] = 1.13983
      self.linear.bias[1] = 0
      -- G
      self.linear.weight[2][1] = 1
      self.linear.weight[2][2] = -0.39465
      self.linear.weight[2][3] = -0.58060
      self.linear.bias[2] = 0
      -- B
      self.linear.weight[3][1] = 1
      self.linear.weight[3][2] = 2.03211
      self.linear.weight[3][3] = 0
      self.linear.bias[3] = 0
   elseif type == 'rgb2yuv' then
      self.islinear = true
      self.linear = nn.SpatialLinear(3,3)
      -- Y
      self.linear.weight[1][1] = 0.299
      self.linear.weight[1][2] = 0.587
      self.linear.weight[1][3] = 0.114
      self.linear.bias[1] = 0
      -- U
      self.linear.weight[2][1] = -0.14713
      self.linear.weight[2][2] = -0.28886
      self.linear.weight[2][3] = 0.436
      self.linear.bias[2] = 0
      -- V
      self.linear.weight[3][1] = 0.615
      self.linear.weight[3][2] = -0.51499
      self.linear.weight[3][3] = -0.10001
      self.linear.bias[3] = 0
   elseif type == 'rgb2y' then
      self.islinear = true
      self.linear = nn.SpatialLinear(3,1)
      -- Y
      self.linear.weight[1][1] = 0.299
      self.linear.weight[1][2] = 0.587
      self.linear.weight[1][3] = 0.114
      self.linear.bias[1] = 0
   elseif type == 'hsl2rgb' then
      self.islinear = false
   elseif type == 'hsv2rgb' then
      self.islinear = false
   elseif type == 'rgb2hsl' then
      self.islinear = false
   elseif type == 'rgb2hsv' then
      self.islinear = false
   elseif type == 'rgb2nrgb' then
      self.islinear = false
   elseif type == 'rgb2y+nrgb' then
      self.islinear = false
   else
      xlua.error('transform required','nn.SpatialColorTransform',self.usage)
   end      
end

function SpatialColorTransform:forward(input)
   if self.islinear then
      self.output = self.linear:forward(input)
   else
      if self.type == 'rgb2hsl' then
         self.output = image.rgb2hsl(input, self.output)
      elseif self.type == 'rgb2hsv' then
         self.output = image.rgb2hsv(input, self.output)
      elseif self.type == 'hsl2rgb' then
         self.output = image.hsl2rgb(input, self.output)
      elseif self.type == 'rgb2hsv' then
         self.output = image.rgb2hsv(input, self.output)
      elseif self.type == 'rgb2nrgb' then
         self.output = image.rgb2nrgb(input, self.output)
      elseif self.type == 'rgb2y+nrgb' then
         self.output:resize(4, input:size(2), input:size(3))
         image.rgb2y(input, self.output:narrow(1,1,1))
         image.rgb2nrgb(input, self.output:narrow(1,2,3))
      end
   end
   return self.output
end

function SpatialColorTransform:backward(input, gradOutput)
   if self.islinear then
      self.gradInput = self.linear:backward(input, gradOutput)
   else
      xlua.error('backward not implemented for non-linear transforms',
                 'SpatialColorTransform.backward')
   end
   return self.gradInput
end

function SpatialColorTransform:write(file)
   parent.write(self, file)
   file:writeObject(self.type)
   file:writeBool(self.islinear)
   if self.islinear then
      file:writeObject(self.linear)
   end
end

function SpatialColorTransform:read(file)
   parent.read(self, file)
   self.type = file:readObject()
   self.islinear = file:readBool()
   if self.islinear then
      self.linear = file:readObject()
   end
end
