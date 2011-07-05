
require 'torch'
require 'nn'

-- create global nnx table:
nnx = {}

-- c lib:
require 'libnnx'

-- for testing:
torch.include('nnx', 'jacobian.lua')
torch.include('nnx', 'test-all.lua')

-- modules:
torch.include('nnx', 'Narrow.lua')
torch.include('nnx', 'SpatialLinear.lua')
