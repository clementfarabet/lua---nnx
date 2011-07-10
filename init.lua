----------------------------------------------------------------------
--
-- Copyright (c) 2011 Clement Farabet, Koray Kavukcuoglu
-- 
-- Permission is hereby granted, free of charge, to any person obtaining
-- a copy of this software and associated documentation files (the
-- "Software"), to deal in the Software without restriction, including
-- without limitation the rights to use, copy, modify, merge, publish,
-- distribute, sublicense, and/or sell copies of the Software, and to
-- permit persons to whom the Software is furnished to do so, subject to
-- the following conditions:
-- 
-- The above copyright notice and this permission notice shall be
-- included in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
-- EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
-- MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
-- NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
-- LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
-- OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
-- WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
-- 
----------------------------------------------------------------------
-- description:
--     xlua - lots of new trainable modules that extend the nn 
--            package.
--
-- history: 
--     July  5, 2011, 8:51PM - import from Torch5 - Clement Farabet
----------------------------------------------------------------------

require 'torch'
require 'xlua'
require 'nn'

-- create global nnx table:
nnx = {}

-- c lib:
require 'libnnx'

-- for testing:
torch.include('nnx', 'test-all.lua')
torch.include('nnx', 'test-omp.lua')

-- tools:
torch.include('nnx', 'ConfusionMatrix.lua')
torch.include('nnx', 'Logger.lua')

-- OpenMP module:
torch.include('nnx', 'OmpModule.lua')

-- pointwise modules:
torch.include('nnx', 'Abs.lua')
torch.include('nnx', 'Power.lua')
torch.include('nnx', 'Square.lua')
torch.include('nnx', 'Sqrt.lua')
torch.include('nnx', 'HardShrink.lua')
torch.include('nnx', 'Threshold.lua')

-- table-based modules:
torch.include('nnx', 'CMulTable.lua')
torch.include('nnx', 'CAddTable.lua')
torch.include('nnx', 'CDivTable.lua')
torch.include('nnx', 'CSubTable.lua')

-- reshapers:
torch.include('nnx', 'Narrow.lua')
torch.include('nnx', 'Replicate.lua')

-- spatial (images) operators:
torch.include('nnx', 'SpatialLinear.lua')
torch.include('nnx', 'SpatialLogSoftMax.lua')
torch.include('nnx', 'SpatialConvolutionSparse.lua')
torch.include('nnx', 'SpatialMaxPooling.lua')
torch.include('nnx', 'SpatialPadding.lua')
torch.include('nnx', 'SpatialNormalization.lua')
torch.include('nnx', 'SpatialUpSampling.lua')
torch.include('nnx', 'SpatialReSampling.lua')
torch.include('nnx', 'SpatialRecursiveFovea.lua')
torch.include('nnx', 'SpatialFovea.lua')
torch.include('nnx', 'SpatialGraph.lua')
torch.include('nnx', 'SpatialColorTransform.lua')

-- criterions:
torch.include('nnx', 'SuperCriterion.lua')
torch.include('nnx', 'SparseCriterion.lua')
torch.include('nnx', 'SpatialMSECriterion.lua')
torch.include('nnx', 'SpatialClassNLLCriterion.lua')
torch.include('nnx', 'SpatialSparseCriterion.lua')

-- trainers:
torch.include('nnx', 'Trainer.lua')
torch.include('nnx', 'StochasticTrainer.lua')

-- datasets:
torch.include('nnx', 'DataSet.lua')
torch.include('nnx', 'DataList.lua')
torch.include('nnx', 'DataSetLabelMe.lua')
