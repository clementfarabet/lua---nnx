# nnx: experimental 'nn' components

The original neural network from Torch7, [nn](https://github.com/torch/nn), contains stable and widely
used modules. 'nnx' contains more experimental, unproven modules, and
optimizations. Modules that become stable and which are proven useful make 
their way into 'nn' (some already have).

## Library Documentation ##
This section includes documentation for the following objects:

  * [SoftMaxTree](#nnx.SoftMaxTree) : a hierarchical log-softmax Module;
  * [TreeNLLCriterion](#nnx.TreeNLLCriterion) : a negative log-likelihood Criterion for the SoftMaxTree;
  * [CTCCriterion](#nnx.CTCCriterion) : a Connectionist Temporal Classification Criterion based on [warp-ctc](https://github.com/baidu-research/warp-ctc);
  * [PushTable (and PullTable)](#nnx.PushTable) : extracts a table element and inserts it later in the network;
  * [MultiSoftMax](#nnx.MultiSoftMax) : performs a softmax over the last dimension of a 2D or 3D input;
  * [SpatialReSampling](#nnx.SpatialReSampling) : performs bilinear resampling of a 3D or 4D input image;
  * [QDRiemaNNLinear] (#nnx.QDRiemaNNLinear) : quasi-diagonal reduction for Riemannian gradient descent
  * [Recurrent](#nnx.Recurrent) : a generalized recurrent neural network container;
  
<a name='nnx.SoftMaxTree'/>
### SoftMaxTree ###
A hierarchy of parameterized log-softmaxes. Used for computing the likelihood of a leaf class. 
This Module should be used in conjunction with the [TreeNLLCriterion](#nnx.TreeNLLCriterion). 
Using this for large vocabularies (100,000 and more) greatly accelerates training and evaluation 
of neural network language models (NNLM). 
A vocabulary hierarchy is provided via the [dp](https://github.com/nicholas-leonard/dp/blob/master/README.md) package's
[BillionWords](https://github.com/nicholas-leonard/dp/blob/master/doc/data.md#dp.BillionWords) 
[DataSource](https://github.com/nicholas-leonard/dp/blob/master/doc/data.md#dp.DataSource).

The constructor takes 2 mandatory and 4 optional arguments : 
 * `inputSize` : the number of units in the input embedding representation;
 * `hierarchy` : a Tensor mapping one `parent_id` to many `child_id` (a tree);
 * `rootId` : a number identifying the root node in the hierarchy. Defaults to `-1`;
 * `accUpdate` : when the intent is to use `backwardUpdate` or `accUpdateGradParameters`, set this to true to save memory. Defaults to false;
 * `static` : when true (the defualt), returns parameters with keys that don't change from batch to batch;
 * `verbose` : prints some additional information concerning the hierarchy during construction.

The `forward` method returns an `output` Tensor of size 1D, while 
`backward` returns a table `{gradInput, gradTarget}`. The second 
variable is just a Tensor of zeros , such that the `targets` can be 
propagated through [Containers](https://github.com/torch/nn/blob/master/doc/containers.md#nn.Containers) 
like [ParallelTable](https://github.com/torch/nn/blob/master/doc/table.md#nn.ParallelTable).

```lua
> input = torch.randn(5,10)
> target = torch.IntTensor{20,24,27,10,12}
> gradOutput = torch.randn(5)
> root_id = 29
> input_size = 10	
> hierarchy = {
>>    [29]=torch.IntTensor{30,1,2}, [1]=torch.IntTensor{3,4,5}, 
>>    [2]=torch.IntTensor{6,7,8}, [3]=torch.IntTensor{9,10,11},
>>    [4]=torch.IntTensor{12,13,14}, [5]=torch.IntTensor{15,16,17},
>>    [6]=torch.IntTensor{18,19,20}, [7]=torch.IntTensor{21,22,23},
>>    [8]=torch.IntTensor{24,25,26,27,28}
>> }
> smt = nn.SoftMaxTree(input_size, hierarchy, root_id)
> smt:forward{input, target}
-3.5186
-3.8950
-3.7433
-3.3071
-3.0522
[torch.DoubleTensor of dimension 5]
> smt:backward({input, target}, gradOutput)
{
  1 : DoubleTensor - size: 5x10
  2 : IntTensor - size: 5
}

```

<a name='nnx.TreeNLLCriterion'/>
### TreeNLLCriterion ###
Measures the Negative log-likelihood (NLL) for [SoftMaxTrees](#nnx.SoftMaxTree). 
Used for maximizing the likelihood of SoftMaxTree outputs.
The SoftMaxTree Module outputs a column Tensor representing the log likelihood
of each target in the batch. Thus SoftMaxTree requires the targets.
So this Criterion only computes the negative of those outputs, as 
well as its corresponding gradients.

<a name='nnx.PullTable'/>
<a name='nnx.PushTable'/>
### PushTable (and PullTable) ###
PushTable and PullTable work together. The first can be put earlier
in a digraph of Modules such that it can communicate with a 
PullTable located later in the graph. `PushTable:forward(input)` 
for an `input` table of Tensors to the output, excluding one, the index of which 
is specified by the `index` argument in the `PushTable(index)` constructor.
The Tensor identified by this `index` is communicated to one or many 
PullTables created via the `PushTable:pull(index)` factory method. 
These can be inserted later in the digraph such that 
a call to `PushTable:forward(input)`, where `input` is a table or a Tensor, 
will output a table with the previously *pushed* Tensor inserted 
at index `index`.

An example utilizing the above [SoftMaxTree](#nnx.SoftMaxTree) Module
and a Linear Module demonstrates how the PushTable can be used to 
forward the `target` Tensor without any other 
[Table Modules](https://github.com/torch/nn/blob/master/doc/table.md#table-layers):
```lua
> mlp = nn.Sequential()
> linear = nn.Linear(50,100)
> push = nn.PushTable(2)
> pull = push:pull(2)
> mlp:add(push)
> mlp:add(nn.SelectTable(1))
> mlp:add(linear)
> mlp:add(pull)
> mlp:add(smt) --smt is a SoftMaxTree instance
> mlp:forward{input, target} -- input and target are defined above
-3.5186
-3.8950
-3.7433
-3.3071
-3.0522
[torch.DoubleTensor of dimension 5]
> mlp:backward({input, target}, gradOutput) -- so is gradOutput
{
  1 : DoubleTensor - size: 5x10
  2 : IntTensor - size: 5
}
```
The above code is equivalent to the following:
```lua
> mlp2 = nn.Sequential()
> para = nn.ParallelTable()
> para:add(linear)
> para:add(nn.Identity())
> mlp2:add(para)
> mlp2:add(smt)
> mlp2:forward{input, target}
-3.5186
-3.8950
-3.7433
-3.3071
-3.0522
[torch.DoubleTensor of dimension 5]
> mlp2:backward({input, target}, gradOutput)
{
  1 : DoubleTensor - size: 5x10
  2 : IntTensor - size: 5
}
```
In some cases, this can simplify the digraph of Modules. Note that 
a PushTable can be associated to many PullTables, but each PullTable 
is associated to only one PushTable.

<a name='nnx.CTCCriterion'/>
### CTCCriterion ###
```
criterion = nn.CTCCriterion()
```
Creates a Criterion based on Baidus' [warp-ctc](https://github.com/baidu-research/warp-ctc) implementation.
This Module measures the loss between a 3D output of (batch x time x inputdim) and a target without needing alignment of inputs and labels.
Must have installed warp-ctc which can be installed via luarocks:
```
luarocks install http://raw.githubusercontent.com/baidu-research/warp-ctc/master/torch_binding/rocks/warp-ctc-scm-1.rockspec
```
Supports cuda via:
```
criterion = nn.CTCCriterion():cuda()
```
Example:
```
output = torch.Tensor({{{1,2,3,4,5},{6,7,8,9,10}}}) -- Tensor of size 1x1x5 (batch x time x inputdim).
label = {{1,3}}
ctcCriterion = nn.CTCCriterion()

print(ctcCriterion:forward(output,label))

ctcCriterion = ctcCriterion:cuda() -- Switch to cuda implementation.
output = output:cuda()

print(ctcCriterion:forward(output,label))
```

gives the output:
```
4.9038286209106
4.9038290977478
```
<a name='nnx.MultiSoftMax'/>
### MultiSoftMax ###
This Module takes 2D or 3D input and performs a softmax over the last dimension. 
It uses the existing [SoftMax](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.SoftMax) 
CUDA/C code to do so such that the Module can be used on both GPU and CPU. 
This can be useful for [keypoint detection](https://github.com/nicholas-leonard/dp/blob/master/doc/facialkeypointstutorial.md#multisoftmax).

<a name='nnx.SpatialReSampling'/>
### SpatialReSampling ###
Applies a 2D re-sampling over an input image composed of
several input planes (or channels, colors). The input tensor in `forward(input)` is 
expected to be a 3D or 4D tensor of size : `[batchSize x] nInputPlane x width x height`. 
The number of output planes will be the same as the number of input
planes.

The re-sampling is done using [bilinear interpolation](http://en.wikipedia.org/wiki/Bilinear_interpolation). 
For a simple nearest-neihbor upsampling, use `nn.SpatialUpSampling()`,
and for a simple average-based down-sampling, use 
`nn.SpatialDownSampling()`.

If the input image is a 3D tensor of size `nInputPlane x height x width`,
the output image size will be `nInputPlane x oheight x owidth` where
`owidth` and `oheight` are given to the constructor.

Instead of `owidth` and `oheight`, one can provide `rwidth` and `rheight`, 
such that `owidth = iwidth*rwidth` and `oheight = iheight*rheight`.

As an example, we can run the following code on the famous Lenna image:
```lua
require 'image'                                                           
require 'nnx'
input = image.loadPNG('doc/image/Lenna.png')
l = nn.SpatialReSampling{owidth=150,oheight=150}
output = l:forward(input)
image.save('doc/image/Lenna-150x150-bilinear.png', output)
```

The input:

![Lenna](doc/image/Lenna.png) 

The re-sampled output:

![Lenna re-sampled](doc/image/Lenna-150x150-bilinear.png) 

<a name='nnx.QDRiemaNNLinear'/>
### QDRiemaNNLinear ###
The Quasi-Diagonal Riemannian Neural Network Linear (QDRiemaNNLinear) module is an implementation
of the quasi-diagonal reduction of metrics, used for Riemannian gradient descent.
The algorithm is defined in Riemannian metrics for neural networks I: feedforward networks by Yann Ollivier (http://arxiv.org/abs/1303.0818) and an efficient implementation is described in Practical Riemannian Neural Networks by Yann Ollivier and Gaetan Marceau-Caron (http://arxiv.org/abs/1602.08007).
To use this module, simply replace `nn.Linear(ninput,noutput)` with `nnx.QDRiemaNNLinear(ninput,noutput)`.
As always, the step-size must be chosen accordingly.
Two additional arguments are also possible:
* gamma (default=0.01): determine the update rate of the metric for a minibatch setting, i.e., (1-gamma) * oldMetric + gamma newMetric. Smaller minibatches require a smaller gamma. A default value depending on the size of the minibatches is `gamma = 1. - torch.pow(1.-1./nTraining,miniBatchSize)` where `nTraining` is the number of training examples of the dataset and `miniBatchSize` is the number of training examples per minibatch. 
* qdFlag (default=true): Whether to use the quasi-diagonal reduction (true) or only the diagonal (false). The former should be better.

This module is a straightforward implementation of the outer product gradient descent.

## Requirements

* Torch7 (www.torch.ch)

## Installation

* Install Torch7 (refer to its own documentation).
* clone this project into dev directory of Torch7.
* Rebuild torch, it will include new projects too.

## Use the library

First run torch, and load nnx:

``` sh
$ torch
``` 

``` lua
> require 'nnx'
```

Once loaded, tab-completion will help you navigate through the
library (note that most function are added directly to nn):

``` lua
> nnx. + TAB
...
> nn. + TAB
```

In particular, it's good to verify that all modules provided pass their
tests:

``` lua
> nnx.test_all()
> nnx.test_omp()
```

<a name='nnx.Recurrent'/>
### Recurrent ###

DEPRECATED July 6th, 2015. Use [rnn](https://github.com/Element-Research/rnn) instead.
