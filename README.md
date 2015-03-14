# nnx: experimental 'nn' components

The original neural network from Torch7, [nn](https://github.com/torch/nn), contains stable and widely
used modules. 'nnx' contains more experimental, unproven modules, and
optimizations. Modules that become stable and which are proven useful make 
their way into 'nn' (some already have).

## Library Documentation ##
This section includes documentation for the following objects:
 * [Recurrent](#nnx.Recurrent) : a generalized recurrent neural network container;
 * [SoftMaxTree](#nnx.SoftMaxTree) : a hierarchical log-softmax Module;
 * [TreeNLLCriterion](#nnx.TreeNLLCriterion) : a negative log-likelihood Criterion for the SoftMaxTree;
 * [PushTable (and PullTable)](#nnx.PushTable) : extracts a table element and inserts it later in the network;
 * [MultiSoftMax](#nnx.MultiSoftMax) : performs a softmax over the last dimension of a 2D or 3D input;
 * [SpatialReSampling](#nnx.SpatialReSampling) : performs bilinear resampling of a 3D or 4D input image;
 
<a name='nnx.Recurrent'/>
### Recurrent ###
References :
 * A. [Sutsekever Thesis Sec. 2.5 and 2.8](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
 * B. [Mikolov Thesis Sec. 3.2 and 3.3](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)
 * C. [RNN and Backpropagation Guide](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.3.9311&rep=rep1&type=pdf)

A [composite Module](https://github.com/torch/nn/blob/master/doc/containers.md#containers) for implementing Recurrent Neural Networks (RNN), excluding the output layer. 

The `nn.Recurrent(start, input, feedback, [transfer, rho, merge])` constructor takes 5 arguments:
 * `start` : the size of the output (excluding the batch dimension), or a Module that will be inserted between the `input` Module and `transfer` module during the first step of the propagation. When `start` is a size (a number or `torch.LongTensor`), then this *start* Module will be initialized as `nn.Add(start)` (see Ref. A).
 * `input` : a Module that processes input Tensors (or Tables). Output must be of same size as `start` (or its output in the case of a `start` Module), and same size as the output of the `feedback` Module.
 * `feedback` : a Module that feedbacks the previous output Tensor (or Tables) up to the `transfer` Module.
 * `transfer` : a non-linear Module used to process the element-wise sum of the `input` and `feedback` module outputs, or in the case of the first step, the output of the *start* Module.
 * `rho` : the maximum amount of backpropagation steps to take back in time. Limits the number of previous steps kept in memory. Due to the vanishing gradients effect, references A and B recommend `rho = 5` (or lower). Defaults to 5.
 * `merge` : a [table Module](https://github.com/torch/nn/blob/master/doc/table.md#table-layers) that merges the outputs of the `input` and `feedback` Module before being forwarded through the `transfer` Module.
 
An RNN is used to process a sequence of inputs. 
Each step in the sequence should be propagated by its own `forward` (and `backward`), 
one `input` (and `gradOutput`) at a time. 
Each call to `forward` keeps a log of the intermediate states (the `input` and many `Module.outputs`) 
and increments the `step` attribute by 1. 
A call to `backward` doesn't result in a `gradInput`. It only keeps a log of the current `gradOutput` and `scale`.
Back-Propagation Through Time (BPTT) is done when the `updateParameters` or `backwardThroughTime` method
is called. The `step` attribute is only reset to 1 when a call to the `forget` method is made. 
In which case, the Module is ready to process the next sequence (or batch thereof).
Note that the longer the sequence, the more memory will be required to store all the 
`output` and `gradInput` states (one for each time step). 

To use this module with batches, we suggest using different 
sequences of the same size within a batch and calling `updateParameters` 
every `rho` steps and `forget` at the end of the Sequence. 

Note that calling the `evaluate` method turns off long-term memory; 
the RNN will only remember the previous output. This allows the RNN 
to handle long sequences without allocating any additional memory.

Example :
```lua
require 'nnx'

batchSize = 8
rho = 5
hiddenSize = 10
nIndex = 10000
-- RNN
r = nn.Recurrent(
   hiddenSize, nn.LookupTable(nIndex, hiddenSize), 
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(), 
   rho
)

rnn = nn.Sequential()
rnn:add(r)
rnn:add(nn.Linear(hiddenSize, nIndex))
rnn:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()

-- dummy dataset (task is to predict next item, given previous)
sequence = torch.randperm(nIndex)

offsets = {}
for i=1,batchSize do
   table.insert(offsets, math.ceil(math.random()*batchSize))
end
offsets = torch.LongTensor(offsets)

lr = 0.1
updateInterval = 4
i = 1
while true do
   -- a batch of inputs
   local input = sequence:index(1, offsets)
   local output = rnn:forward(input)
   -- incement indices
   offsets:add(1)
   for j=1,batchSize do
      if offsets[j] > nIndex then
         offsets[j] = 1
      end
   end
   local target = sequence:index(1, offsets)
   local err = criterion:forward(output, target)
   local gradOutput = criterion:backward(output, target)
   -- the Recurrent layer is memorizing its gradOutputs (up to memSize)
   rnn:backward(input, gradOutput)
   
   i = i + 1
   -- note that updateInterval < rho
   if i % updateInterval == 0 then
      -- backpropagates through time (BPTT) :
      -- 1. backward through feedback and input layers,
      -- 2. updates parameters
      r:updateParameters(lr)
   end
end
```

Note that this won't work with `input` and `feedback` modules that use more than their
`output` attribute to keep track of their internal state between 
calls to `forward` and `backward`.
 
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
