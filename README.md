# nnx: experimental stuff for the 'nn' package.

The original neural network from Torch7, 'nn', contains stable and widely
used modules. 'nnx' contains more experimental, unproven modules, and
optimizations. Eventually, modules that become stable enough will make 
their way into 'nn' (some already have).

Disclaimer: DONT USE THIS PACKAGE WITHOUT FIRST CHECKING ITS MODULES !!!

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

## Library Documentation ##
This section includes documentation for the following objects:
 * [SoftMaxTree](#nnx.SoftMaxTree) : a hierarchical log-softmax Module;
 * [TreeNLLCriterion](#nnx.TreeNLLCriterion) : a negative log-likelihood Criterion for the SoftMaxTree;

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

Method `forward` returns an `output` Tensor of size 1D.

<a name='nnx.TreeNLLCriterion''/>
### TreeNLLCriterion ###
Measures the Negative log-likelihood (NLL) for [SoftMaxTrees](#nnx.SoftMaxTree). 
Used for maximizing the likelihood of SoftMaxTree outputs.
The SoftMaxTree Module outputs a column Tensor representing the log likelihood
of each target in the batch. Thus SoftMaxTree requires the targets.
So this Criterion only computes the negative of those outputs, as 
well as its corresponding gradients.
