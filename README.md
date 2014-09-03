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
 * [TreeNLLCriterion](#nnx.TreeNLLCriterion) : a Negative log-likelihodd Criterion for the SoftMaxTree;

<a name='nnx.SoftMaxTree'/>
### SoftMaxTree ###
Computes the log of a product of softmaxes in a path.
Returns an output tensor of size 1D.
Only works with a tree (one parent per child).

<a name='nnx.TreeNLLCriterion''/>
### TreeNLLCriterion ###
Measures the Negative Log Likelihood (NLL) for SoftMaxTrees. 
Used for maximizing the likelihood of SoftMaxTree outputs.
SoftMaxTree outputs a column tensor representing the log likelihood
of each target in the batch. Thus SoftMaxTree requires the targets.
So this Criterion only computes the negative of those outputs, as 
well as its corresponding gradients.
