# nnx: an Xperimental package for neural network modules + optimizations

The original neural network from Torch7, 'nn', contains stable and widely
used modules. 'nnx' contains more experimental, unproven modules, and
optimizations. Eventually, modules that become stable enough will make 
their way into 'nn' (some already have).

## Install dependencies 

1/ third-party libraries:

On Linux (Ubuntu > 9.04):

``` sh
$ apt-get install gcc g++ git libreadline5-dev cmake wget
```

On Mac OS (Leopard, or more), using [Homebrew](http://mxcl.github.com/homebrew/):

``` sh
$ brew install git readline cmake wget
```

2/ Lua 5.1 + Luarocks + xLua:

``` sh
$ git clone https://github.com/clementfarabet/lua4torch
$ cd lua4torch
$ make install PREFIX=/usr/local
```

3/ nnx:

Note: this automatically installs Torch7+nn, and other Lua dependencies.

``` sh
$ luarocks install nnx
```

## Use the library

First run xlua, and load nnx:

``` sh
$ xlua
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
