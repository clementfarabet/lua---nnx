
INSTALL:
$ luarocks --from=http://data.neuflow.org/lua/rocks install nnx

USE:
> require 'nnx'
> n1 = nn.SpatialLinear(16,4)

-- run tests:
> nnx.test_all()
...
> nnx.test_omp()
...
