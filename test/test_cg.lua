dofile('rosenbrock.lua')
dofile('l2.lua')

require 'liblbfgs'
neval = 0
maxIterations = 100
maxLineSearch = 40
linesearch = 0
momentum   = 0
verbose = 2
nparam = 2
local testfunc = rosenbrock

local parameters  = torch.Tensor(nparam):fill(0.1)
local gradParameters = torch.Tensor(nparam):zero()

output, gradParameters = testfunc(parameters,gradParameters)

function printstats ()
   print('nEval: '..neval)
   print('+ fx: '..output)
   local xstring = string.format("%2.2f",parameters[1])
   for i = 2,parameters:size(1) do 
      xstring = string.format("%s, %2.2f", xstring, parameters[i])
   end
   print('+  x: ['..xstring..']')
   local dxstring = string.format("%2.2f",gradParameters[1])
   for i = 2,gradParameters:size(1) do 
      dxstring = string.format("%s, %2.2f", dxstring, gradParameters[i])
   end

   print('+ dx: ['..dxstring..']')
end
print('Starting:')
printstats()
lbfgs.evaluate 
   = function()
	output, gradParameters = testfunc(parameters,gradParameters)
	neval = neval + 1
	printstats()
	return output
     end

-- init CG state
cg.init(parameters, gradParameters,
           maxEvaluation, maxIterations, maxLineSearch,
           sparsity, linesearch, verbose)

output = cg.run()

printstats()
