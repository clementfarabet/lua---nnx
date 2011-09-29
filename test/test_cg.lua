dofile('rosenbrock.lua')

require 'liblbfgs'
neval = 0
maxIterations = 100
maxLineSearch = 40
linesearch = 2 
sparsity = 0
verbose = 2
nparam = 8

local parameters  = torch.Tensor(nparam):fill(0.1)

output, gradParameters = rosenbrock(parameters)

function printstats ()
   print('nEval: '..neval)
   print('+ fx: '..output)
   local xstring = ""
   for i = 1,parameters:size(1) do 
      xstring = string.format("%s, %2.2f", xstring, parameters[i])
   end
   print('+  x: ['..xstring..']')
   local dxstring = ""
   for i = 1,gradParameters:size(1) do 
      dxstring = string.format("%s, %2.2f", dxstring, gradParameters[i])
   end

   print('+ dx: ['..dxstring..']')
end
print('Starting:')
printstats()
lbfgs.evaluate 
   = function()
	output, gradParameters = rosenbrock(parameters)
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
