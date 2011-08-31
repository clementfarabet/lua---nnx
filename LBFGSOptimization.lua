local LBFGS,parent = torch.class('nn.LBFGSOptimization', 'nn.BatchOptimization')

function LBFGS:__init(...)
   require 'liblbfgs'
   parent.__init(self, ...)
   xlua.unpack_class(self, {...},
      'LBFGSOptimization', nil,
      {arg='maxIterations', type='number', 
       help='maximum nb of iterations per pass (0 = no max)', default=0},
      {arg='maxLineSearch', type='number', 
       help='maximum nb of steps in line search', default=20},
      {arg='sparsity', type='number', 
       help='sparsity coef (Orthantwise C)', default=0},
      {arg='parallelize', type='number', 
       help='parallelize onto N cores (experimental!)', default=1}
   )
   self.parameters = nnx.flattenParameters(nnx.getParameters(self.module))
   self.gradParameters = nnx.flattenParameters(nnx.getGradParameters(self.module))
end

function LBFGS:optimize()
   lbfgs.evaluate = self.evaluate
   -- the magic function: will update the parameter vector
   -- according to the l-BFGS method
   self.output = lbfgs.run(self.parameters, self.gradParameters,
                           self.maxIterations, self.maxLineSearch,
                           self.sparsity, self.verbose)
end
