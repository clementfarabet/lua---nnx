local LBFGS,parent = torch.class('nn.LBFGSOptimization', 'nn.BatchOptimization')

function LBFGS:__init(...)
   require 'liblbfgs'
   parent.__init(self, ...)
   xlua.unpack_class(self, {...},
      'LBFGSOptimization', nil,
      {arg='maxEvaluation', type='number', 
       help='maximum nb of function evaluations per pass (0 = no max)', default=0},
      {arg='maxIterations', type='number', 
       help='maximum nb of iterations per pass (0 = no max)', default=0},
      {arg='maxLineSearch', type='number', 
       help='maximum nb of steps in line search', default=20},
      {arg='sparsity', type='number', 
       help='sparsity coef (Orthantwise C)', default=0},
      {arg='parallelize', type='number', 
       help='parallelize onto N cores (experimental!)', default=1}
   )
   -- get module parameters/
   self.parameters = nnx.flattenParameters(nnx.getParameters(self.module))
   self.gradParameters = nnx.flattenParameters(nnx.getGradParameters(self.module))
   -- init LBFGS state
   lbfgs.init(self.parameters, self.gradParameters,
              self.maxEvaluation, self.maxIterations, self.maxLineSearch,
              self.sparsity, self.verbose)
end

function LBFGS:optimize()
   -- callback for lBFGS
   lbfgs.evaluate = self.evaluate
   -- the magic function: will update the parameter vector according to the l-BFGS method
   self.output = lbfgs.run()
end
