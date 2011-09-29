local CG,parent = torch.class('nn.CGOptimization', 'nn.BatchOptimization')

function CG:__init(...)
   require 'liblbfgs'
   parent.__init(self, ...)
   xlua.unpack_class(self, {...},
      'CGOptimization', nil,
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
   -- init CG state
   cg.init(self.parameters, self.gradParameters,
           self.maxEvaluation, self.maxIterations, self.maxLineSearch,
           self.sparsity, self.verbose)
end

function CG:optimize()
   -- callback for lBFGS
   lbfgs.evaluate = self.evaluate
   -- the magic function: will update the parameter vector according to the l-BFGS method
   self.output = cg.run()
end
