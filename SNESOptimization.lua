local SNES,parent = torch.class('nn.SNESOptimization', 'nn.BatchOptimization')

function SNES:__init(...)
   parent.__init(self,...)
   xlua.unpack_class(self, {...},
                     'SNESOptimization', nil,
                     {arg='lambda', type='number', help='number of parallel samples', default=100},
                     {arg='eta_mu', type='number', help='learning rate for mu', default=1e-2},
                     {arg='eta_sigma', type='number', help='learning rate for sigma', default=1e-2}
                  )
   -- original parameters
   self.parameter = nnx.flattenParameters(nnx.getParameters(self.module))
   -- SNES needs one module per lambda
   self.modules = {}
   self.criterions = {}
   self.parameters = {}
   for i = 1,self.lambda do
      self.modules[i] = self.module:clone()
      self.criterions[i] = self.criterion:clone()
      self.parameters[i] = nnx.flattenParameters(nnx.getParameters(self.modules[i]))
   end
   -- SNES initial parameters
   self.mu = lab.zeros(#self.parameters[1])
   self.sigma = lab.ones(#self.parameters[1])
   -- SNES gradient vectors
   self.gradmu = torch.Tensor():resizeAs(self.mu)
   self.gradsigma = torch.Tensor():resizeAs(self.sigma)
end

function SNES:f(th, X, inputs, targets)
   -- set parameter to X
   self.parameters[th]:copy(X)
   -- estimate f on given mini batch
   local f = 0
   for i = 1,#inputs do
      local output = self.modules[th]:forward(inputs[i])
      f = f + self.criterions[th]:forward(output, targets[i])
   end
   f = f/#inputs
   return f
end

function SNES:utilities(fitness)
   -- sort fitness tables
   table.sort(fitness, function(a,b) if a.f < b.f then return a end end)
   -- compute utilities
   local sum = 0
   for i,fitness in ipairs(fitness) do
      local x = (i-1)/#fitness -- x in [0..1]
      fitness.u = math.max(0, x-0.5)
      sum = sum + fitness.u
   end
   -- normalize us
   for i,fitness in ipairs(fitness) do
      fitness.u = fitness.u / sum
   end
end

function SNES:optimize(inputs, targets)
   -- fitness for each sample drawn
   local fitness = {}

   -- draw samples
   for i = 1,self.lambda do
      -- random distribution
      local s_k = lab.randn(self.sigma:size())
      local z_k = self.mu + self.sigma*s_k

      -- evaluate fitness of f(X)
      local f_X = self:f(i, z_k, inputs, targets)

      -- store s_k, z_k
      fitness[i] = {f=f_X, s=s_k, z=z_k}
   end

   -- compute utilities
   self:utilities(fitness)

   -- set current output to best f_X (lowest)
   self.output = fitness[1].f

   -- compute gradients
   self.gradmu:zero()
   self.gradsigma:zero()
   for i = 1,self.lambda do
      local fitness = fitness[i]
      self.gradmu:add(fitness.u, fitness.s)
      self.gradsigma:add(fitness.u, fitness.s:clone():pow(2):add(-1))
   end

   -- update parameters
   for i = 1,self.lambda do
      self.mu:add( self.sigma * self.gradmu * self.eta_mu )
      self.sigma:add( (self.gradsigma * self.eta_sigma/2):exp() )
   end

   -- optimization done, copy back best parameter vector
   self.parameter:copy(fitness[1].z)
end
