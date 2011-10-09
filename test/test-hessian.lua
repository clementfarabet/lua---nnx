------------------------------------------------------------
-- this simple script demonstrates the use of 
-- approximate second-order information to calibrate
-- the learning rates individually
--
-- given an input vector X, we want to learn a mapping
-- f(X) = \sum_i X_i
--

-- libs
require 'nnx'

-- SGD params
learningRate = 1e-3
diagHessianEpsilon = 1e-2

-- fake data
inputs = {}
targets = {}
for i = 1,1000 do
   inputs[i] = lab.randn(10)
   targets[i] = torch.Tensor(1):fill(inputs[i]:sum())
end

-- create module
module = nn.Sequential()
module:add(nn.Linear(10,1))

-- loss
criterion = nn.MSECriterion()

-- init diag hessian
module:initDiagHessianParameters()
diagHessianParameters = nnx.flattenParameters(nnx.getDiagHessianParameters(module))

-- estimate diag hessian over dataset
diagHessianParameters:zero()
for i = 1,#inputs do
   local output = module:forward(inputs[i])
   local critDiagHessian = criterion:backwardDiagHessian(output, targets[i])
   module:backwardDiagHessian(inputs[i], critDiagHessian)
   module:accDiagHessianParameters(inputs[i], critDiagHessian)
end
diagHessianParameters:div(#inputs)

-- protect diag hessian
diagHessianParameters:apply(function(x)
                               return math.max(x, diagHessianEpsilon)
                            end)

-- now learning rates are obtained like this:
learningRates = diagHessianParameters.new()
learningRates:resizeAs(diagHessianParameters):fill(1)
learningRates:cdiv(diagHessianParameters)

-- print info
print('learning rates calculated to')
print(learningRates)

-- regular SGD
parameters = nnx.flattenParameters(nnx.getParameters(module))
gradParameters = nnx.flattenParameters(nnx.getGradParameters(module))

for epoch = 1,10 do
   error = 0
   for i = 1,#inputs do
      -- backprop gradients
      local output = module:forward(inputs[i])
      local critGradInput = criterion:backward(output, targets[i])
      module:backward(inputs[i], critGradInput)

      -- print current error
      error = error + criterion:forward(output, targets[i])

      -- gradients wrt parameters
      gradParameters:zero()
      module:accGradParameters(inputs[i], critGradInput)

      -- given a parameter vector, and a gradParameter vector, the update goes like this:
      deltaParameters = deltaParameters or diagHessianParameters.new()
      deltaParameters:resizeAs(gradParameters):copy(learningRates):cmul(gradParameters)
      parameters:add(-learningRate, deltaParameters)
   end
   error = error / #inputs
   print('current average error: ' .. error)
end

-- test vector
input = lab.range(1,10)
grountruth = input:sum()
output = module:forward(input)
print('test input:') print(input)
print('predicted output:', output[1])
print('groundtruth (\sum_i X_i):', output[1])
