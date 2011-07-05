
function nnx.test_all()

   require 'lunit'
   module("test_all", lunit.testcase, package.seeall)

   local precision = 1e-6

   local jac = nnx.jacobian

   function test_SpatialLinear()
      local fanin = math.random(1,10)
      local fanout = math.random(1,10)
      local sizex = math.random(4,16)
      local sizey = math.random(4,16)
      local module = nn.SpatialLinear(fanin, fanout)
      local input = lab.rand(fanin,sizey,sizex)

      local error = jac.test_jac(module, input)
      assert_equal((error < precision), true, 'error on state: ' .. error)

      local error = jac.test_jac_param(module, input, module.weight, module.gradWeight)
      assert_equal((error < precision), true, 'error on weight: ' .. error)

      local error = jac.test_jac_param(module, input, module.bias, module.gradBias)
      assert_equal((error < precision), true, 'error on bias: ' .. error)

      local ferr, berr = jac.test_io(module, input)
      assert_equal(0, ferr, 'error in forward after i/o')
      assert_equal(0, berr, 'error in backward after i/o')
   end

   lunit.main()
end
