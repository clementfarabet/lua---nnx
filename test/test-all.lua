
function nnx.test_all()

   require 'lunit'

   nnx._test_all_ = nil
   module("nnx._test_all_", lunit.testcase, package.seeall)

   math.randomseed(os.time())

   local precision = 1e-5

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

   function test_Power()
      local in1 = lab.rand(10,20)
      local mod = nn.Power(2)
      local out = mod:forward(in1)
      local err = out:dist(in1:cmul(in1))
      assert_equal(0, err, torch.typename(mod) .. ' - forward error: ' .. err)

      local ini = math.random(5,10)
      local inj = math.random(5,10)
      local ink = math.random(5,10)
      local pw = random.uniform()*math.random(1,10)
      local input = torch.Tensor(ink, inj, ini):zero()

      local module = nn.Power(pw)

      local err = jac.test_jac(module, input, 0.1, 2)
      assert_equal((err < precision), true, 'error on state: ' .. err)

      local ferr, berr = jac.test_io(module,input)
      assert_equal(0, ferr, torch.typename(module) .. ' - i/o forward err: ' .. ferr)
      assert_equal(0, berr, torch.typename(module) .. ' - i/o backward err: ' .. berr)
   end

   function test_Square()
      local in1 = lab.rand(10,20)
      local mod = nn.Square()
      local out = mod:forward(in1)
      local err = out:dist(in1:cmul(in1))
      assert_equal(0, err, torch.typename(mod) .. ' - forward err: ' .. err)

      local ini = math.random(5,10)
      local inj = math.random(5,10)
      local ink = math.random(5,10)
      local input = torch.Tensor(ink, inj, ini):zero()

      local module = nn.Square()

      local err = jac.test_jac(module, input)
      assert_equal((err < precision), true, 'error on state: ' .. err)

      local ferr, berr = jac.test_io(module, input)
      assert_equal(0, ferr, torch.typename(module) .. ' - i/o forward err: ' .. ferr)
      assert_equal(0, berr, torch.typename(module) .. ' - i/o backward err: ' .. berr)
   end

   function test_Sqrt()
      local in1 = lab.rand(10,20)
      local mod = nn.Sqrt()
      local out = mod:forward(in1)
      local err = out:dist(in1:sqrt())
      assert_equal(0, err, torch.typename(mod) .. ' - forward err: ' .. err)

      local ini = math.random(5,10)
      local inj = math.random(5,10)
      local ink = math.random(5,10)
      local input = torch.Tensor(ink, inj, ini):zero()

      local module = nn.Sqrt()

      local err = jac.test_jac(module, input, 0.1, 2)
      assert_equal((err < precision), true, 'error on state: ' .. err)

      local ferr, berr = jac.test_io(module, input, 0.1, 2)
      assert_equal(0, ferr, torch.typename(module) .. ' - i/o forward err: ' .. ferr)
      assert_equal(0, berr, torch.typename(module) .. ' - i/o backward err: ' .. berr)
   end

   function test_Tanh()
      local ini = math.random(5,10)
      local inj = math.random(5,10)
      local ink = math.random(5,10)
      local input = torch.Tensor(ink, inj, ini):zero()

      local module = nn.Tanh()

      local err = jac.test_jac(module, input, 0.1, 2)
      assert_equal((err < precision), true, 'error on state: ' .. err)

      local ferr, berr = jac.test_io(module, input, 0.1, 2)
      assert_equal(0, ferr, torch.typename(module) .. ' - i/o forward err: ' .. ferr)
      assert_equal(0, berr, torch.typename(module) .. ' - i/o backward err: ' .. berr)
   end

   function test_HardShrink()
      local ini = math.random(5,10)
      local inj = math.random(5,10)
      local ink = math.random(5,10)
      local input = torch.Tensor(ink, inj, ini):zero()

      local module = nn.HardShrink()

      local err = jac.test_jac(module, input, 0.1, 2)
      assert_equal((err < precision), true, 'error on state: ' .. err)

      local ferr, berr = jac.test_io(module, input, 0.1, 2)
      assert_equal(0, ferr, torch.typename(module) .. ' - i/o forward err: ' .. ferr)
      assert_equal(0, berr, torch.typename(module) .. ' - i/o backward err: ' .. berr)
   end

   function test_Sigmoid()
      local ini = math.random(5,10)
      local inj = math.random(5,10)
      local ink = math.random(5,10)
      local input = torch.Tensor(ink, inj, ini):zero()

      local module = nn.Sigmoid()

      local err = jac.test_jac(module, input, 0.1, 2)
      assert_equal((err < precision), true, 'error on state: ' .. err)

      local ferr, berr = jac.test_io(module, input, 0.1, 2)
      assert_equal(0, ferr, torch.typename(module) .. ' - i/o forward err: ' .. ferr)
      assert_equal(0, berr, torch.typename(module) .. ' - i/o backward err: ' .. berr)
   end

   function test_Threshold()
      local ini = math.random(5,10)
      local inj = math.random(5,10)
      local ink = math.random(5,10)
      local input = torch.Tensor(ink, inj, ini):zero()

      local module = nn.Threshold(random.uniform(-2,2),random.uniform(-2,2))

      local err = jac.test_jac(module, input)
      assert_equal((err < precision), true, 'error on state: ' .. err)

      local ferr, berr = jac.test_io(module, input)
      assert_equal(0, ferr, torch.typename(module) .. ' - i/o forward err: ' .. ferr)
      assert_equal(0, berr, torch.typename(module) .. ' - i/o backward err: ' .. berr)
   end

   function test_Abs()
      local ini = math.random(5,10)
      local inj = math.random(5,10)
      local ink = math.random(5,10)
      local input = torch.Tensor(ink, inj, ini):zero()

      local module = nn.Abs()

      local err = jac.test_jac(module, input)
      assert_equal((err < precision), true, 'error on state: ' .. err)

      local ferr, berr = jac.test_io(module, input)
      assert_equal(0, ferr, torch.typename(module) .. ' - i/o forward err: ' .. ferr)
      assert_equal(0, berr, torch.typename(module) .. ' - i/o backward err: ' .. berr)
   end

   function test_SpatialConvolution()
      local from = math.random(1,10)
      local to = math.random(1,10)
      local ki = math.random(1,10)
      local kj = math.random(1,10)
      local si = math.random(1,1)
      local sj = math.random(1,1)
      local ini = math.random(10,20)
      local inj = math.random(10,20)
      local module = nn.SpatialConvolution(from, to, ki, kj, si, sj)
      local input = torch.Tensor(from, inj, ini):zero()

      local err = jac.test_jac(module, input)
      assert_equal((err < precision), true, 'error on state: ' .. err)

      local err = jac.test_jac_param(module, input, module.weight, module.gradWeight)
      assert_equal((err < precision), true, 'error on weight: ' .. err)

      local err = jac.test_jac_param(module, input, module.bias, module.gradBias)
      assert_equal((err < precision), true, 'error on bias: ' .. err)

      local ferr, berr = jac.test_io(module, input)
      assert_equal(0, ferr, torch.typename(module) .. ' - i/o forward err: ' .. ferr)
      assert_equal(0, berr, torch.typename(module) .. ' - i/o backward err: ' .. berr)
   end

   function test_SpatialConvolutionTable_1()
      local from = math.random(1,10)
      local to = math.random(1,10)
      local ini = math.random(10,20)
      local inj = math.random(10,20)
      local ki = math.random(1,10)
      local kj = math.random(1,10)
      local si = math.random(1,1)
      local sj = math.random(1,1)

      local ct = nn.tables.full(from,to)
      local module = nn.SpatialConvolutionTable(ct, ki, kj, si, sj)
      local input = torch.Tensor(from, inj, ini):zero()
      module:reset()

      local err = jac.test_jac(module, input)
      assert_equal((err < precision), true, 'error on state: ' .. err)

      local err = jac.test_jac_param(module, input, module.weight, module.gradWeight)
      assert_equal((err < precision), true, 'error on weight: ' .. err)

      local err = jac.test_jac_param(module, input, module.bias, module.gradBias)
      assert_equal((err < precision), true, 'error on bias: ' .. err)

      local ferr, berr = jac.test_io(module, input)
      assert_equal(0, ferr, torch.typename(module) .. ' - i/o forward err: ' .. ferr)
      assert_equal(0, berr, torch.typename(module) .. ' - i/o backward err: ' .. berr)
   end

   function test_SpatialConvolutionTable_2()
      local from = math.random(1,10)
      local to = math.random(1,10)
      local ini = math.random(10,20)
      local inj = math.random(10,20)
      local ki = math.random(1,10)
      local kj = math.random(1,10)
      local si = math.random(1,1)
      local sj = math.random(1,1)

      local ct = nn.tables.oneToOne(from)
      local module = nn.SpatialConvolutionTable(ct, ki, kj, si, sj)
      local input = torch.Tensor(from, inj, ini):zero()
      module:reset()

      local err = jac.test_jac(module, input)
      assert_equal((err < precision), true, 'error on state: ' .. err)

      local err = jac.test_jac_param(module, input, module.weight, module.gradWeight)
      assert_equal((err < precision), true, 'error on weight: ' .. err)

      local err = jac.test_jac_param(module, input, module.bias, module.gradBias)
      assert_equal((err < precision), true, 'error on bias: ' .. err)

      local ferr, berr = jac.test_io(module, input)
      assert_equal(0, ferr, torch.typename(module) .. ' - i/o forward err: ' .. ferr)
      assert_equal(0, berr, torch.typename(module) .. ' - i/o backward err: ' .. berr)
   end

   function test_SpatialConvolutionTable_3()
      local from = math.random(2,10)
      local to = math.random(1,10)
      local ini = math.random(10,20)
      local inj = math.random(10,20)
      local ki = math.random(1,10)
      local kj = math.random(1,10)
      local si = math.random(1,1)
      local sj = math.random(1,1)

      local ct = nn.tables.random(from,to,from-1)
      local module = nn.SpatialConvolutionTable(ct, ki, kj, si, sj)
      local input = torch.Tensor(from, inj, ini):zero()
      module:reset()

      local err = jac.test_jac(module, input)
      assert_equal((err < precision), true, 'error on state: ' .. err)

      local err = jac.test_jac_param(module, input, module.weight, module.gradWeight)
      assert_equal((err < precision), true, 'error on weight: ' .. err)

      local err = jac.test_jac_param(module, input, module.bias, module.gradBias)
      assert_equal((err < precision), true, 'error on bias: ' .. err)

      local ferr, berr = jac.test_io(module, input)
      assert_equal(0, ferr, torch.typename(module) .. ' - i/o forward err: ' .. ferr)
      assert_equal(0, berr, torch.typename(module) .. ' - i/o backward err: ' .. berr)
   end

   lunit.main()
end
