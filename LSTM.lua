------------------------------------------------------------------------
--[[ LSTM ]]--
-- Long Short Term Memory architecture.
-- Ref. A.: http://arxiv.org/pdf/1303.5778v1
-- Expects 1D or 2D input.
-- The first input in sequence uses zero value for cell and hidden state
------------------------------------------------------------------------
local LSTM, parent = torch.class('nn.LSTM', 'nn.Module')

function LSTM:__init(inputSize, outputSize, inputTransfer, forgetTransfer, cellTransfer, outputTransfer)
   parent.__init(self)
   self.inputSize = inputSize
   self.outputSize = outputSize
   self.inputTransfer = inputTransfer or nn.Sigmoid()
   self.forgetTransfer = forgetTransfer or nn.Sigmoid()
   self.cellTransfer = cellTransfer or nn.Tanh()
   self.outputTransfer = outputTransfer or nn.Tanh()
   
   self.inputGate = self:buildInputGate()
   self.forgetGate = self:buildForgetGate()
end

function LSTM:buildInputGate()
   -- Note : inputGate:forward expects an input table : {input, output, cell}
   local gate = nn.Sequential()
   local para = nn.ParallelTable()
   local inputOutput = nn.Linear(self.inputSize, self.outputSize)
   local cellOutput = nn.CMul(self.outputSize) -- diagonal cell to gate weight matrix
   local outputOutput = nn.Linear(self.outputSize, self.outputSize)
   outputOutput:noBias() --TODO
   para:add(inputOutput):add(cellOutput):add(outputOutput)
   gate:add(self.inputTransfer)
   return gate
end
