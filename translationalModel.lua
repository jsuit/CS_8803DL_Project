require 'rnn'

version = 1.2 -- refactored numerical gradient test into unit tests. Added training loop
local word2vec 
local style = "uniform"
local hiddenSize =300

local opt = {}
opt.learningRate = 0.1
opt.hiddenSize = 6
opt.vocabSize = 6
opt.seqLen = 2 -- length of the encoded sequence
opt.niter = 1000

--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
local function forwardConnect(encLSTM, decLSTM)
  decLSTM.userPrevOutput = nn.rnn.recursiveCopy(decLSTM.userPrevOutput, encLSTM.outputs[opt.seqLen])
  decLSTM.userPrevCell = nn.rnn.recursiveCopy(decLSTM.userPrevCell, encLSTM.cells[opt.seqLen])
end
[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
local function backwardConnect(encLSTM, decLSTM)
	encLSTM.userNextGradCell = nn.rnn.recursiveCopy(encLSTM.userNextGradCell, decLSTM.userGradPrevCell)
	encLSTM.gradPrevOutput = nn.rnn.recursiveCopy(encLSTM.gradPrevOutput, decLSTM.userGradPrevOutput)
end
--Encoder
local enc = nn.Sequential()
enc:add(nn.LookupTable(opt.vocabSize, opt.hiddenSize))
enc:add(nn.SplitTable(1, 2)) -- works for both online and mini-batch mode
local encLSTM = nn.LSTM(opt.hiddenSize, opt.hiddenSize)
enc:add(nn.Sequencer(encLSTM))
enc:add(nn.SelectTable(-1))

--Decoder
local dec = nn.Sequential()
dec:add(nn.LookupTable(opt.vocabSize, opt.hiddenSize))
dec:add(nn.SplitTable(1, 2)) -- works for both online and mini-batch mode
local decLSTM = nn.LSTM(opt.hiddenSize, opt.hiddenSize)
dec:add(nn.Sequencer(decLSTM))
dec:add(nn.Sequencer(nn.Linear(opt.hiddenSize, opt.vocabSize)))
dec:add(nn.Sequencer(nn.LogSoftMax()))

local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

local hiddenSize = 300
local dataLoader = dataLoad()
local dataTable = dataLoader:getData

local dataLoader = dataLoad()
local dataTable = dataLoader:getDataAndCharMapping()
local charToIndx = dataTable.charToIndx
local indxToChar = dataTable.indxToChar
local vocabToIndx = dataTable.vocabToIndx
local lines = dataTable.lines
for i=curEpoch,maxEpoch do
	print("EPOCH: " .. tostring(i))
  	local timer = torch.Timer()
  	--for each epoch
   	torch.setdefaulttensortype('torch.FloatTensor')
   	local indices = torch.randperm(#lines-1)
	for j=1:indices:size(1) do
		
		local inputsTargets = getNextSeqOfChars(maxSeqLen,lines,j,vocabToIndx,charToIndx,indxToChar)
		local encInSeq	
	end
