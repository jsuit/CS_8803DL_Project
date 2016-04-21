require 'dpnn'
require 'rnn'
require 'optim'
require 'cunn'
require 'cutorch'
torch.setheaptracking(true)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.CudaTensor')
require 'dataLoad'
local grad_clip =5
local word2vec = false
local style = "random"
--local threads = require 'threads'
--local nthread = 4
--[[local pool = threads.Threads(
  nthread,
  function(threadid)
    require 'cunn'
  end
)]]--
local trainFunc = require 'trainFunction'

local hiddenSize = 300
local dataLoader = dataLoad()
local dataTable = dataLoader:getData()
assert(dataTable)
local vocabToIndx = dataTable.vocabToIndx
assert(dataTable.wordRepVocab)
local wordRepVocab = dataTable.wordRepVocab
local wordRepCount = 0
for i,v in pairs(wordRepVocab) do
	wordRepCount = wordRepCount + 1
end
local lines = dataTable.lines
local indxToVocab = dataTable.indxToVocab
assert(indxToVocab)
local numVocab = #indxToVocab
local nIndex = numVocab
local vectors = dataLoader:getVectors(word2vec, style, hiddenSize,dataTable)
assert(vectors)
for i=1,#indxToVocab do
 	 assert(indxToVocab)
	 vectors[indxToVocab[i]] = vectors[indxToVocab[i]]:cuda()
end


print("defining Model")
-- define the model
local restart = false
local model
local modelFile
local optFile
if modelFile then
  model = torch.load(modelFile)
  optimState = torch.load(optFile)
else
  model = nn.Sequential()
  model:add(nn.Sequencer(nn.Linear(hiddenSize,hiddenSize)))
  local LSTM = nn.FastLSTM(hiddenSize, hiddenSize)
  LSTM.usenngraph=true
  model:add(nn.Sequencer(LSTM))
  model:add(nn.Sequencer(nn.Linear(hiddenSize, nIndex)))
  model:add(nn.Sequencer(nn.LogSoftMax()))
  criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
end
model:remember('both')


local batchsize = 1
local lineNum = 1
local maxSeqLen = 8

local maxEpoch = 20
local curEpoch = 1
collectgarbage()
params, grad_params = model:getParameters()
model:cuda()
criterion:cuda()

collectgarbage()
collectgarbage()
local optimMethod = optim.adam
local dateTable = os.date("*t")
local fileName = "M2090_8Seq_ADAM.log"
trainLogger = optim.Logger("logs/" .. dateTable.month .. "_" .. dateTable.day .. "_" .. dateTable.hour .. fileName ..".log")
--local vWrite = assert(io.open("logs/LearnedVectors_".. fileName .. ".csv","a"))
local seenVocab={}
local prevError = 0
local backpropToWord= true
local loader = dataLoader.getNextSequences
for i=curEpoch,maxEpoch do
  print("EPOCH: " .. tostring(i))
  local timer = torch.Timer()
  --for each epoch
   torch.setdefaulttensortype('torch.FloatTensor')
   local indices = torch.randperm(#lines-1)
  --randomly shuffle lines (paragraphs)
  torch.setdefaulttensortype('torch.CudaTensor')
  local curError = 0
  print("NUMLines = " ..#lines )
  for j=1, indices:size(1),2 do
    print("current line = " .. tostring(j))
    --pool:addjob( 
	--function()  
	
	local linesTwo = {lines[indices[j]],lines[indices[j+1]]}
	local dataTargetsTables= dataLoad:getNextSequences(maxSeqLen,linesTwo,vectors,dataTable)
	--local dataTargetsTable = loader(maxSeqLen,lines[j],vectors, dataTable)	--return dataTargetsTable
	--end,
	--function(dataTargetsTable)	
	assert(wordRepVocab)	
	for t=1,#dataTargetsTables do
	trainFunc(i, dataTargetsTables[t], optimMethod,model,curError,vectors,vocabToIndx,wordRepVocab,wordRepCount)	--end
--	)		
  	end
  end
  print("Epoch " .. tostring(i) .." took " .. timer.time().real .. " seconds")
  if prevError > curError or prevError == 0 then
    model:clearState()
    prevError = curError
    collectgarbage()
    dateTable = os.date("*t")
    print("SAVING MODEL")
    torch.save("models/" .. tostring(curEpoch) .. "_" .. dateTable.month .. "_" .. dateTable.day .. "_" .. dateTable.hour .. "Model"..fileName, model)
    torch.save("models/" .. tostring(curEpoch) .. "_" .. dateTable.month .. "_" .. dateTable.day .. "_" .. dateTable.hour .. "OptimState"..fileName , optimState)
    torch.save("models/" .. tostring(curEpoch) .. "_" .. dateTable.month .. "_" .. dateTable.day .. "_" .. dateTable.hour.."LearnedVectorsEpoch" .. tostring(i) .. ".txt", vectors)
    collectgarbage()
    collectgarbage()
  end

end
