require 'dpnn'
require 'rnn'
require 'optim'
require 'cunn'
require 'cutorch'
torch.setheaptracking(true)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.CudaTensor')
local dataLoader = require 'dataLoad'
local grad_clip =5
local word2vec = false
local style = "random"
local threads = require 'threads'
local nthread = 4
local pool = threads.Threads(
  nthread,
  function(threadid)
    require 'cunn'
  end
)
local trainFunc = require 'trainFunc'
local batchSize = 1

local hiddenSize = 300

local dataTable = dataLoader.getData()
assert(dataTable)
local vocabToIndx = dataTable.vocabToIndx
local lines = dataTable.lines
local indxToVocab = dataTable.indxToVocab
local numVocab = #indxToVocab
local nIndex = numVocab
local vectors = dataLoader.getVectors(word2vec, style, hiddenSize,dataTable,dataTable)
for i=1,#indxToVocab do
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
local adam_params = {
  learningRate = 1e-3,
  learningRateDecay = 1e-5,
  weightDecay =1e-5,
  momentum = .95
}
local wordOptim = {
  learningRate = 1e-2,
  learningRateDecay = 1e-5,
  weightDecay =1e-5,
  momentum = .95
}

optimState = adam_params
local optimMethod = optim.adam
local dateTable = os.date("*t")
local fileName = "M2090_8Seq_ADAM.log"
local trainLogger = optim.Logger("logs/" .. dateTable.month .. "_" .. dateTable.day .. "_" .. dateTable.hour .. fileName ..".log")
--local vWrite = assert(io.open("logs/LearnedVectors_".. fileName .. ".csv","a"))
local prevError = 0
local backpropToWord= true
for i=curEpoch,maxEpoch do
  print("EPOCH: " .. tostring(i))
  local timer = torch.Timer()
  --for each epoch
  --randomly shuffle lines (paragraphs)
  torch.setdefaulttensortype('torch.FloatTensor')
  local indices = torch.randperm(#lines)
  torch.setdefaulttensortype('torch.CudaTensor')
  local curError = 0
  local seenVocab={}
  print("NUMLines = " .. tostring(indices:size(1)))
  for j=1, indices:size(1) do
    print("current line = " .. tostring(j))
    local paragraphs=  lines[indices[j]]
    pool:addjob(
      function()
        local dataTable= dataLoader.getNextSequences(batchsize,maxSeqLen, paragraphs,vectors)
        return dataTable
      end,
      function(dataTable)
        trainFunc(dataTable)
      end
    )
    cutorch.synchronize()
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
