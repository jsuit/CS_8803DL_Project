require 'dpnn'
require 'rnn'
require 'optim'
torch.setheaptracking(true)
torch.setdefaulttensortype('torch.FloatTensor')
local dataLoader = require 'dataLoad'
local grad_clip =5
local word2vec = false
local style = "random"

local batchSize = 1

local hiddenSize = 300

local dataTable = dataLoader.getData()
assert(dataTable)
local indxToVocab = dataTable.indxToVocab
local nIndex = #indxToVocab
local lines = dataTable.lines
local indexToVocab = dataTable.indxToVocab

local vectors = dataLoader.getVectors(word2vec, style, hiddenSize,dataTable,dataTable)
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
--model:remember('both')


local batchsize = 1
local lineNum = 1
local maxSeqLen = 32

local maxEpoch = 20
local curEpoch = 1
collectgarbage()
params, grad_params = model:getParameters()
collectgarbage()
collectgarbage()
local adam_params = {
  learningRate = 1e-3,
  learningRateDecay = 0,
  weightDecay = 0,
  momentum = 0
}
optimState = adam_params
local dateTable = os.date("*t")
local trainLogger = optim.Logger("logs/" .. dateTable.month .. "_" .. dateTable.day .. "_" .. dateTable.hour)
local prevError = 0
local backpropToWord= true
for i=curEpoch,maxEpoch do
  --for each epoch
  --randomly shuffle lines (paragraphs)
  local indices = torch.randperm(#lines)

  local curError = 0
  for j=1, indices:size(1) do

    local dataTable= dataLoader.getNextSequences(batchsize,maxSeqLen, lines[j],vectors)
    local seqOfSeq = dataTable.data
    local seqOfTargets = dataTable.targets
    local wordTable
    if backpropToWord then
      wordTable = dataTable.words
    end
    --for each line get seq of sequences. #seqOfSeq tells us how many sequences of maxSeqLen we could make
    -- from jth line.
    -- if the last seqOfSeq might be less than maxSeqLen we could make
    -- if #words in lines[j] < maxSeqLen, then #seqOfSeq =1 and seqOfSeq[1] == #words in lines[j]
    for k =1, #seqOfSeq do

      local eval = function(x)
        collectgarbage()
        grad_params:zero()
        local data = seqOfSeq[k]
        local output = model:forward(data)
        local err = criterion(output,seqOfTargets[k])
        model:backward(data, criterion:backward(output, seqOfTargets[k]))
        grad_params:clamp(-grad_clip, grad_clip)
        print(err)

        return err, grad_params
      end
      _, E = optim.adam(eval,params, adam_params)
      prevError = prevError + E[1]
      --if E[1] < 50 then
        --require 'mobdebug'.start()
      --end
      trainLogger:add{['% CE (train set)']=E[1]}
      trainLogger:style{['% CE (train set)'] = '-'}
      trainLogger:plot()
      if backpropToWord then
        local gradTable = model.modules[1].gradInput
        assert(wordTable)
        local words= wordTable[k]
        local learningRate = 1e4
      
        assert(#words == #gradTable)
        for i=1, #gradTable do
          vectors[words[i]]:add(-1*learningRate,model.modules[1].gradInput[i])
          local norm = vectors[words[i]]:norm()
          if norm > 0 then
            vectors[words[i]]:div(vectors[words[i]]:norm())
          end
        end


      end

    end
  end

  if prevError > curError or prevError == 0 then
    model:clearState()
    collectgarbage()
    dateTable = os.date("*t")
    print("SAVING MODEL")
    torch.save("models/" .. tostring(curEpoch) .. "_" .. dateTable.month .. "_" .. dateTable.day .. "_" .. dateTable.hour .. "Model", model)
    torch.save("models/" .. tostring(curEpoch) .. "_" .. dateTable.month .. "_" .. dateTable.day .. "_" .. dateTable.hour .. "OptimState" , optimState)
    torch.save("models/" .. tostring(curEpoch) .. "_" .. dateTable.month .. "_" .. dateTable.day .. "_" .. dateTable.hour.."LearnedVectors.txt", vectors)
    collectgarbage()
    collectgarbage()
  end

end