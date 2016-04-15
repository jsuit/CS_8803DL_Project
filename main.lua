require 'dpnn'
require 'rnn'
require 'optim'
require 'cunn'
torch.setheaptracking(true)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.CudaTensor')
local dataLoader = require 'dataLoad'
local grad_clip =5
local word2vec = false
local style = "random"

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
    local dataTable= dataLoader.getNextSequences(batchsize,maxSeqLen, lines[indices[j]],vectors)
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
--	print("seqOfSeq= " .. tostring(#seqOfSeq)) 

   for k =1, #seqOfSeq do
      local eval = function(x)
        collectgarbage()
        grad_params:zero()
        local data = seqOfSeq[k]
        local output = model:forward(data)
        local err = criterion(output,seqOfTargets[k])
        model:backward(data, criterion:backward(output, seqOfTargets[k]))
        grad_params:clamp(-grad_clip, grad_clip)
        --print(err)

        return err, grad_params
      end
      
      _, E = optimMethod(eval,params, adam_params)
      if adam_params.t ~= nil and adam_params.t % 2 == 0 then
                print("Epoch " .. tostring(i) .. " iteration " .. tostring(adam_params.t))
		print("Error " .. tostring(E[1]))
      end
      curError = curError+ E[1]
      --if E[1] < 50 then
        --require 'mobdebug'.start()
      --end
      trainLogger:add{['% CE (train set)']=E[1]}
      trainLogger:style{['% CE (train set)'] = '-'}
      --trainLogger:plot()
      if backpropToWord then
	collectgarbage()
        local gradTable = model.modules[1].gradInput
        assert(wordTable)
        local words= wordTable[k]
      
        assert(#words == #gradTable)
        for i=1, #gradTable do
        --local fevalWord = function() return 0,model.modules[1].gradInput[i]:float()end  
	 local vCuda = vectors[words[i]]
	  optimMethod(function(x) return 0,gradTable[i] end,vCuda,wordOptim)	  
          --vCuda:add(-1*learningRate,model.modules[1].gradInput[i])
        
          vCuda:div(vCuda:norm())
          vectors[words[i]] = vCuda
	 --vectors[words[i]]:div(vectors[words[i]]:norm())
          if vocabToIndx[words[i]] ~= nil then
		if seenVocab[words[i]] == nil then seenVocab[words[i]] = 1
                else seenVocab[words[i]] = seenVocab[words[i]] + 1
	        end
          end
	  if #seenVocab == numVocab then
		csvigo.save({path="LearnedVectors_" .. tostring(i) .. "_" .. tostring(optimState.t) .. ".csv",data=vectors})
	 	seenVocab = {} 
		collectgarbage()
	 end
	  if #seenVocab == .5*numVocab then
		print("SEEN 1/2 of all VOCAB")
         end
	 if #seenVocab	== .25*numVocab then
		print("SEEN 1/4 of all VOCAB")
	 end 


      end

    end
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
