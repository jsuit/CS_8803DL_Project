
--local vectors = require 'utils'.word2Vectors()
local pl = require 'pl.utils'


local dataLoad = torch.class('dataLoad')

function dataLoad:__init()

end
local insertAndCleanTables = function(dataTable,wordsT,targets,data)

  table.insert(dataTable.targets, targets)
  table.insert(dataTable.words, wordsT)
  table.insert(dataTable.data, data)
  --targets is a tensor
  data={}
  wordsT = {}
  collectgarbage()
end

function dataLoad:getDataAndCharMapping()
  local vT = self:getData()
  local table = require 'utils'.getCharMap()
  return {vT, table.charToIndx, table.indxToChar}
end

function dataLoad:getData()
  local vT = require 'utils'.getVocabMap()
  local indexToVocab = vT.indxToVocab

  assert(vT)
  local numWords = vT.numWords
  assert(numWords)
  print("Number of Words = " .. tostring(numWords))
  local numLines = vT.numLines
  local lines = vT.lines
  assert(lines)
  return vT
end

function dataLoad:getVectors(word2vec, style, dimension,vTable)
  print("Loading Vectors")
  assert(vTable)
  local vectors = require 'vectors'.initVectors()
  local indices = vTable.indxToVocab

  --for i=1, #indices do
--	assert(vectors[indices[i]])
--	vectors[indices[i]]= vectors[indices[i]]:cuda()
  -- end
  return vectors
end

local getSeq = function(maxSeqLen, line,vectors,vT)
  assert(maxSeqLen)
  assert(line)
  assert(vectors)
  local seqOfSequences = {}
  local words = line --pl.split(line)
  if words[1] == "" then
    table.remove(words,1)
  end
  local inputWordsTable = {}
  local tableOfWords = {}
  inputWordsTable = {}
  --loop over words
  local inputs = {}
  local targets = {}
  local targetsOfTargets = {}
  local numWords = #words
  --local targetsTensor = torch.Tensor()
  local vocabTable = vT
  assert(vT)
  local vocabToIndx = vocabTable.vocabToIndx
  local indexToVocab = vocabTable.indxToVocab
  assert(indexToVocab)
  assert(vocabToIndx)

  for i=1,numWords do
    if words[i] == "" then 

    else
      --keep adding to targets until we hit maxSeqLen, then add it to targetsOfTargets and clear out targets
      local target = torch.Tensor(1)
      if i+1 > numWords then
        target[1] = vocabToIndx["stop"]
        table.insert(targets, target:cuda()) 
      else
        assert(vocabToIndx[words[i+1]])
        target[1] = vocabToIndx[words[i+1]]
        table.insert(targets, target:cuda())
      end

      --inputs is a table with num entries == seqLen of size dimensions, where dimensions = v[word]:size(1)

      table.insert(inputs, vectors[words[i]]:cuda())
      table.insert(inputWordsTable, words[i])

      if i % maxSeqLen == 0 then
        table.insert(tableOfWords, inputWordsTable)
        table.insert(seqOfSequences,inputs)

        --local j = 0;
        --targetsTensor:apply(function() j=j+1; return targets[j] end)
        --targetsTensor:resize(#targets)
        table.insert(targetsOfTargets,targets)
        assert(#inputWordsTable == #inputs)
        assert(#inputs == #targets)
        --clear inputs
        inputs = {}
        targets ={}
        inputWordsTable ={}

        collectgarbage()
      end
    end
  end

  --sanity check
  if #targets > 0 then assert(#inputs >0) end

  if #inputs > 0 then
    assert(#targets > 0)
    --i % maxSeqLen == 0 will never be true
    table.insert(seqOfSequences,inputs)
    table.insert(targetsOfTargets,targets)
    table.insert(tableOfWords,inputWordsTable)

  end
  assert(#seqOfSequences == #targetsOfTargets)
  assert(#targetsOfTargets == #tableOfWords)
  return {seqOfSequences, targetsOfTargets,tableOfWords}
end


function dataLoad:getNextSequences(maxSeqLen,lines,vectors,vT)
  --assert(maxSeqLen)
  --assert(line)
  --assert(vectors)
  --assert(vT)
  --seqofSequenes == table of tables of 
  seqSeqTargetsTable ={}
  for i=1,#lines do
    seqSeqTargetsTable[i] = {}
    tempTable = getSeq(maxSeqLen,lines[i], vectors,vT)
    seqSeqTargetsTable[i].data=tempTable[1]
    seqSeqTargetsTable[i].targets=tempTable[2]
    seqSeqTargetsTable[i].words=tempTable[3]
  end
  --local seqSeqTargetsTable = getSeq(maxSeqLen,line, vectors,vT)
  return seqSeqTargetsTable 
  --{data = seqSeqTargetsTable[1], targets=seqSeqTargetsTable[2],words = seqSeqTargetsTable[3]}


end

function dataLoad:sample(batchsize,seqLen, lines, vectors,hiddensize,vocabToIndx)
  local seqLen = math.random(4,seqLen)
  local vocabToIndx = vocabToIndx
  local dataTable= {}
  dataTable["data"] = {}
  torch.setdefaulttensortype("torch.CudaTensor")
  dataTable["targets"] = {}
  dataTable["words"] ={}
  local words = lines[math.random(1,#lines)]
  if words[1] == "" then 
    table.remove(words,1)
  end
  while #words < seqLen do
    words = lines[math.random(1,#lines)]
  end
  local start = math.random(1,#words - seqLen + 1)
  local vTensor = torch.Tensor(hiddensize)
  local targetTensor = torch.Tensor(seqLen)
  if start+ seqLen-1 == #words then
    table.insert(words,"stop")
  end
  for i=1,seqLen do 
    local word = words[start]
    table.insert(dataTable.data, vectors[word])
    targetTensor[i] = vocabToIndx[word]
    table.insert(dataTable.targets, targetTensor[i])
    table.insert(dataTable.words, word)
    start = start+1
  end

  return dataTable
end

function dataLoad:getNextSeqOfChars(batchSize, maxSeqLen,line,vocabToIndx,charToIndx,indxToChar)
  --encInseq == inputs
  --decOutsSeq == targets
  torch.setdefaulttensortype("torch.FloatTensor")
  local minSeqLen = 2
  --local indices = torch.randperm(#lines)

  local charsPerLine ={}
  local chars ={}
  local input ={}
  local decOutput ={}
  local decInputSeq = {}
  local seqOfSeq = {}
  for j=1,#line-1 do
    local word = line[j]
    assert(word)
    assert(vocabToIndx[line[j+1]])
    assert(decOutput)
   
    table.insert(decOutput,torch.Tensor({vocabToIndx[line[j+1]]}))
    table.insert(decInputSeq,torch.Tensor({vocabToIndx[line[j]]}))
    local size = #word
    --for k=1,size do
      --table.insert(chars, torch.Tensor({charToIndx[word:sub(k,k)]}))
    --end
  end
  table.insert(decOutput,torch.Tensor({vocabToIndx["stop"]}))
  table.insert(decInputSeq,torch.Tensor({vocabToIndx[line[#line]]}))
  local word = line[#line]
  --for k=1,#word do
    --  table.insert(chars, torch.Tensor({charToIndx[word:sub(k,k)]}))
    --end
  local wTensor = nn.JoinTable(1):forward(decOutput)
  local wITensor = nn.JoinTable(1):forward(decInputSeq)
  local cTensor = nn.JoinTable(1):forward(chars)
  collectgarbage()
  return {decOutSeq=wTensor,decInSeq=wITensor}
   



end
--[[ local batches = {}
  for i=1,indices:size(1) do
    local size = chars[i]:size(1)
    if batches[size] == nil then
      batches[size] = {i}
    else
      table.insert(batches[size],i)
    end
]]


--[[local targets = torch.Tensor(batchSize,maxSeqLen):fill(0)
  local decInSeq = torch.Tensor(batchSize,maxSeqLen):fill(0)
  local t_max = -1
  local batchTensor ={}
  local seqLen ={}
  for i=1,batchSize do
    local line = lines[indices[i]]
--[[ local start
    local stop
    local ending
    if maxSeqLen < #line then
      stop = #line - maxSeqLen + 1
    else
      stop = #line
    end
    start = math.random(1,stop)

    if start + maxSeqLen - 1 <= #line then
      ending = start+maxSeqLen - 1
    else
      ending = #line
    end

    local charsT = {}
    if ending == #line then
      table.insert(line, "stop")
    end
    local count = 1
    for j=start,ending do
      local word = line[j+1]
      decInSeq[i][count] = vocabToIndx[line[j]]
--[[targets[i][count] = vocabToIndx[word]
      count = count + 1
      local chs ={}
      for k=1,#word do
        local c = word:sub(k,k)
        table.insert(chs, torch.Tensor({charToIndx[c]}))
      end
      local charTensor = nn.JoinTable(1):forward(chs)
      table.insert(charsT, charTensor)
      if seqLen[charTensor:size(1)] == nil then
        seqLen[charTensor:size(1)] = {charTensor:size(1)}
      else
        table.insert(seqLen[charTensor:size(1)], charTensor)
      end
    end

    local charTensor = nn.JoinTable(1):forward(charsT)
    t_max = math.max(t_max,charTensor:size(1)) 
    table.insert(batchTensor,charTensor)
  end

  local inputs= torch.Tensor(batchSize,t_max)
  assert(#batchTensor == batchSize)

  for i=1,batchSize do
    local count = 0
    local endingPos = batchTensor[i]:size(1)
    if batchTensor[i]:size(1) == t_max then
      inputs[i] = batchTensor[i]
    else
      inputs[i]:apply(function() count = count + 1; if count < endingPos then return batchTensor[i][count] else return 0 end end)
    end
  end
--]]


