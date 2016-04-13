
--local vectors = require 'utils'.word2Vectors()
local pl = require 'pl.utils'

local dataLoader = {}

function dataLoader.getData()
  dataLoader.vT = require 'utils'.getVocabMap()
  dataLoader.indexToVocab = dataLoader.vT.indxToVocab

  assert(dataLoader.vT)
  local vT = dataLoader.vT
  local numWords = vT.numWords
  assert(numWords)
  print("Number of Words = " .. tostring(numWords))
  local numLines = vT.numLines
  local lines = vT.lines
  assert(lines)
  return vT
end

function dataLoader.getVectors(word2vec, style, dimension,vTable)
  print("Loading Vectors")
  local vectors = require 'vectors'.initVectors()
  local indices = vTable.indxToVocab
  --for i=1, #indices do
--	assert(vectors[indices[i]])
--	vectors[indices[i]]= vectors[indices[i]]:cuda()
 -- end
  return vectors
end

local getSeq = function(maxSeqLen, line,vectors)
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
  local vocabTable = dataLoader.vT
  local vocabToIndx = vocabTable.vocabToIndx
  local indexToVocab = dataLoader.indexToVocab
  assert(indexToVocab)
  assert(vocabToIndx)

  for i=1,numWords do
    --keep adding to targets until we hit maxSeqLen, then add it to targetsOfTargets and clear out targets
    local target = torch.Tensor(1)
    if i+1 > numWords then
	target[1] = vocabToIndx["stop"]
	table.insert(targets, target:cuda()) 
    else
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


function dataLoader.getNextSequences(batchsize,seqLen, line,vectors)
  assert(batchsize == 1)
  assert(line)

  --seqofSequenes == table of tables of 
  local seqSeqTargetsTable = getSeq(seqLen,line, vectors)
  return {data = seqSeqTargetsTable[1], targets=seqSeqTargetsTable[2],words = seqSeqTargetsTable[3]}


end

return dataLoader
