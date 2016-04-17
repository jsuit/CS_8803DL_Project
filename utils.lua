local f = {}
local tds = require 'tds'
local pl = require 'pl.utils'
require 'paths'

function lines_from(file)
  if not paths.filep(file) then return {} end
  local i = 0
  lines = {}
  for line in io.lines(file) do 
    lines[line] = i
    i=i+1
  end
  print("num of word rep words = " .. tostring(i))
  return lines
end

f.getVocabMap = function()

  local fileName = "06-comparativeDataSet.txt"
  --local file = assert(io.open(fileName,"r"))
  --local file = torch.DiskFile(fileName,"r")
  local vocabToIndx = tds.Hash()

  local IndxToVocab = tds.Hash()
  local wordRepVocab = {}
  local index = 0
  print("Utils: Starting to Read Data")
	
  --local t = file:readString("*a")
  require 'csvigo'

  local lines = csvigo.load{path=fileName, mode='large',separator=" "}
  local numLines = #lines
  local count 
  assert(lines)
  if paths.filep('IndxToVocabB.t7') and paths.filep('vocabToIndxB.t7') then
    vocabToIndx = torch.load('vocabToIndxB.t7', 'binary')
    IndxToVocab = torch.load('IndxToVocabB.t7','binary')
    count = #IndxToVocab
    print('Loading WordRep Vocab')
    wordRepVocab = lines_from("wordRepVocab06.csv")
    assert(wordRepVocab)
   else
    for i=1,numLines do
      local wordTable = lines[i]
      for j=1,#wordTable do
        local word = wordTable[j]
        if vocabToIndx[word] == nil then
          index = index + 1
          vocabToIndx[word] = index
          assert(IndxToVocab[index] == nil)
          IndxToVocab[index] = word
        end
      end
    end
    local stop = #IndxToVocab
    IndxToVocab[stop] = "stop"
    vocabToIndx["stop"] = stop
    print("saving mappings")
    torch.save("vocabToIndxB.t7",vocabToIndx,"binary")
    print("saved 1 mapping")
    torch.save("IndxToVocabB.t7",IndxToVocab, "binary")
    count = #IndxToVocab
  end
  print(#IndxToVocab)
  print("Done reading lines")
  return {wordRepVocab = wordRepVocab, vocabToIndx = vocabToIndx,indxToVocab=IndxToVocab, numWords = count, numLines = numLines,lines=lines}
end




f.getVectors = function()

end

f.getNumLines = function()
  local fileName = "06-comparativeDataSet.txt"
  local file = assert(io.open(fileName,"r"))
  local count = 0

  while true do
    local t = file:read("*line")
    if t == nil then break end
    count = count + 1

  end
  return count
end

return f



