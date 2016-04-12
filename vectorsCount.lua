local evalFileLoc = "WordRep-1.0/Tuples_from_Wikipedia_and_Dictionary.txt"

local f = assert(io.open(evalFileLoc,'r'))

local vec = assert(io.open('vectors.txt','r'))
require 'csvigo'
local vec2 = csvigo.load({path='vectors.csv',mode='large'})
local line = vec:read("*line")
print(#vec2)

local f = function(str)
  str = str:gsub("%s+", "")
  word = nil

  if  tonumber(str:sub(1,1)) == nil then
   
    for i=1,#str do
      local c = str:sub(i,i)
      if tonumber(c) == nil then
        if word == nil then word = c 
        else
          word = word .. str:sub(i,i)
        end
      else
        break
      end 
    end
    return word
  end
end
count = 0
for i=1,#vec2 do
  local line = vec:read("*line")
  local word = f(line)
  if word ~= nil then
    print(word)
    count=count+1
    for j,v in pairs(vec2[i]) do
        
    end
  end

end

print(count)