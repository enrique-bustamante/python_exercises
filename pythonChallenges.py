# %%
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math

# %%
# create a sum function
def sumList(list):
    total = 0
    for item in range(len(list)):
        total = total + list[item]
    return total

# %%
numberList = [1,2,3,4,21, 11.4,0]

sum = sumList(numberList)

# %%
# create a count function
def counter(list):
    itemCount = 0
    for item in range(len(list)):
        itemCount += 1
    return itemCount

# %%
itemCount = counter(numberList)
# %%
# create an average function
def average(list):
    listSum = sumList(list)
    listCount = counter(list)
    return listSum/listCount
# %%
average(numberList)
# %%
# create a sort function
def sorter(list):
    sortedList = []
    while list:
        minimum = list[0]
        for item in list:
            if item < minimum:
                minimum = item
        sortedList.append(minimum)
        list.remove(minimum)
    return sortedList
# %%
unsortList = [3, 2, 6, 4, 1]
# %%
sorted = sorter(unsortList)
# %%
# create a word frequency counter
def wordFreq(string):
    string = string.lower()
    stringNoPunct = re.sub(r'[^\w\s]','', string)
    stringList = stringNoPunct.split()
    dictionary = {}

    for word in stringList:
        if word not in dictionary:
            dictionary[word] = 0
        dictionary[word] += 1
    dataframe = pd.DataFrame({'Word': dictionary.keys(), 'Count': dictionary.values()})
    dataframe = dataframe.sort_values('Count', ascending=False)
    return dataframe
# %%
wordString = "When I was in my early teens, I didn’t necessarily think about being a writer, but I did think about working for myself and establishing my own company. I envisioned a life where I would be in charge. I had my mind set that I was going to do great things and that I wasn’t going to be average. Think back to your teenage years. What did you want to be? What did you think your life was going to look like in ten years? Does your reality now match what you thought back then? If not, the good news is that you still have the opportunity to change things and shape your life around your truest self. When you grow older, you face more pressure to enter a career path for reasons that have little to do with our natural preferences. You may choose a type of career for the salary and prestige that it offers. You may have been influenced by the people around you to choose a certain career. It might feel good for a while to have this type of job. If you’re in a job that has a lucrative salary, you will enjoy some material benefits. If you’re not careful, however, you may end up at a point in life where you feel stuck. Life is okay, but it’s not great. You have a decent job, but it’s not very fulfilling. You’ll start to wonder if you’ve made the right choice. You’ll think about the dreams and aspirations you had when you were younger, and you’ll regret not taking any chances. What you do for a living is important. You will spend much of your life working. One of the keys to happiness is doing something for a living that you enjoy. Life is too short to be doing something you don’t like just to pay the bills. The good news is that it doesn’t matter how long you’ve been traveling down a certain path. As long as you’re alive, you have the opportunity to change directions. If you’re just starting out in your career, you will have an easier transition. But even if you’ve been at it for a while, there’s still time to pursue the path you were destined for. No matter how far you’ve gone, the place to look for direction is at the beginning. Take some time to reflect on your earliest memories. Dive deep into your consciousness and rediscover the passion and interest you were born with. Listen to Yourself When it comes to discovering the path you were meant to follow, the decision rests with you and you alone. It’s important not to let outside influences determine what you decide to do. If I was to ask your childhood self what you wanted to be when you grew up, I doubt that you would say you wanted to be an accountant or a management consultant. We all have big dreams at the beginning of our lives, but as we grow older, limitations start embedding into our minds. People you trust and look up to tell you that life has to be certain way. Oftentimes you’re told what you can’t do as opposed to what you can do. Although they aren’t trying to do harm, your parents can be one of the main sources of this limiting belief system being imposed on you. They care about you, love you, and want the best for you. But they aren’t able to see life through your lens, and what they want for you may not be what you want for yourself. They may suggest that you choose the same career that they have or one that seems safe and secure. They will most likely try to steer you away from anything deemed risky or unrealistic. This seemingly harmless advice can have a negative impact on your future. If I could rewind my life and go back to the age of eighteen, I probably wouldn’t have gone to college. At the time however, going to college wasn’t one of several options, it was the only option. My parents sent me to the best schools growing up. They wanted me to get good grades and get accepted to a prestigious university. The problem was that I never really enjoyed school. I was always an average student. The way that school worked didn’t make sense to me. Everyone was so worried about their grades but I always thought to myself, “this isn’t going to matter much in the real world.” I coasted along until it was time to apply for colleges. They make you take standardized tests that colleges look at to find students that they want to recruit. I always did very well on those tests and as a result I had letters coming in from top universities around the country. Harvard sent me a full application to fill out. I remember it being elaborate. My parents pestered me to fill it out, but I never did. My mother still brings up the Harvard application to this day. I settled on a small school in Minnesota. It was the only one I applied to. I continued to be an average student. When I was done with my undergrad my parents suggested graduate school. I told them that I had no plans on furthering my formal education but it went in one ear an out of the other. Every time I talk to my dad on the phone he reminds me that I need to get my MBA. If I had told them at eighteen that I wanted to skip going to college and start my writing career, they wouldn’t have allowed it. They mean well, but they just don’t get it. That’s okay. I’ve come to realize that when you are seeking to do something that’s different, the people around you might not understand it. You’re the one who has to live your life. You’re the one who has to decide what’s right for you and go for it regardless of what the people around you think. It’s up to you to have the courage to do something that most other people wouldn’t. Human beings by nature have the tendency to care about what other people think. This is due to the wiring of our brains. A long time ago in hunter-gatherer societies, social rejection meant alienation from the group, which ultimately meant death. Our brains haven’t quite caught up to the conditions we live in now, so your brain actually believes that social rejection truly means danger."

# %%
wordCount = wordFreq(wordString)
# %%
countDf = pd.DataFrame({'Word': wordCount.keys(), 'Count': wordCount.values()})
# %%
countDf = countDf.sort_values('Count', ascending=False)
# %%
countDf.head(20)
# %%
# work count using nltk and without stop words
string = wordString.lower()
stringNoPunct = re.sub(r'[^\w\s]','', string)
textTokens = word_tokenize(stringNoPunct)
tokensWithoutStopwords = [word for word in textTokens if not word in stopwords.words()]


# %%
wordDict = {}
# %%
for word in tokensWithoutStopwords:
    if word not in wordDict:
        wordDict[word] = 0
    wordDict[word] += 1

wordDf = pd.DataFrame({'Word': wordDict.keys(), 'Count': wordDict.values()})
wordDf = wordDf.sort_values('Count', ascending=False)
print(wordDf.head(20))
# %%
arr = np.array([11,2,4,3,5,6,10,8,-12])
# %%
arr = arr.reshape([3,3])
# %%
arr[0,0]
# %%
def diagonalDifference(arr):
    diag1 = []
    diag2 = []
    for i in range(len(arr)):
        diag1.append(arr[i][i])
        diag2.append(arr[-i-1][i])
    return abs(sum(diag1) - sum(diag2))
# %%
diagonalDifference(arr)
# %%
def plusMinusRatios(arr):
    pos = 0
    neg = 0
    zer = 0
    tot = len(arr)
    for i in range(len(arr)):
        if arr[i] > 0:
            pos += 1
        elif arr[i] < 0:
            neg += 1
        else:
            zer += 1
    
    print("{:.6f}".format(pos/tot))
    print("{:.6f}".format(neg/tot))
    print("{:.6f}".format(zer/tot))
# %%
array = [-4, 3, -9, 0, 4, 1]
# %%
plusMinusRatios(array)
# %%
# Complete the staircase function below.
def staircase(n):
    for i in range(n):
        print(" "*((n-1)-i) + "#"*(i+1))
# %%
staircase(7)
# %%
def miniMaxSum(arr):
    
    maximum = max(arr)
    minimum = min(arr)
    maxSum = sum(arr) - minimum
    minSum = sum(arr) - maximum
    print(str(minSum) + " " + str(maxSum))

# %%
arr = [769082435, 210437958, 673982045, 375809214, 380564127]
# %%
maximum = 0
minimum = 0
for i in range(len(arr)):
    arrDup = arr.copy()
    arrDup.remove(arrDup[i])
    arrSum = sum(arrDup)
    if arrSum > maximum:
        maximum = arrSum
    elif arrSum > 0 or arrSum < minimum:
        minimum = arrSum
    
print(str(minimum) + " " + str(maximum))
# %%
miniMaxSum(arr)
# %%
def birthdayCakeCandles(candles):
    # Write your code here
    maxHeight = max(candles)
    candleCount = 0
    for i in range(len(candles)):
        if candles[i] == maxHeight:
            candleCount += 1
    return candleCount

candles = [3,1,2,3,2,1,1,3]

# %%
birthdayCakeCandles(candles)
# %%
# convert 12 hour to 24 hour time
def timeConversion(s):
    #
    # Write your code here.
    #
    if s[-2:] == 'AM' and s[:2] == "12":
        return "00" + s[2:-2]
    elif s[-2:] == 'AM':
        return s[:-2]
    elif s[-2:] == 'PM'and s[:2] == "12":
        return s[:-2]
    else: 
        return str(int(s[:2])+12) + s[2:-2]
# %%
timeConversion("12:03:45AM")
# %%
def reverseArray(arr):
    reversedList = []
    while arr:
        reversedList.append(arr[-1])
        arr.pop()
    return reversedList
# %%
arr = [1,2,3,4,5]
# %%
reverseArray(arr)
# %%
def sockMerchant(n, arr):
    dictionary = {}
    for num in sockArr:
        if num not in dictionary:
            dictionary[num] = 0
        dictionary[num] += 1

    valuesList = list(dictionary.values())

    pairsList = []
    for item in valuesList:
        pair = item // 2
        pairsList.append(pair)

    return sum(pairsList)
# %%
sockArr = [1,1, 3, 1, 2, 1, 3, 3, 3, 3]
# %%
sockMerchant(9, sockArr)
# %%
def pageFlipCount(n,p):
    if n % 2 == 0:
        right = (n - p + 1)//2
        left = p//2
        if right > left:
            return left
        else:
            return right
    else:
        right = (n - p)//2
        left = p//2
        if right > left:
            return left
        else:
            return right

pageFlipCount(6,5)
# %%
def countingValleys(steps, path):
    # Write your code here
    elevation = 0
    valleys = 0
    for i in range(len(path)):
        if path[i] == "U":
            elevation = elevation + 1
        elif path[i] == "D":
            elevation = elevation - 1
        if elevation == 0 and path[i] == "U":
            valleys += 1
    return valleys

# %%
path = ['U', 'D','D','D','U','U','D','U','U','U','D','D']
# %%
countingValleys(len(path),path)
# %%
def getMoneySpent(keyboards, drives, b):
    #
    # Write your code here.
    #
    maximum = -1
    for i in range(len(keyboards)):
        for j in range(len(drives)):
            if (keyboards[i] + drives[j] > maximum) and (keyboards[i] + drives[j] <= b):
                maximum = keyboards[i] + drives[j]
    return maximum

# %%
keyboards = [4]
drives = [5]
b = 5
# %%
getMoneySpent(keyboards, drives, b)
# %%
def catAndMouse(x, y, z):
    if abs(x-z) > abs(y-z):
        return "Cat B"
    elif abs(x-z) < abs(y-z):
        return "Cat A"
    else:
        return "Mouse C"
# %%
arr = [[4,9,2],[3,5,7],[8,1,5]]

# %%
arr1 = []
for i in arr:
    arr1 += i
# %%
#l = [[a,b,c,d,e,f,g,h,i] for a in range(1,10) for b in range(1,10) for c in range(1,10) for d in range(1,10) for e in range(1,10) for f in range(1,10) for g in range(1,10) for h in range(1,10) for i in range(1,10) if a+b+c == 15 and d+e+f == 15 and g+h+i==15 and a+d+g==15 and b+e+h==15 and c+f+i==15 and a+e+i==15 and c+e+g==15 and len(set([a,b,c,d,e,f,g,h,i]))==9]
# %%
lists = [[2, 7, 6, 9, 5, 1, 4, 3, 8], [2, 9, 4, 7, 5, 3, 6, 1, 8], [4, 3, 8, 9, 5, 1, 2, 7, 6], [4, 9, 2, 3, 5, 7, 8, 1, 6], [6, 1, 8, 7, 5, 3, 2, 9, 4], [6, 7, 2, 1, 5, 9, 8, 3, 4], [8, 1, 6, 3, 5, 7, 4, 9, 2], [8, 3, 4, 1, 5, 9, 6, 7, 2]]
# %%
costValue = []
for i in lists:
    costValue.append(sum([abs(i[j]- arr1[j]) for j in range(9)]))
print(min(costValue))