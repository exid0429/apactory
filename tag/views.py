from django.shortcuts import render
import networkx
import re
import time
import requests
from django.http import JsonResponse
from django.views import View
from selenium import webdriver
from konlpy.tag import Komoran
from konlpy.tag import Okt
import docx2txt
from io import StringIO
import math
from selenium.webdriver.chrome.options import Options
text_result = docx2txt.process("/Users/munyeonglee/특허/1번특허.docx")
opts = Options()
opts.add_argument("user-agent=Chrome/51.0.2704.103")



driver = webdriver.Chrome('/Users/munyeonglee/chromedriver',chrome_options=opts)




time.sleep(3)
f = open("/Users/munyeonglee/nlp/fire.txt","r")
stop_words = f.read()
stop_words = stop_words.split("\n")

f = open("/Users/munyeonglee/nlp/tag.txt","r")
tag = f.read()
tag = tag.replace("\n","")
tag = tag.replace('"',"")
tag = tag.replace('[',"")
tag = tag.replace(']',"")
tag = tag.replace(" ","")
tag = tag.split(",")





class RawSentence:
    def __init__(self, textIter):
        if type(textIter) == str: self.textIter = textIter.split('\n')
        else: self.textIter = textIter
        self.rgxSplitter = re.compile('([.!?:](?:["\']|(?![0-9])))')

    def __iter__(self):
        for line in self.textIter:
            ch = self.rgxSplitter.split(line)
            for s in map(lambda a, b: a + b, ch[::2], ch[1::2]):
                if not s: continue
                yield s

class RawSentenceReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.rgxSplitter = re.compile('([.!?:](?:["\']|(?![0-9])))')

    def __iter__(self):
        for line in open(self.filepath, encoding='utf-8'):
            ch = self.rgxSplitter.split(line)
            for s in map(lambda a, b: a + b, ch[::2], ch[1::2]):
                if not s: continue
                yield s

class RawTagger:
    def __init__(self, textIter, tagger = None):
        if tagger:
            self.tagger = tagger
        else :
            self.tagger = Komoran()
        if type(textIter) == str: self.textIter = textIter.split('\n')
        else: self.textIter = textIter
        self.rgxSplitter = re.compile('([.!?:](?:["\']|(?![0-9])))')

    def __iter__(self):
        for line in self.textIter:
            ch = self.rgxSplitter.split(line)
            for s in map(lambda a,b:a+b, ch[::2], ch[1::2]):
                if not s: continue
                yield self.tagger.pos(s)

class RawTaggerReader:
    def __init__(self, filepath, tagger = None):
        if tagger:
            self.tagger = tagger
        else :
            self.tagger = Komoran()
        self.filepath = filepath
        self.rgxSplitter = re.compile('([.!?:](?:["\']|(?![0-9])))')

    def __iter__(self):
        for line in open(self.filepath, encoding='utf-8'):
            ch = self.rgxSplitter.split(line)
            for s in map(lambda a,b:a+b, ch[::2], ch[1::2]):
                if not s: continue
                yield self.tagger.pos(s)

class TextRank:
    def __init__(self, **kargs):
        self.graph = None
        self.window = kargs.get('window', 5)
        self.coef = kargs.get('coef', 1.0)
        self.threshold = kargs.get('threshold', 0.005)
        self.dictCount = {}
        self.dictBiCount = {}
        self.dictNear = {}
        self.nTotal = 0


    def load(self, sentenceIter, wordFilter = None):
        def insertPair(a, b):
            if a > b: a, b = b, a
            elif a == b: return
            self.dictBiCount[a, b] = self.dictBiCount.get((a, b), 0) + 1

        def insertNearPair(a, b):
            self.dictNear[a, b] = self.dictNear.get((a, b), 0) + 1

        for sent in sentenceIter:
            for i, word in enumerate(sent):
                if wordFilter and not wordFilter(word): continue
                self.dictCount[word] = self.dictCount.get(word, 0) + 1
                self.nTotal += 1
                if i - 1 >= 0 and (not wordFilter or wordFilter(sent[i-1])): insertNearPair(sent[i-1], word)
                if i + 1 < len(sent) and (not wordFilter or wordFilter(sent[i+1])): insertNearPair(word, sent[i+1])
                for j in range(i+1, min(i+self.window+1, len(sent))):
                    if wordFilter and not wordFilter(sent[j]): continue
                    if sent[j] != word: insertPair(word, sent[j])

    def loadSents(self, sentenceIter, tokenizer = None):
        def similarity(a, b):
            n = len(a.intersection(b))
            return n / float(len(a) + len(b) - n) / (math.log(len(a)+1) * math.log(len(b)+1))

        if not tokenizer: rgxSplitter = re.compile('[\\s.,:;-?!()"\']+')
        sentSet = []
        for sent in filter(None, sentenceIter):
            if type(sent) == str:
                if tokenizer: s = set(filter(None, tokenizer(sent)))
                else: s = set(filter(None, rgxSplitter.split(sent)))
            else: s = set(sent)
            if len(s) < 2: continue
            self.dictCount[len(self.dictCount)] = sent
            sentSet.append(s)

        for i in range(len(self.dictCount)):
            for j in range(i+1, len(self.dictCount)):
                s = similarity(sentSet[i], sentSet[j])
                if s < self.threshold: continue
                self.dictBiCount[i, j] = s

    def getPMI(self, a, b):
        import math
        co = self.dictNear.get((a, b), 0)
        if not co: return None
        return math.log(float(co) * self.nTotal / self.dictCount[a] / self.dictCount[b])

    def getI(self, a):
        import math
        if a not in self.dictCount: return None
        return math.log(self.nTotal / self.dictCount[a])

    def build(self):
        self.graph = networkx.Graph()
        self.graph.add_nodes_from(self.dictCount.keys())
        for (a, b), n in self.dictBiCount.items():
            self.graph.add_edge(a, b, weight=n*self.coef + (1-self.coef))

    def rank(self):
        return networkx.pagerank(self.graph, weight='weight')

    def extract(self, ratio = 0.1):
        ranks = self.rank()
        cand = sorted(ranks, key=ranks.get, reverse=True)[:int(len(ranks) * ratio)]
        pairness = {}
        startOf = {}
        tuples = {}
        result = []
        z = 0
        for k in cand:
            tuples[(k,)] = self.getI(k) * ranks[k]
            for l in cand:
                if k == l: continue
                pmi = self.getPMI(k, l)
                if pmi: pairness[k, l] = pmi

        for (k, l) in sorted(pairness, key=pairness.get, reverse=True):
            
            a = k[0] + " " +  l[0]
            result.append(a)
            #print(result[z])
            z= z+1
            #print(k[0],l[0])
            if k not in startOf: startOf[k] = (k, l)

        for (k, l), v in pairness.items():
            pmis = v
            rs = ranks[k] * ranks[l]
            path = (k, l)
            tuples[path] = pmis / (len(path) - 1) * rs ** (1 / len(path)) * len(path)
            last = l
            while last in startOf and len(path) < 7:
                if last in path: break
                pmis += pairness[startOf[last]]
                last = startOf[last][1]
                rs *= ranks[last]
                path += (last,)
                tuples[path] = pmis / (len(path) - 1) * rs ** (1 / len(path)) * len(path)

        used = set()
        both = {}
        for k in sorted(tuples, key=tuples.get, reverse=True):
            if used.intersection(set(k)): continue
            both[k] = tuples[k]
            for w in k: used.add(w)

        #for k in cand:
        #    if k not in used or True: both[k] = ranks[k] * self.getI(k)

        return result

    def summarize(self, ratio = 0.333):
        r = self.rank()
        ks = sorted(r,key=r.get, reverse=True)[:int(len(r)*ratio)]
        return ' '.join(map(lambda k:self.dictCount[k],sorted(ks)))

tr = TextRank(window=5, coef=1)
tagger = Komoran()
#token = re.sub("\.","",v)
#token = tagger.morphs(token)
#print(type(token))
#print(type(tr))


#tr.loadSents(RawSentence(v), lambda sent: filter(lambda x:x not in stopword and x[1] in ('NNG', 'NNP', 'VV', 'VA')and len(x[0])>2, tagger.pos(sent)))
#print('Build...')
#tr.build()
#ranks = tr.rank()

#for k in sorted(ranks, key=ranks.get, reverse=True)[:50]:
#    print("\t".join([str(k), str(ranks[k]), str(tr.dictCount[k])]))
#print(tr.summarize(0.2))


stopword = set(stop_words)
tr.load(RawTagger(text_result), lambda w: w not in stopword and (w[1] in ('NNG', 'NNP', 'VV', 'VA')))
#print('Build...')
tr.build()
number = []

t = tr.extract(0.05)
for i in t:
    for j in tag:
        driver.get("https://www.google.com/")
        time.sleep(2)
        search = '"' + i + j + '"'
        input = driver.find_element_by_name("q")
        input.send_keys(search)
        input.submit()
        num = driver.find_element_by_css_selector('#result-stats').text
        if num:
            num = num.split("약")[1]
            num = int(num.split("개")[0].replace(",",""))
        
            number.append(num)
            print(i,j,num)
        else:
            pass
            
        

print(number)


print(t)

#for k in sorted(kw, key=kw.get, reverse=True):
#    print("%s\t%g" % (k, kw[k]))
