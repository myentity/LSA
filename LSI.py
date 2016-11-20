import matplotlib
from pandas.tools.plotting import radviz
import pandas as pd
import numpy
from numpy import linalg
import scipy

import nltk
from nltk.stem.snowball import RussianStemmer
stemmer = RussianStemmer()

import bokeh
from bokeh.models import ColumnDataSource, HoverTool, HBox, VBoxForm
from bokeh.models.widgets import Slider, Select, TextInput
from bokeh.io import curdoc
from bokeh.sampledata.movies_data import movie_path
from bokeh.plotting import figure, output_file, output_notebook, show, ColumnDataSource, gridplot

def searchword(lsa, word):
	distance = []
	doc_num = []
	count = 0
	#print('номер документа в списке  | растояние |  документ разложеный на коды  |  сам документ')
	for res in lsa.find(word):
		if res[5] == "СОВПАДЕНИЕ": count += 1
		distance.append(res[4])  
		#label = text[res[0]]   
		#doc_num.append(label[:50])
		doc_num.append(res[0]+1)
		#doc_num.append(res[0])
		#print('номер документа в списке  | растояние |  документ разложеный на коды  |  сам документ')
		print (res[0]+1, res[4], res[5])
		#print('k - номер документа\t|\tдокумент в кодах\t|\tax\t|\tay\t|\tевклидово расстояние')
		#print(res)
	print('количество документов, содержащих данную форму слова', count)
	res = [doc_num, distance]
	return res

class LSI(object):
	def __init__(self, stopwords, ignorechars, docs):
		# все слова которые встречаются в документах, и содержит номера документов в которых встречается каждое слово
		self.wdict = {}
		# dictionary - Ключевые слова в матрице слева   содержит коды слов
		self.dictionary = []
		self.fulldictionary = []
		# слова которые исключаем из анализа типа и, в, на
		self.stopwords = stopwords
		#содержит документы, в которых слова заменены их кодами из словаря        
		self.docs = []
		if type(ignorechars) == str: ignorechars = ignorechars.encode('utf-8')
		self.ignorechars = ignorechars
		# инициализируем сами документы
		for doc in docs: self.add_doc(doc)
		#print('Матрица wdict без стоп слов')    
		#print(self.wdict)
        
	def prepare(self):
		self.build()
		self.calc()

	def dic(self, word, add = False):
		if type(word) == str: word = word.encode('utf-8')
		# чистим от лишних символов
		word = word.lower().translate(None, self.ignorechars)
		word = word.decode('utf-8')
		fullword = word
		# приводим к начальной форме NLTK http://text-processing.com/demo/stem/ 
		word = stemmer.stem(word)
		#print(word)
		#проверяем на стопслова
		if word not in self.stopwords: 
			# если слово есть в словаре возвращаем его номер
			if word in self.dictionary: return self.dictionary.index(word)
			else:
				# если нет и стоит флаг автоматически добавлять то пополняем словари возвращаем код слова
				if add:
					#self.ready = False
					self.dictionary.append(word) 
					self.fulldictionary.append(fullword) 
					return len(self.dictionary) - 1
				else: return None
		else: return -1      

	def add_doc(self, doc):
		#берем каждое слово из текущего документа и добавляем в dictionary  в self.dic
		#в words записываем вектор из номеров слов из словаря для данного документа     
		words = [self.dic(word, True) for word in doc.lower().split()]
		#print('документ',words) 
		words = [x for x in words if x != -1]
		#print('документ',words)     
		#print('стопы', self.stopwords)  
		self.docs.append(words)
		for word in words:
			if self.dictionary[word] in self.stopwords:  
				#print('стопслово')
				continue
			elif word in self.wdict:   self.wdict[word].append(len(self.docs) - 1)
			else:                      self.wdict[word] = [len(self.docs) - 1]

	def build(self):
		#print('Матрица wdict без стоп слов')    
		#print(self.wdict) 
		# убираем одиночные слова
		self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
		self.keys.sort()
		#print('Ключи wdict без стоп слов')  
		#print(self.keys, len(self.keys))        
		# создаём пустую матрицу  
		self.A = numpy.zeros([len(self.keys), len(self.docs)])
		# наполняем эту матрицу
		for i, k in enumerate(self.keys):
			for d in self.wdict[k]:
				#должны определить, сколько раз слово встречается в документе 
                #d - номер документа, в котором есть слово
				self.A[i,d] += 1     

	def calc(self):
		#self.TFIDF()
		#сингулярное разложение матриц (U, S Vt)
		self.U, self.S, self.Vt = numpy.linalg.svd(self.A)

	#определяем важность термина в зависимости от его встречаемости
	def TFIDF(self):
		# всего кол-во слов на документ
		wordsPerDoc = numpy.sum(self.A, axis=0)
		# сколько документов приходится на слово
		docsPerWord = numpy.sum(numpy.asarray(self.A > 0, 'i'), axis=1)
		rows, cols = self.A.shape
		for i in range(rows):
			for j in range(cols):
				self.A[i,j] = numpy.round((self.A[i,j] / wordsPerDoc[j]) * numpy.log(float(cols) / docsPerWord[i]), 3)

	def dump_src(self):
		self.prepare()
		#print('Матрица А частот слов в документах без стоп слов (указаны полные версии слов без стемминга)')  
		#for i, row in enumerate(self.A):
			#print (row, self.fulldictionary[self.keys[i]])
		self.TFIDF()
		#print('Матрица TF-IDF частот слов в документах без стоп слов (указаны полные версии слов без стемминга)')  
		#for i, row in enumerate(self.A):
			#print (row, self.fulldictionary[self.keys[i]])

	def print_svd(self):
		self.prepare()
		self.TFIDF()
		print ('Сингулярные значения')
		print (numpy.round(self.S, 3))
		#print ('Первые 3 колонки U матрица (указаны полные версии слов без стемминга)')
		#for i, row in enumerate(numpy.round(self.U,3)):
			#print (self.fulldictionary[self.keys[i]], numpy.round(row[0:3],3))
		print ('Первые 3 строчки Vt матрица')
		print (numpy.round(-1*self.Vt[0:3, :], 3))

	def find(self, word):
		count = 0
		self.prepare()
		idx = -1
		idx = self.dic(word)
		if idx == -1:
			print ('слово не встерчается')
			return []
		if not idx in self.keys:
			print ('слово отброшено')
			return []
		idx = self.keys.index(idx)
		print ('слово в поиске --- ', word, '= в словаре --- ', self.dictionary[self.keys[idx]], '.\n')
		# получаем координаты слова
		wx, wy = (-1 * self.U[:, 1:3])[idx]
		print ('координаты слова (idx, wx, wy, word) {}\t{:0.2f}\t{:0.2f}\t{}\n'.format(idx, wx, wy, word))
		arts = []
		xx, yy = -1 * self.Vt[1:3, :]
		for k, v in enumerate(self.docs):
			#работаем только с документами, в кодах которого есть это слово            
			if idx in v: inc = "СОВПАДЕНИЕ"
			else: inc = ""
			# получаем координаты документа
			ax, ay = xx[k], yy[k]
			#вычисляем расстояние между словом и документом
			dx, dy = float(wx - ax), float(wy - ay)
			arts.append((k, v, ax, ay, numpy.sqrt(dx * dx + dy * dy), inc))
		print('документы, отсортированные по ближайшему расстоянию (евклидово расстояние)\n')
		return sorted(arts, key = lambda a: a[4])