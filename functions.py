
import time, os
from os import path
from bs4 import BeautifulSoup
import requests
import re
import csv
import spacy
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import langdetect
from langdetect import detect
driver = webdriver.Chrome(executable_path= '/Users/paolaantonicoli/Downloads/chromedriver') #insert the path of the driver
import pandas as pd
import pickle
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
import time
import seaborn as sns
import ipywidgets as widgets
from PyDictionary import PyDictionary
dictionary=PyDictionary()
import heapq

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#welcome
def welcome():
    text = ' '.join(df['plot'])

    # Create and generate a word cloud image:
    wordcloud = WordCloud(background_color="white",width=800, height=400).generate(text)

    # Display the generated image:
    plt.figure( figsize=(40,40) )
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def ask_query():
    """[Needed to create a text box where the user can insert the query to search]

    Returns:
        [Q]: [Text, for getting string you must use Q.value]
    """
    Q = widgets.Text(placeholder='Insert your query here', description='Query:',disabled=False)
    display(Q)
    return Q

##########[DATA COLLECTION]##############

def get_list_of_URLs(page_number):
    """[Get List of URls from the pages BBE]
    Args:
        page_number ([int]): [page number associated to 'https://www.goodreads.com/list/show/1.Best_Books_Ever?page=i' ]

    Returns:
        [list]: [the list of the URLs of books that are present in it]
    """    

    
    href = 'https://www.goodreads.com/list/show/1.Best_Books_Ever?page={}'.format(page_number)
    page_html= driver.get(href)
    page_soup = BeautifulSoup(driver.page_source, features = 'lxml')
    list_of_URLs = [page_soup.find_all('a','class' == 'bookTitle',href= re.compile('^/book/show'))[i]['href'] for i in range(0,100)]
    return list_of_URLs
    
def write_pages(URL_txt):
    """[Returns a doc 'txt_doc' in which are reported the URLs of the first 300 pages]

    Args:
        URL_txt ([.txt]): [a doc in which we will write the the books of the first 300 pages of BestBookEver site]
    """    
    
    #we need to know how many pages are already downloaded in case the process stops 
    if not os.path.exists(URL_txt):
        with open(URL_txt,"w+") as f:
            pages_already_downloaded = 0
    else:
        with open(URL_txt,'r') as f:
            #it can stop only for a page (n of rows/100), since it is updated all at once
            pages_already_downloaded = sum(1 for _ in f)//100
    
    #for each page I collect all the urls present in it and write in the txt
    for i in range(pages_already_downloaded,300):
        page_number = i+1
        URLs = get_list_of_URLs(page_number)
        
        #write in the txt file
        with open(URL_txt , 'a') as f:
            for item in URLs:
                f.write("%s\n" % item)
#        print('page {} saved'.format(page_number))
        time.sleep(5)

######[1.2 DOWNLOAD HTLM PAGES]######
def get_txt(URL_txt):
    """[summary]

    Args:
        URL_txt ([.txt file]): [file were are collected the URLs]

    Returns:
        [type]: [a list of the rows, without '\n']
    """
    with open(URL_txt, 'r') as f:
        txt_lines = f.readlines()
    txt=[line.strip() for line in txt_lines]
    return txt

def write_file(path,html):
    '''
    write html file
    '''
    with open(path, 'w') as f:
        f.write(html)

def is_downloaded(folder,path_):
    """[check if the .html file exists]

    Args:
        folder ([folder in which is placed the book]): [description]
        path_ ([type]): [path of the .html page]

    Returns:
        [type]: [description]
    """
  
    if not os.path.exists(folder):
        os.makedirs(folder)
    return bool(os.path.exists(path_))

def download(folder,page,path):
    
    '''
    Check if the page is downloaded, if not -> download
    '''
    if not is_downloaded(folder,path):
        html_book=requests.get(page).text
        write_file(path,html_book)
        
        
#download all pages from txt file to every folder (100 page for each folder)
def download_books_by_page(URL_txt,main_folder):
    '''
    Download all the book that are in the txt and 
    place them in folders (100 in each)
    '''
    txt = get_txt(URL_txt)
    
    for j in range(0,len(txt)):
        #book we check
        url_book = 'https://www.goodreads.com{}'.format(txt[j])

        #where to save
        folder='{}/page_{}'.format(main_folder,j//100 +1)
        path= folder +'/{}.html'.format(j+1)

        #download if not downloaded
        download(folder,url_book,path)

        time.sleep(1)
 #       print("{} page downloaded".format(j+1))


# ######[1.3 PARSE DOWNLOADED PAGES]########
def scrap_book(href):
    '''
    HTML file associated to the HTML pages of book
    return: bookTitle, ..., URL of the book associated to the HTML page
    
    '''
    page_soup = BeautifulSoup(open(href), features = 'lxml')
    
    if  page_soup.find_all('div', id = "description")[0].get_text().split('\n'):
        description = page_soup.find_all('div', id = "description")[0].get_text().split('\n')
        Plot = max(description, key = len)
    else:
        Plot = ''
    #in order to remove the not english pages
    if detect(Plot) !='en':
        return 'not_english'
    
    bookTitle = page_soup.find_all('h1')[0].contents[0].replace('\n', '').strip()
    
    if page_soup.find_all('a',href =re.compile("/series")):
        bookSeries = page_soup.find_all('a',href =re.compile("/series"))[0].get_text().strip().replace('(','').replace(')','')
    else:
        bookSeries = '-'
    bookAuthors = page_soup.find_all('span', itemprop='name')[0].contents[0]
    
    ratingCount = page_soup.find_all('meta',itemprop='ratingCount')[0].get('content')

    reviewCount = page_soup.find_all('meta',itemprop='reviewCount')[0].get("content")

    ratingValue = page_soup.find_all('span', itemprop="ratingValue")[0].get_text().strip()

    number_of_pages = page_soup.find_all('span', itemprop="numberOfPages")[0].get_text().split()[0]

    publication_section =  page_soup.find_all('div', id="details")[0].contents[3]
    
    publication_date = publication_section.get_text().split('\n')[2].strip()
    
    if len(publication_section)>1:
        first_publication_date =  ' '.join(publication_section.contents[1].get_text().strip('\n ( )').split()[2:])
    else:
        first_publication_date = publication_date
    
    characters = ', '.join([c.get_text() for c in page_soup.find_all('a',href =re.compile("/character"))])

    places =', '.join([c.get_text() for c in page_soup.find_all('a',href =re.compile("/places"))])
    
    URL = page_soup.find_all('link')[0]['href']
    
    
    return(bookTitle,
           bookSeries,
           bookAuthors,
           ratingCount,
           reviewCount,
           ratingValue,
           Plot,
           number_of_pages,
           publication_date,
           first_publication_date,
           characters,
           places,
           URL
          )


#Create the .tsv file
def create_headers(tsv_file):
    with open(tsv_file, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['index',
                            'bookTitle',
                            'bookSeries',
                            'bookAuthors',
                            'ratingCount',
                            'reviewCount',
                            'ratingValue',
                            'plot',
                            'number_of_pages',
                            'publication_date',
                            'first_publication_date',
                            'characters',
                            'places',
                            'URL' ])    
    out_file.close()


def create_tmp(tsv_file, path_, page_number):

    """[creating the tsv file with the required informations]

    Args:
        tsv_file ([.tsv]): [file tsv in wich we want to save the infos]
        path ([path]): [path for the folder in which are saved the html pages]
        page_number ([int]): [page number, whose book we want to parse]
    """    
    if not path.exists(tsv_file):
        create_headers(tsv_file)
        create_tmp(tsv_file,path_,page_number)
    else:
        with open(tsv_file, 'a') as out_file:
            #parse all he books within a page
            for i in range(1,101):
                book_number = (page_number-1)*100+i
                book = '{}/page_{}/{}.html'.format(path_,page_number, book_number)
                tsv_writer = csv.writer(out_file, delimiter='\t')
                try:
                    scrap = scrap_book(book)
                    if scrap != 'not_english':
                        tsv_writer.writerow((book_number,)+scrap)
                except:
                    pass 
                
            out_file.close()
        
    

####[2 SEARCH ENGINE]####

df = pd.read_csv('/Users/paolaantonicoli/Desktop/HW3/CLEANED_BOOK.tsv')

#cleaning the plot
def preprocess(data):
    
    #removing punctuation (not removing numbers because title can be like "1984")
    x=re.sub('[^a-zA-Z]', ' ',data) 
    
    #lowering words
    lower=str.lower(x).split() 
    words=set(stopwords.words('english'))
    
    #removing stopwords
    no_stopwords=[w for w in lower if not w in words]  
    lmtzr = WordNetLemmatizer()
    
    #stemming
    cleaned=[lmtzr.lemmatize(w) for w in no_stopwords] 
    
    
    return (" ".join( cleaned ))

#cleaning the dataframe
def cleaning(df):
    df['plot']=df.apply(lambda x: preprocess(x['plot']),axis=1)
    return df


#saving dict function
def save_dict(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#load dict function
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    

''' 
vocabulary

:key:= word_str
:value:= word_id

'''

def create_vocabulary(df):

    #the Dict we wanna build
    vocabulary = defaultdict()

    #set in which i collect all the terms
    term_set = set()
    plot = list(df['plot'])
    for elem in plot:
        try:
            term_set =term_set.union(set(elem.split()))
        except:
            pass

    #convert the set in list to enumerate
    term_list = list(term_set)

    for i, elem in enumerate(term_list):
        vocabulary[elem]= i 

    save_dict(vocabulary,'vocabulary')

''' 
word_doc (Inverted Index)

:key:= word_id
:value:= list of doc_id that cointains that word

'''
def create_dict(text, book_id, word_doc):
    try: 
        word_list = set(text.split())
        for word in word_list:      
            d[vocabulary[word]]+=[book_id]
    except:
        pass

def create_word_doc(df):
    word_doc = defaultdict(list)
    df.apply(lambda x: create_dict(x['plot'],x['index']), axis = 1)

    save_dict(word_doc,'word_doc')


'''
rev_vocabulary
:key: word_id
:value:= word_str
'''

def create_rev_vocabulary(vocabulary):
    rev_vocabolary = {}
    for key in vocabulary.keys():
        rev_vocabolary[vocabulary[key]]=key
    save_dict(rev_vocabolary,'rev_vocabolary')

'''
tf_word_doc (Inverted Index)

:key:= word_id of j
:value:= list tuple (doc_id of i ,tf_ij)


'''
    
def tf_idf(id_word,id_doc,df):
    word = rev_vocabolary[id_word]
    d_j = df.iloc[id_doc]['plot'] #document j
    n_ij = d_j.count(word)  #number of occurrences of term i in document j
    tf_ij = n_ij/(len(d_j)) 
    
    n = len(df) #total number of docs
    N_i = len(word_doc[id_word]) #number of docs that contain the document i  
    
    idf_i = np.log10(n/N_i)
    
    return tf_ij*idf_i

def create_tf_word_doc(word_doc):
    tf_word_doc = defaultdict(list)

    for word in word_doc.keys():
        for doc in word_doc[word]:
            tf_word_doc[word] += [(doc, tf_idf(word,doc))]
            
    save_dict(tf_word_doc,'tf_word_doc')


'''
word_occ
key: doc_id
values: (word_id, freq_word) freq_word = frequency of word_id in doc
'''
def get_occ(text):
    out = []
    for word in text.split():
        out+= [(vocabulary[word], text.count(word))]
    return out

def create_word_occ(df):
    word_occ = defaultdict(list)
    for i, row in df.iterrows():
        word_occ[i] = get_occ(row['plot'])

'''
occ_word_doc
:key:= word_id of j
:value:= list tuple (doc_id of i ,occ_ij)

'''
def create_occ_word_doc(word_doc,rev_vocabolary):
    occ_word_doc = defaultdict(list)
    for word_id in word_doc.keys():
        word_str = rev_vocabolary[word_id]
        for doc_id in word_doc[word_id]:
            freq = df.iloc[doc_id]['plot'].count(word_str)
            occ_word_doc[word_id] += [(doc_id, freq)]
            
    save_dict(occ_word_doc,'occ_word_doc')

'''
doc_norm
:key: the document_id
:value: the norm of the document

'''

def norm(l):
    return np.linalg.norm(np.array(l))

def create_doc_norm():
    doc_vector = defaultdict(list)
    doc_norm = defaultdict(None)
    for word_val in list(tf_word_doc.values()):
        for tup in word_val:
            doc_id = tup[0]
            tf_idf = tup[1]
            doc_vector[doc_id] += [tf_idf]

    for doc in doc_vector.keys():
        doc_norm[doc] = norm(doc_vector[doc])
    save_dict(doc_norm,'doc_norm')

#########################################################
vocabulary= load_obj('/Users/paolaantonicoli/Downloads/vocabulary')
word_doc= load_obj('/Users/paolaantonicoli/Downloads/word_doc')
tf_word_doc= load_obj('/Users/paolaantonicoli/Downloads/tf_word_doc')
doc_norm = load_obj('/Users/paolaantonicoli/Downloads/doc_norm')
rev_vocabolary = load_obj('/Users/paolaantonicoli/Downloads/rev_vocabolary')
#######
def format_query(q):
    '''
    get in input a str of words
    return a list with stemmed word
    '''
    q = q.split()
    return [preprocess(w) for w in q]
     

def all_equal(l):
    '''
    check if all the documents in the list of tuples (docs, tf_idf) are equal
    '''
    l = [x[0] for x in l]
    return bool(l == [l[0]]*len(l))
    
def update_positions(l,index):
    """[Update position of the index in query function, increase of 1 the index of the doc with the minimum ID]

    Args:
        l ([list]): [list of current tuples (doc, tf_idf)]
        index ([list]): [list of indies for all the values of the queries]

    Returns:
        [list]: [updated index]
    """
    
    #we want to consider the minimum id of the documents, t = list of tuples (doc, tf_idf), t[0] = documents
    min_value= min(l, key = lambda t: t[0])[0]
    
    #update only the index of minimum values 
    for i,elem in enumerate(l):
        if elem[0]==min_value:
            index[i]+=1
            
    return index

def get_prod(l):
    """[Get the doct product between the query and the doc]

    Args:
        l ([list of tuples(doc, tf_idf_word)]): [doc are the same, tf_idf for all the words of the query]

    Returns:
        [list]: [document, value of the dot product between the doc and the words of the query]
    """
    #first element first couple, we know that the documents are the same for all the couples since all_equal ==True
    doc = l[0][0]
    
    #the norm of the doc
    docnorm = doc_norm[doc]
    
    #sum of the tf_idf of the documents 
    comp_sum =  sum([t[1] for t in l]) 
    
    return [doc,comp_sum/docnorm]

    
def query(q,df):
    """[given a query, returns a dictionary key:doc, value: doc product between the doc and the query]

    Args:
        q ([string]): [query in input]
        df ([dataframe]): [dataframe for whome plot we consider the tf_idfs ]

    Returns:
        [type]: [description]
    """
    d_query = defaultdict()
    
    #preprocessing the words of the query
    q = format_query(q)
    
    #list of the ID of the words in the query
    term_id_list = [vocabulary[word] for word in q if word in vocabulary.keys()]
    
    #list of the [(doc, tf_idf)...] for the words in the query
    doc_tf_list = [tf_word_doc[term_id] for term_id in term_id_list]
    
    #sort the id of the Documents
    doc_tf_list.sort(key=lambda x:x[0])
    
    #empty list, will collect the documents that are in the values of every of the words of the query (intersection)
    q_out = []
    #A: Doc1,tf1 , Doc2 .......
    #B: Doc1, .. , Doc3 
    #index for parsing all the documents, increase of one for the minimun document
    index = [0]*len(doc_tf_list)
    while True:
        try:
            #document in position [index] in the values of the dictiorary doc_tf_list
            current = []
            
            #qi = i-th word of the query, docs_of_qi document in the values of tf_word_doc[qi]
            for qi, docs_of_qi  in enumerate(doc_tf_list):
                
                #list of all the couples (document,tf-idf) of the current docs, for every qi
                current += [doc_tf_list[qi][index[qi]]]
                
            index = update_positions(current,index)
            
            #check if all the minimum indicies are equal
            if all_equal(current):
                
                #list of couples doc, score 
                q_out += [get_prod(current)]

        #just for interrupting if index overcome the maximum lenght of one value of the dictonary
        except:
            for couple in q_out:
                
                #dict key = doc, value = score
                d_query[couple[0]] = couple[1]
            return d_query

def df_query(q,df):
    """[returns the df with the similarity score, rows sorted by similarity score]

    Args:
        q ([str]): [query in input]
        df ([dataframe]): []

    Returns:
        [df]: [df with the similarity score, sorted by it]
    """
    d_query = query(q,df)
    df_q = df[df['index'].isin(d_query.keys())]

    #we apply the function score = d_query[doc_index]
    df_q['similarity'] = df_q.apply(lambda x: d_query[x['index']], axis = 1)
    return df_q.sort_values('similarity',ascending = False).reset_index(drop = True)



#######[DEFINE A NEW SCORE]###############
#score by synonimus 

def sinonimi(que):
    """[returns the synonimus of the query]

    Args:
        que ([str]): [query in input]

    Returns:
        [str]: [synonimus of the query]

    """
    l = []
    que = preprocess(que)
    for word in que.split():
        l+=dictionary.synonym(word)

    #splitting composed words (i.g. athletic game -> athletic, game)
    q = ' '.join(l)
    q = ' '.join(set(q.split()).difference(set(que)))
    return q

#score by synonimus 
def query_syns(que,df):
    d_query = defaultdict(lambda: 0)
    d_up= defaultdict(lambda: 0)
    #preprocessing the words of the query
    syns = sinonimi(que)
    q = format_query(syns)

    #list of the ID of the words in the query
    term_id_list = [vocabulary[word] for word in q if word in vocabulary.keys()]
    
    #list of the [(doc, tf_idf)...] for the words in the query
    doc_tf_list= []
    for term_id in term_id_list:
        doc_tf_list += [tf_word_doc[term_id] ]
    
    for list_word in doc_tf_list:
        for tupla in list_word:
            d_query[tupla[0]]+=tupla[1]
    
    for doc in d_query.keys():
        d_query[doc] = d_query[doc]
        
    return d_query

#score by title

def jaccard(A,B):
    """[Jaccard similarity between 2 sets]

    Args:
        A ([set]): [description]
        B ([set]): [description]

    Returns:
        [int]: [Jaccard Similarity]
    """
    union = A.union(B)
    inter = A.intersection(B)
    return len(inter)/len(union)


def score_title(title, q):
    listTitle = title.split()
    listQuery = q.split()
    
    setTitle = set([preprocess(T.strip()) for T in listTitle])
    setQuery = set([preprocess(Q.strip()) for Q in listQuery])
    return jaccard(setTitle,setQuery)

#score_review_rating

def get_score_rating(q,df,ratingCount,reviewCount):
    df_q = df_query(q,df)
    #list whose elments are the logaritmic value of the rating count
    log_rgc = [np.log(x) for x in df_q['ratingCount'] if x>0]

    #list whose elments are the logaritmic value of the review count
    log_rwc = [np.log(x) for x in df_q['reviewCount'] if x>0]

    #norm of the vectors
    total_ratings = np.linalg.norm(np.array(log_rgc))
    total_reviews = np.linalg.norm(np.array(log_rwc))

    #we exclude 0 value to get values between [0,1]
    if ratingCount !=0:
        score_rating = np.log(ratingCount)/total_ratings
    else: 
        score_rating = 0 
        
    if ratingCount !=0:
        score_reviews= np.log(reviewCount)/total_reviews
    else: 
        score_reviews= 0 

    return (score_rating+score_reviews)*0.5 



# score by place
def get_score_place(place,q):
    #instead of considering the Jaccard distance, we consider the ration between the intersection and the lenght of 
    #the list of places
    try:
        q = q.split()
        place = place.lower()
        list_of_places = [loc.strip() for loc in place.replace(',',' ').split()]
        place_set = set(list_of_places)
        q_set = set(q)
        places_present = place_set.intersection(q_set)
        score = len(places_present)/len(place_set)
        return score
    except:
        return 0 

#score by author
def dict_authors(df):
    """[returns a dict that gives the ratio of book an author published respect to the total books in the df]

    Args:
        df ([DataFrame]): [description]

    Returns:
        [dictionary]: [dict that gives the ratio of book an author published respect to the total books in the df]
    """
    dict_authors = defaultdict(lambda: 0 )
    list_of_authors = df['bookAuthors'].to_list()
    for author in list_of_authors:
        dict_authors[author] = list_of_authors.count(author)/np.sqrt(len(list_of_authors))
    return dict_authors

dict_authors = dict_authors(df)

####CLASS BOOK
class Book:    
     
    def __init__(self, index, bookTitle, bookSeries, bookAuthors, ratingCount, reviewCount, ratingValue, plot, number_of_pages, publication_date, first_publication_date, characters, places, URL,similarity):
        self.index = index
        self.bookTitle = bookTitle
        self.bookSeries = bookSeries
        self.bookAuthors = bookAuthors
        self.ratingCount = ratingCount
        self.reviewCount = reviewCount
        self.ratingValue = ratingValue
        self.plot = plot
        self.number_of_pages = number_of_pages
        self.publication_date = publication_date
        self.first_publication_date = first_publication_date
        self.characters = characters
        self.places = places
        self.URL = URL
        self.similarity = similarity if not None else '-'

        
    def score(self):
        return self.similarity
    
    def score_authors(self):
        return dict_authors[self.bookAuthors]
    
    def score_syns(self,q,df):
        d_query_syns = query_syns(q,df)
        return d_query_syns[self.index]
    
    def score_title(self,q):
        return score_title(self.bookTitle, q)
    
    def score_review_rating(self,q,df):
        return get_score_rating(q,df,self.ratingCount,self.reviewCount)
    
    def score_place(self,q):
        return get_score_place(self.places,q)
    
    #total score of each obj
    def total_score(self, weights,q,df):
        scores = np.array([
            self.score(),
            self.score_authors(),
            self.score_syns(q,df),
            self.score_title(q),
            self.score_review_rating(q,df),
            self.score_place(q)])
        
        #we consider a weighted average between the scores
        w = np.array(weights)
        return np.dot(scores,w)



# Use heaps to extract top k rows
scores = ['score','author','synon','title','reviews & rating','palce']

def widget(feature):
    return widgets.FloatSlider(
    value=7.5,
    min=0,
    max=10.0,
    step=0.1,
    description='{}'.format(feature),
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
    )

#used to make the user choise the weight to give to each feature
def collect_weights(score=scores):
    weights = []
    for score in scores:
        w = widget(score)
        weights+=[w]
        display(w)
    return weights

def top_k_df(Q,weights,df,k):
    s_query= df_query(Q,df)
    query_books = [Book(**kwargs) for kwargs in s_query.to_dict(orient='records')]
    s_query['my_score'] = [b.total_score(weights, Q, df) for b in query_books]
    s_query_list = s_query[['bookTitle', 'plot', 'URL','similarity','my_score']].values.tolist()
    heapq.heapify(s_query_list)
    top_k = heapq.nlargest(k, s_query_list, key = lambda x: x[4])
    top_k_df = pd.DataFrame(data=top_k, columns=['bookTitle', 'plot', 'URL','similarity', 'my_score'])
    return top_k_df


##########[MAKE A NICE VISUALIZATION]#########

#function to split serie name and number of serie
def get_serie_name(s):
    """[get the name and the number of a book belonging to a serie]

    Args:
        s ([str]): [ SerieName #NumberOfSerie]

    Returns:
        [list]: [l[0]: serie name, l[1]: number of serie]
    """
    s = s.split('#')
    l = ['-','-']
    if len(s) ==2:
        s[1] =s[1].replace('–','-')
        return s
    else:
        return l

#funtion to get only the Year 
def get_year(data):
    """[We want to remove day and mounth]

    Args:
        data ([str]): [Data in Day Mounth Year format]

    Returns:
        [str]: [Year]
    """
    return str(data).split()[-1].strip()

def start_from(year, starting_year):
    """[Returns the years from the first publication of the book]

    Args:
        year ([str]): [Year of the book]
        starting_year ([str]): [Year of pulication of the first book of the serie]

    Returns:
        [int]: [Years since the publication of the first book of the serie]
    """
    return int(year)-int(starting_year)


def get_df_serie(df):
    """[returns a df in which are present the first 10 series of the original df, in which are indicated the serie and the serie number of each book
    the years of publication since the first book of the serie, and the cumulative number of pages ]

    Args:
        df ([DataFrame]): [description]

    Returns:
        [DataFrame]: [a df in which are present the first 10 series of the original df, in which are indicated the serie and the serie number of each book]
    """
    df['Serie'] = df.apply(lambda x: get_serie_name(x['bookSeries'])[0],axis = 1 )
    df['Number'] = df.apply(lambda x: get_serie_name(x['bookSeries'])[1],axis = 1 )
    
    #removing all the wrong values of numbers and the book that doesn't have a serie
    df_serie = df.loc[df['Number'].str.contains('^((?![-,.]).)*$')][['index','bookTitle','Serie','Number','number_of_pages','first_publication_date']]
    
    #the first 10 series in order of appearance
    first_10= list(df_serie['Serie'])[:11]
    
    #filtering the df
    df_serie = df_serie.loc[df['Serie'].isin(first_10)]
    
    df_serie['Year'] = df.apply(lambda x : get_year(x['first_publication_date']),axis = 1)

    ord_df_series = pd.DataFrame()
    for serie_name, serie in df_serie.groupby('Serie'):
        serie = serie.sort_values('Year')
        starting_year = min(serie['Year'])
        serie['Year'] = serie.apply(lambda x: start_from(x['Year'],starting_year),axis = 1)
        serie['Sum of Pages']= serie['number_of_pages'].cumsum()
        ord_df_series = pd.concat([ord_df_series,serie])
    return ord_df_series

def get_previous(y,df_series):
    """[Returns a df with all the books published before y]

    Args:
        y ([int]): [year]
        df_series ([df]): [description]

    Returns:
        [DataFrame]: [df with all the books published before y]
    """
    d = df_series[df_series['Year']<=y]
    return d

def serie_plot(df_series):
    """[dynamic plot of the cumulative number of pages of the book series]

    Args:
        df_series ([DataFrame]): [description]
    """
    df_series =df_series.sort_values('Year')
    s = pd.DataFrame()
    for y, year in df_series.groupby('Year'):
        cs = pd.DataFrame()
        cs = get_previous(y,df_series)
        cs['Years'] = y
        s = pd.concat([s,cs])                 
    fig = px.line(s,x = 'Year', y="Sum of Pages",color = 'Serie',animation_frame= 'Years')
    fig.show()


#static plot
def static_serie_plot(df_serie):
    """[Static Plot of the cumul numb. of pages]

    Args:
        df_serie ([df]): [description]
    """
    sns.set_context("notebook", font_scale=3, rc={"lines.linewidth": 10,'lines.markersize': 30})
    sns.set_style("whitegrid")

    fig_dims = (40, 20)
    palette = sns.color_palette("pastel")

    fig, ax =plt.subplots(1,figsize = fig_dims)
    sns.lineplot(data=df_serie, x="Year", y="Sum of Pages", hue="Serie",marker = 'o',markers = True ,palette = 'Paired')
    plt.legend(fontsize='40', title_fontsize='40')

    fig.show()
############[ALGORITHMS]########################

#Recursiv
def maxord(s,i = -1):    
    if i == -1:
        return max([maxord(s,i) for i in range(len(s))])
    l=[maxord(s,j) for j in range(i) if s[j]<s[i]]
    if l==[]:
        return 1
    else:
        
        return 1+max(l)


def plot_maxord_runningtime(strings):
    df_maxord = pd.DataFrame()
    time_s = []
    length_s = []
    for i in range(len(strings)):
        input_string = strings[i]
        start = time.time()
        maxord(input_string)
        end= time.time()
        time_run=end-start
        l = len(input_string)
        time_s.append(time_run)
        length_s.append(l)

    df_maxord['lenght'] = length_s
    df_maxord['time'] = time_s
    df_maxord['teorical running time (2^n)'] = df_maxord.apply(lambda x: 2**(x['lenght']) ,axis = 1)
    sns.set(font_scale = 2.5)
    fig_dims = (40, 10)
    fig, ax =plt.subplots(1,2,figsize = fig_dims)
    sns.scatterplot(data=df_maxord, x="lenght", y="time",color = '#99B898',s = 250,ax=ax[0])
    sns.scatterplot(data=df_maxord, x="lenght", y="teorical running time (2^n)",color = '#E84A5F',s= 250,ax=ax[1])
    fig.show()


###Dynamic
def maxord_d(s,i = -1, m = []):    
    if i==-1:
        return max([maxord_d(s,i=i,m=[0]*len(s)) for i in range(len(s))])
    if m[i]!= 0:
        return m[i]
    s = s[:i+1]
    l=[j for j in range(i) if s[j]<s[i]]
    if l==[]:
        m[i]=1
        return 1
    else:
       
        m[i]=max(m[j] for j in range(i) if s[j]<s[i])+1 
        return 1+max(maxord_d(s,j,m) for j in l)

def plot_maxord_dynamic_runningtime(strings):
    df_maxord = pd.DataFrame()
    time_s = []
    length_s = []
    for i in range(len(strings)):
        input_string = strings[i]
        start = time.time()
        maxord_d(input_string)
        end= time.time()
        time_run=end-start
        l = len(input_string)
        time_s.append(time_run)
        length_s.append(l)

    df_maxord['lenght'] = length_s
    df_maxord['time'] = time_s
    df_maxord['teorical running time (n^2)'] = df_maxord.apply(lambda x: (x['lenght'])**2 ,axis = 1)
    sns.set(font_scale = 2.5)
    fig_dims = (40, 10)
    fig, ax =plt.subplots(1,2,figsize = fig_dims)
    sns.scatterplot(data=df_maxord, x="lenght", y="time",s = 250,ax=ax[0],color = "#BA55D3")
    sns.scatterplot(data=df_maxord, x="lenght", y="teorical running time (n^2)",s= 250,ax=ax[1],color = '#48D1CC')
    fig.show()