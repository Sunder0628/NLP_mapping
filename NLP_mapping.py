#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#removed the duplicates in excel data sets( manuf_data and retailers_data).we are going to open this file with Python and split sentences.
#Program will open file and read its content. Then it will add tokenized sentences into the array for word tokenization.


# In[ ]:


import nltk
import gensim
from nltk.tokenize import word_tokenize, sent_tokenize

file_docs = []

with open ('manf.csv') as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file_docs.append(line)

print("Number of documents:",len(file_docs))
gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in file_docs]


# In[ ]:


#Once we added tokenized sentences in array, then tokenize words for each sentence. Gensim requires the words (\tokens) be converted to unique ids. So, using Gensim create a Dictionary object that maps each word to a unique id. Lets convert our sentences to a [list of words] and pass it to the corpora.Dictionary() object.


# In[ ]:


gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in file_docs]
dictionary = gensim.corpora.Dictionary(gen_docs)
print(dictionary.token2id)


# In[ ]:


#Now, we are going to create similarity object. The main class is Similarity, which builds an index for a given set of documents.The Similarity class splits the index into several smaller sub-indexes, which are disk-based. Let's just create similarity object then you will understand how we can use it for comparing.


# In[ ]:


corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

tf_idf = gensim.models.TfidfModel(corpus)
for doc in tfidf[corpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

  #building the index  
sims = gensim.similarities.Similarity('workdir/',tf_idf[corpus],
                                        num_features=len(dictionary))


# In[ ]:


#Once the index is built, we are going to calculate how similar is this query document to each document in the index. So, create second .txt file (retailer data)which will include query documents or sentences and tokenize them as we did before.


# In[ ]:


file2_docs = []

with open ('retail.csv') as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file2_docs.append(line)

print("Number of documents:",len(file2_docs))  
for line in file2_docs:
    query_doc = [w.lower() for w in word_tokenize(line)]
    query_doc_bow = dictionary.doc2bow(query_doc)


# In[ ]:


#At this stage, we will see similarities between the query and all index documents. To obtain similarities of our query document against the indexed documents:


# In[ ]:


# perform a similarity query against the corpus
query_doc_tf_idf = tf_idf[query_doc_bow]
# print(document_number, document_similarity)
print('Comparing Result:', sims[query_doc_tf_idf]) 
#Cosine measure returns similarities in the range (the greater, the more similar).


# In[ ]:


#to calculate average similarity of query document. At this time, we are going to import numpy to calculate sum of these similarity outputs.


# In[ ]:


import numpy as np

sum_of_sims =(np.sum(sims[query_doc_tf_idf], dtype=np.float32))
print(sum_of_sims)


# In[ ]:


#To calculate average similarity we have to divide this value with count of documents.


# In[ ]:


percentage_of_similarity = round(float((sum_of_sims / len(file_docs)) * 100))
print(f'Average similarity float: {float(sum_of_sims / len(file_docs))}')
print(f'Average similarity percentage: {float(sum_of_sims / len(file_docs)) * 100}')
print(f'Average similarity rounded percentage: {percentage_of_similarity}')


# In[ ]:


avg_sims = [] # array of averages

# for line in query documents
for line in file2_docs:
        # tokenize words
        query_doc = [w.lower() for w in word_tokenize(line)]
        # create bag of words
        query_doc_bow = dictionary.doc2bow(query_doc)
        # find similarity for each document
        query_doc_tf_idf = tf_idf[query_doc_bow]
        # print (document_number, document_similarity)
        print('Comparing Result:', sims[query_doc_tf_idf]) 
        # calculate sum of similarities for each query doc
        sum_of_sims =(np.sum(sims[query_doc_tf_idf], dtype=np.float32))
        # calculate average of similarity for each query doc
        avg = sum_of_sims / len(file_docs)
        # print average of similarity for each query doc
        print(f'avg: {sum_of_sims / len(file_docs)}')
        # add average values into array
        avg_sims.append(avg)  
   # calculate total average
    total_avg = np.sum(avg_sims, dtype=np.float)
    # round the value and multiply by 100 to format it as percentage
    percentage_of_similarity = round(float(total_avg) * 100)
    # if percentage is greater than 100
    # that means documents are almost same
    if percentage_of_similarity >= 100:
        percentage_of_similarity = 100

