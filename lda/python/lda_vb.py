
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from gensim import corpora, models
from time import time
from scipy import sparse

import pdb

np.random.seed(0)

def plot_perplexity_iter(A_tfidf, num_topics):
    
    print "computing perplexity vs iter..."
    max_iter = 5
    perplexity = []
    em_iter = []
    for sweep in range(1,max_iter+1):
        lda = LatentDirichletAllocation(n_topics = num_topics, max_iter=sweep, learning_method='online', batch_size = 512, random_state=0, n_jobs=-1)    
        tic = time()
        lda.fit(A_tfidf)  #online VB
        toc = time()
        print "sweep %d, elapsed time: %.4f sec" %(sweep, toc - tic)
        perplexity.append(lda.perplexity(A_tfidf))
        em_iter.append(lda.n_batch_iter_)
    #end    
    np.save('./data/perplexity_iter.npy', perplexity)
    
    f = plt.figure()
    plt.plot(em_iter, perplexity, color='b', marker='o', lw=2.0, label='perplexity')
    plt.title('Perplexity (LDA, online VB)')
    plt.xlabel('EM iter')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.legend()
    plt.show()
    f.savefig('./figures/perplexity_iter.png')
        
def plot_perplexity_topics(A_tfidf):
    
    print "computing perplexity vs K..."
    max_iter = 5    #based on plot_perplexity_iter()
    #num_topics = np.linspace(2,20,5).astype(np.int)
    num_topics = np.logspace(1,2,5).astype(np.int)
    perplexity = []
    em_iter = []
    for k in num_topics:
        lda = LatentDirichletAllocation(n_topics = k, max_iter=max_iter, learning_method='online', batch_size = 512, random_state=0, n_jobs=-1)
        tic = time()
        lda.fit(A_tfidf)  #online VB
        toc = time()
        print "K= %d, elapsed time: %.4f sec" %(k, toc - tic)
        perplexity.append(lda.perplexity(A_tfidf))
        em_iter.append(lda.n_batch_iter_)
    #end
    
    np.save('./data/perplexity_topics.npy', perplexity)
    np.save('./data/perplexity_topics2.npy', num_topics)    
    
    f = plt.figure()
    plt.plot(num_topics, perplexity, color='b', marker='o', lw=2.0, label='perplexity')
    plt.title('Perplexity (LDA, online VB)')
    plt.xlabel('Number of Topics, K')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.legend()
    plt.show()
    f.savefig('./figures/perplexity_topics.png')

def plot_perplexity_batch(A_tfidf, num_docs):
    
    print "computing perplexity vs batch size..."
    max_iter = 5
    num_topics = 10
    batch_size = np.logspace(6, 10, 5, base=2).astype(int)
    perplexity = np.zeros((len(batch_size),max_iter))
    em_iter = np.zeros((len(batch_size),max_iter))
    for ii, mini_batch in enumerate(batch_size):
        for jj, sweep in enumerate(range(1,max_iter+1)):
            lda = LatentDirichletAllocation(n_topics = num_topics, max_iter=sweep, learning_method='online', batch_size = mini_batch, random_state=0, n_jobs=-1)
            tic = time()
            lda.fit(A_tfidf)  #online VB
            toc = time()
            print "sweep %d, elapsed time: %.4f sec" %(sweep, toc - tic)
            perplexity[ii,jj] = lda.perplexity(A_tfidf)
            em_iter[ii,jj] = lda.n_batch_iter_
        #end
    #end
    np.save('./data/perplexity.npy', perplexity)
    np.save('./data/em_iter.npy', em_iter)    
    
    f = plt.figure()
    for mb in range(len(batch_size)):
        plt.plot(em_iter[mb,:], perplexity[mb,:], color=np.random.rand(3,), marker='o', lw=2.0, label='mini_batch: '+str(batch_size[mb]))
    plt.title('Perplexity (LDA, online VB)')
    plt.xlabel('EM iter')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.legend()
    plt.show()
    f.savefig('./figures/perplexity_batch.png')

def display_topics(model, dictionary, num_words):
    topic_file = open('./top_words.txt', 'w')    
    for topic_idx, topic in enumerate(model.components_):
        top_words = " ".join([dictionary[i] for i in topic.argsort()[:-num_words-1:-1]])
        topic_file.write(top_words+'\n')        
        print "topic: %d" %topic_idx
        print top_words
    #end
    topic_file.close()
        
if __name__ == "__main__":

    #parameters
    num_features = 1000  #vocabulary size
    num_topics = 10      #fixed for LDA
             
    #load data
    dataset = '20newsgroups'  #'AP'
    
    """    
    #Associated Press (use gensim to read in the corpus)
    print "loading AP dataset..."
    AP_corpus = corpora.bleicorpus.BleiCorpus('./data/ap.dat', fname_vocab='./data/vocab.txt')
    tfidf = models.tfidfmodel.TfidfModel(corpus = AP_corpus)    
    AP_tfidf = list(tfidf[AP_corpus])

    #gensim LDA: topics make sense, but it's more hidden (harder to access internals)
    #AP_dict = corpora.Dictionary(line.lower().split() for line in open('./data/vocab.txt'))
    #lda = models.ldamodel.LdaModel(corpus=AP_corpus, id2word=AP_dict, num_topics=num_topics, passes=10)
    #lda.show_topics()

    with open('./data/vocab.txt') as f:
        tfidf_dict = f.read().splitlines()    

    AP_docs = len(AP_tfidf)
    AP_vlen = len(tfidf_dict)
    AP_full = np.zeros((AP_docs, AP_vlen), dtype=np.float64)
    for row, doc in enumerate(AP_tfidf):
        doc_arr = np.array(doc)
        word_idx = doc_arr[:,0].astype(int)
        AP_full[row,word_idx] = doc_arr[:,1]
                
    A_tfidf_sp = sparse.csr_matrix(AP_full)    
        
    plt.spy(AP_full, cmap = 'gray')   
    plt.gca().set_aspect('auto')
    plt.title('AP tf-idf corpus')
    plt.xlabel('dictionary')
    plt.ylabel('documents')    
    plt.show()        
    del AP_full  #free up memory
    """
        
    #20 newsgroups (part of sklearn)
    print "loading 20 newsgroups dataset..."
    tic = time()
    dataset = fetch_20newsgroups(shuffle=True, random_state=0, remove=('headers','footers','quotes'))
    train_corpus = dataset.data  # a list of 11314 documents / entries
    toc = time()
    print "elapsed time: %.4f sec" %(toc - tic)    
    
    #compute tf-idf (equivalent to count-vectorizer followed by tf-idf transformer)
    #count-vectorizer produces term-document matrix tf-idf scales tf counts by log N/nt 
    #(N:num of docs, nt: number of a word occurence in docs)
    #if float (proportion of docs): min_df < nt/N  < max_df, if int: refers to count nt, e.g. min_df = 2
    tfidf = TfidfVectorizer(max_features = num_features, max_df=0.95, min_df=2, stop_words = 'english')
    print "tfidf parameters:"
    print tfidf.get_params()    
        
    #generate tf-idf term-document matrix
    A_tfidf_sp = tfidf.fit_transform(train_corpus)  #size D x V
    
    print "number of docs: %d" %A_tfidf_sp.shape[0]
    print "dictionary size: %d" %A_tfidf_sp.shape[1]

    #tf-idf dictionary    
    tfidf_dict = tfidf.get_feature_names()
             
    #fit LDA model
    print "Fitting LDA model..."
    lda_vb = LatentDirichletAllocation(n_topics = num_topics, max_iter=10, learning_method='online', batch_size = 512, random_state=0, n_jobs=-1)

    tic = time()
    lda_vb.fit(A_tfidf_sp)  #online VB
    toc = time()
    print "elapsed time: %.4f sec" %(toc - tic)
    print "LDA params"
    print lda_vb.get_params()

    print "number of EM iter: %d" % lda_vb.n_batch_iter_
    print "number of dataset sweeps: %d" % lda_vb.n_iter_

    #topic matrix W: K x V
    #components[i,j]: topic i, word j
    topics = lda_vb.components_
        
    f = plt.figure()
    plt.matshow(topics, cmap = 'gray')   
    plt.gca().set_aspect('auto')
    plt.title('learned topic matrix')
    plt.ylabel('topics')
    plt.xlabel('dictionary')
    plt.show()
    f.savefig('./figures/topic.png')
     
    #topic proportions matrix: D x K
    #note: np.sum(H, axis=1) is not 1
    H = lda_vb.transform(A_tfidf_sp)
    
    f = plt.figure()
    plt.matshow(H, cmap = 'gray')   
    plt.gca().set_aspect('auto')
    plt.show()
    plt.title('topic proportions')
    plt.xlabel('topics')
    plt.ylabel('documents')
    f.savefig('./figures/proportions.png')
                
    #compute perplexity
    print "perplexity: %.2f" % lda_vb.perplexity(A_tfidf_sp)    
    plot_perplexity_iter(A_tfidf_sp, num_topics)
    plot_perplexity_topics(A_tfidf_sp)
    plot_perplexity_batch(A_tfidf_sp, A_tfidf_sp.shape[0])

    print "LDA topics:"
    display_topics(lda_vb, tfidf_dict, 20)
            