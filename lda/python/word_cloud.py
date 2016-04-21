#!/usr/bin/env python2
"""
Masked wordcloud
================
Using a mask you can generate wordclouds in arbitrary shapes.
"""

from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

import pdb

if __name__ == "__main__":

    d = path.dirname(__file__)

    # read in text
    # text = open(path.join(d, 'top_words.txt')).read()
    file_name = './data/top_words.txt'
    with open(file_name) as f:
        text = f.readlines()

    # read the mask image
    word_mask = np.array(Image.open(path.join(d, "./figures/circle_mask2.png")))

    # construct wordcloud
    wc = WordCloud(background_color="white", max_words=100, mask=word_mask,\
                stopwords=STOPWORDS.add("and"))

    print "generating word cloud ..."
    for topic_idx in range(len(text)):
        # generate word cloud
        wc.generate(text[topic_idx])
        # store to file
        wc.to_file(path.join(d, "./figures/topic"+str(topic_idx)+".png"))
    #end

    # generate plots 
    plt.figure()
    plt.imshow(wc)
    plt.axis("off")
    plt.show()

    #plt.figure()
    #plt.imshow(word_mask, cmap=plt.cm.gray)
    #plt.axis("off")
    #plt.show()
    
