# the interview question by T"M"D

def segmentation(sentence, dictionary):

    """
    :type sentence: string
    :type dictionary: list
    :     note: dictionary is used to split terms from the sentence
    :rtype: list
    """

    wordlist=[]
    forward=0

    # Define segementation words
    while forward < len(sentence):
        for i in range(len(dictionary)):
            temp = sentence[forward: forward+i]
            if temp in dictionary:
                wordlist.append(temp)
                forward += i

    # Count frequency
    wordfreq = [wordlist.count(w) for w in wordlist]]
    maxFreq = max(wordfreq)

    return( [wordlist[wordfreq.index(maxFreq)],  maxFreq])


