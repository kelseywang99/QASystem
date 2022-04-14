import nltk
import sys
import os
import string
import math

nltk.download('stopwords')

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():
    files = load_files('corpus')
    # # Check command-line arguments
    # if len(sys.argv) != 2:
    #     sys.exit("Usage: python questions.py corpus")

    # # Calculate IDF values across files
    # files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    # iterate through all files in the directory
    for filename in os.listdir(directory):
        # open and read the file, save the value as string
        with open(os.path.join(directory, filename)) as f:
            files[filename] = f.read()
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # use nltk’s word_tokenize to perform tokenization, change to lower_case
    # filter out stopwords and punctuation words
    contents = [
        word.lower() for word in
        nltk.word_tokenize(document)
        if (word not in string.punctuation and
            word not in nltk.corpus.stopwords.words("english"))
    ]
    return contents


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # create a set of all relevant words appear in the documents
    words = set()
    for filename in documents:
        words.update(documents[filename])

    # create a dict to track idfs for words
    idfs = dict()
    for word in words:
        # calculate the number of documents in which the word appears
        f = sum(word in documents[filename] for filename in documents)
        # idf=log(num of documents/ num of documents in which the word appears)
        idf = math.log(len(documents) / f)
        # update the idfs dict
        idfs[word] = idf
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    idfs = compute_idfs(files)

    # create a dict to track word frequency for all files
    freq = dict()
    for filename in files:
        # create a dict to track word frequency in this file
        frequencies = dict()
        # iterate throguh words in the file to calculate word frequency
        for word in files[filename]:
            # up date the word frequency for this word in this file
            if word not in frequencies:
                frequencies[word] = 1
            else:
                frequencies[word] += 1
        # update the word frequency for this file
        freq[filename] = frequencies

    # create a dic to track tfidfs for each file
    tfidfs = dict()
    # iterate through each file
    for filename in files:
        # sum of tfidf of every word in query and in this file
        tfidfs[filename] = sum([freq[filename][word] * idfs[word]
                                for word in query if word in files[filename]])
    # sort tfidf dict by values (tfidf),
    # get a list of (filename, tfidf) items
    tfidfs = sorted(tfidfs.items(), key=lambda tfidf: tfidf[1], reverse=True)
    # choose the top n best-match items
    tfidfs = tfidfs[:n]
    # return the list of filenames
    return [x[0] for x in tfidfs]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # create a dict to track idf for sentences
    sentence_idfs = dict()
    # iterate through each sentence
    for sentence in sentences:
        # sum of idf of every word in query and in this sentence
        sentence_idf = sum(idfs[word] for word in query
                           if word in sentences[sentence])
        # proportion of words in the sentence that are also words in the query
        term_density = sum([word in query for word in sentences[sentence]]) \
            / len(sentences[sentence])
        # add (sentence_idf, term_density) to the dictionary
        sentence_idfs[sentence] = (sentence_idf, term_density)

    # sort the dict first by sentence_idf then by term_density,
    # get a list of (sentence, (sentence_idf, term_density)) items
    sentence_idfs = sorted(sentence_idfs.items(),
                           key=lambda sentence_idf:
                               (sentence_idf[1][0], sentence_idf[1][1]),
                               reverse=True)
    # choose the top n best-match items
    sentence_idfs = sentence_idfs[:n]
    # return the list of sentences
    return [x[0] for x in sentence_idfs]


if __name__ == "__main__":
    main()
