#!/usr/bin/env python2.7

import unittest
from dnaseqlib import *

### Utility classes ###

# Maps integer keys to a set of arbitrary values.
class Multidict:
    # Initializes a new multi-value dictionary, and adds any key-value
    # 2-tuples in the iterable sequence pairs to the data structure.
    def __init__(self, pairs=[]):
        self.Multidict = dict()
        for pair in pairs:
            if pair[0] in self.Multidict.keys():
                self.Multidict[pair[0]].append(pair[1])
            else:
                self.Multidict[pair[0]] = [pair[1]]
    # Associates the value v with the key k.
    def put(self, k, v):
        if k in self.Multidict.keys():
            self.Multidict[k].append(v)
        else:
            self.Multidict[k] = [v]
    # Gets any values that have been associated with the key k; or, if
    # none have been, returns an empty sequence.
    def get(self, k):
        if k not in self.Multidict.keys():
            return []
        else:
            return self.Multidict[k]

# Given a sequence of nucleotides, return all k-length subsequences
# and their hashes.  (What else do you need to know about each
# subsequence?)
def subsequenceHashes(seq, k):
    str = "" 
    offset = 0
    for char in range(k):
        str += next(seq)
    rollingHash_str = RollingHash(str)

    yield str, rollingHash_str.current_hash, offset

    for char in seq:
        current_hash = rollingHash_str.slide(str[0], char)
        str = str[1:] + char
        offset += 1
        yield str, current_hash, offset



# Similar to subsequenceHashes(), but returns one k-length subsequence
# every m nucleotides.  (This will be useful when you try to use two
# whole data files.)
def intervalSubsequenceHashes(seq, k, m):
    str = ""
    offset = 1 - m

    for char in seq:
        str += char
        if offset % m == 0:
            str_hash = RollingHash(str[:k]).current_hash()
            yield str[:k], str_hash, offset + m - 1
            str = ""
        offset += 1

# Searches for commonalities between sequences a and b by comparing
# subsequences of length k.  The sequences a and b should be iterators
# that return nucleotides.  The table is built by computing one hash
# every m nucleotides (for m >= k).
def getExactSubmatches(a, b, k, m):
    dict = Multidict()
    for str_a, hash_a, offset_a in intervalSubsequenceHashes(a, k, m):
        dict.put(hash_a, (str_a, offset_a))
    for str_b, hash_b, offset_b in subsequenceHashes(b, k):
        for item in dict.get(hash_b):
            if item[0] == str_b:
                yield item[1], offset_b

    

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {0} [file_a.fa] [file_b.fa] [output.png]'".format(sys.argv[0]))
        sys.exit(1)

    # The arguments are, in order: 1) Your getExactSubmatches
    # function, 2) the filename to which the image should be written,
    # 3) a tuple giving the width and height of the image, 4) the
    # filename of sequence A, 5) the filename of sequence B, 6) k, the
    # subsequence size, and 7) m, the sampling interval for sequence
    # A.
    compareSequences(getExactSubmatches, sys.argv[3], (500,500), sys.argv[1], sys.argv[2], 8, 100)
