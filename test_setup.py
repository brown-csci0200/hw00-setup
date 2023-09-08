import pytest

import copy
import math
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import random
import re
import requests
import sys
import xml.etree.ElementTree as ET

# Install stopwords with SSL verification disabled
# This should work even if Python's extra certificates are not installed
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

'''
If you can run the tests using the testing tab of VSCode and they pass, you are set up to use
Pytest and almost all of the packages necessary for CS 200
'''

def test_copy():
    deep_struct = [{'a': [1, 2]}, {'b': [3]}]
    deep_struct_2 = copy.deepcopy(deep_struct)
    deep_struct_2[1]['b'].append(4)
    assert len(deep_struct[1]['b']) == 1
    assert len(deep_struct_2[1]['b']) == 2

def test_math():
    assert math.log(math.exp(3)) == pytest.approx(3)

def test_nltk():
    nltk_ps = PorterStemmer()
    assert nltk_ps.stem("doing") == "do"
    assert 'the' in set(stopwords.words('english'))

def test_numpy():
    test_ones = np.ones(3)
    assert test_ones[2] == 1

def test_random():
    test_seq = [1, 2, 3]
    while test_seq[0] == 1:
        random.shuffle(test_seq)
    assert len(test_seq) == 3
    assert 1 in test_seq

def test_re():
    assert re.match("[a-z]+", "abcdef") is not None
    assert re.match("[a-z]+", "123") is None

def test_requests():
    r = requests.get('https://api.github.com/events')
    assert r.encoding != ''

def test_sys():
    a = 'Test String'
    assert sys.getrefcount(a) > 0

def test_xml():
    root = ET.fromstring('<d><p><t>one</t><c>c1</c></p><p><t>two</t><c>c2</c></p></d>')
    pages = root.findall("p")
    assert pages[1].find("c").text == "c2"
