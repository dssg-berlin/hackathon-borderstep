# -*- coding: utf-8 -*-
r"""
Test for utility functions in word count
========================================

Verify good behavior
"""
# Created: Sat Apr 21 16:21:11 2018
# Author: Óscar Nájera
# License: GPL-3

from __future__ import division, absolute_import, print_function

import re
from word_count import *


def test_keyword_filter():
    string = 'Windkraft* , Windenergie*'
    assert keyword_filter(string) == {"desired_keywords": [
        'Windkraft\\w*', 'Windenergie\\w*']}

    negation = 'Solar* , NICHT Solarium*, Solarstudio*, Solarien*'
    assert keyword_filter(negation) == {'desired_keywords': ['Solar\\w*'],
                                        'avoid_keywords': ['Solarium\\w*',
                                                           'Solarstudio\\w*',
                                                           'Solarien\\w*']}


def test_join():
    a = {'desired_keywords': list(range(4))}
    b = {'desired_keywords': list(range(6, 10)),
         'avoid_keywords': list(range(20, 25))}

    assert join_keywords(b, a) == {'desired_keywords': [6, 7, 8, 9, 0, 1, 2, 3],
                                   'avoid_keywords': [20, 21, 22, 23, 24]}
