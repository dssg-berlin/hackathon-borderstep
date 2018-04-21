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


def test_keyword_filter():
    string = 'Windkraft* , Windenergie*'
    assert keyword_filter(string) == {"desired_keywords": [
        'Windkraft\\w*', 'Windenergie\\w*']}

    negation = 'Solar* , NICHT Solarium*, Solarstudio*, Solarien*'
    assert keyword_filter(negation) == {'desired_keywords': ['Solar\\w*'],
                                        'avoid_keywords': ['Solarium\\w*',
                                                           'Solarstudio\\w*',
                                                           'Solarien\\w*']}


test = 'about solarenegy solarium'
r = re.compile('|'.join(r'\b%s\b' % w for w in [
               'Solar\\w*', '(?!solarium)']), re.I)
re.findall(r, test)
