# -*- coding: utf-8 -*-

NESTED_CV_PARAMETERS = {
    'C': [0.1, 1, 10, 100],
    'class_weight': [
        'balanced',
        None,
        {0: 0.01, 1: 10},
        {0: 0.6, 1: 10},
        {0: 0.01, 1: 200},
        {0: 0.6, 1: 200}
    ],
}
