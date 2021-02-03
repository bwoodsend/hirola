# -*- coding: utf-8 -*-
"""
"""
import numpy as np


def random_ids(max, count, at_least_once=True, sort=False):
    if at_least_once:
        assert count >= max
        out = np.append(np.arange(max, dtype=np.intp),
                        np.random.randint(0, max, count - max, np.intp))
    else:
        out = np.random.randint(0, max, count, np.intp)
    if sort:
        out.sort()
    else:
        np.random.shuffle(out)
    return out
