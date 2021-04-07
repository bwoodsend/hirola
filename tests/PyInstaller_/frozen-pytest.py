# -*- coding: utf-8 -*-
"""
Freeze pytest.main() with hirola included.
"""
import sys
import hirola

import pytest

pytest.main(sys.argv[1:] + ["--no-cov"])
