# -*- coding: utf-8 -*-
"""
Freeze pytest.main() with hirola included.
"""
import sys
import hirola

import pytest

sys.exit(pytest.main(sys.argv[1:] + ["--no-cov", "--tb=native"]))
