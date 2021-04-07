# -*- coding: utf-8 -*-
"""
Freeze pytest.main() with hoatzin included.
"""
import sys
import hoatzin

import pytest

pytest.main(sys.argv[1:] + ["--no-cov"])
