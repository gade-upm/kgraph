#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from kgraph.management import execute_from_command_line

if __name__ == '__main__':
    execute_from_command_line()
