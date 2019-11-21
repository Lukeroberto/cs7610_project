import argparse

from src.utils.dqn import *
from src.utils.graph_utils import *
from src.utils.plotting_utils import *
from src.utils.parser import *

def main():

    p = test_parser()
    args = p.parse_args()

    test_dict[args.test]()




def test_1a():
    raise NotImplementedError

def test_1b():
    raise NotImplementedError

def test_1c():
    raise NotImplementedError

def test_2a():
    raise NotImplementedError

def test_2b():
    raise NotImplementedError

def test_3a():
    raise NotImplementedError

def test_3b():
    raise NotImplementedError

test_dict = {
    "1a": test_1a,
    "1b": test_1b,
    "1c": test_1c,
    "2a": test_2a,
    "2b": test_2b,
    "3a": test_3a,
    "3b": test_3b,
}

if __name__ == "__main__":
    main()