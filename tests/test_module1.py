from msp.module1 import add_numbers
import msp

def test_file():
    assert add_numbers(23, 45) == 68
    print(msp.__doc__)
    import sys
    print(sys.path)