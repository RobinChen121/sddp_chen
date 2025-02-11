from msp.module1 import add_numbers
import msp

def test_file():
    assert add_numbers(23, 45) == 40
    print(msp.__doc__)