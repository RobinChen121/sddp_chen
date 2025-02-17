from src import add_numbers
import src

def test_file():
    assert add_numbers(23, 45) == 68
    print(src.__doc__)
    import sys
    print(sys.path)