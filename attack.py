import sys

from linearverifier.attack.jia_new import jia_binary, jia_new

if __name__ == '__main__':
    if sys.argv[1] == 'binary':
        jia_binary()
    else:
        jia_new()
