import sys

from jiarinard.jia_new import jia_binary, jia_new

if __name__ == '__main__':
    if sys.argv[1] == 'binary':
        jia_binary(int(sys.argv[2]))
    else:
        jia_new(int(sys.argv[2]))
