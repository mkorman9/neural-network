import struct
import sys

MAX_SAMPLES = 10000

def parse(path):
    with open(path) as f:
        magic = f.read(4)
        if struct.unpack('>I', magic)[0] != 2049:
            sys.stderr.write('Invalid magic\n')
            return

        number_of_items = min(struct.unpack('>I', f.read(4))[0], MAX_SAMPLES)

        for item in xrange(number_of_items):
            label = struct.unpack('>B', f.read(1))[0]
            sys.stdout.write('{0} '.format(label))
            print('')

if __name__ == '__main__':
    parse(sys.argv[1])
