import struct
import sys

MAX_SAMPLES = 5000

def parse(path):
    with open(path) as f:
        magic = f.read(4)
        if struct.unpack('>I', magic)[0] != 2051:
            sys.stderr.write('Invalid magic\n')
            return

        number_of_images = min(struct.unpack('>I', f.read(4))[0], MAX_SAMPLES)
        rows = struct.unpack('>I', f.read(4))[0]
        columns = struct.unpack('>I', f.read(4))[0]

        for image in xrange(number_of_images):
            for row in xrange(rows):
                for column in xrange(columns):
                    pixel = float(struct.unpack('>B', f.read(1))[0]) / 255.0
                    sys.stdout.write('{0} '.format(pixel))
            print('')


if __name__ == '__main__':
    parse(sys.argv[1])