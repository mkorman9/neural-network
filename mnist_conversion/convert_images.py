import struct
import sys
import numpy
import skimage.feature

MAX_SAMPLES = 60000

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
            image_pixels = []
            for column in xrange(columns):
                row_pixels = []
                for row in xrange(rows):
                    pixel = struct.unpack('>B', f.read(1))[0]
                    row_pixels.append(pixel)
                image_pixels.append(row_pixels)
            image_data = numpy.array(image_pixels, dtype=numpy.uint8)
            print(' '.join([str(x) for x in skimage.feature.hog(image_data).tolist()]))


if __name__ == '__main__':
    parse(sys.argv[1])