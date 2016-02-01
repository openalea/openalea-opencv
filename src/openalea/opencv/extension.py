# -*- python -*-
#
#       Copyright 2015 INRIA - CIRAD - INRA
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       OpenAlea WebSite : http://openalea.gforge.inria.fr
#
# ==============================================================================

"""
Opencv based extension functions for additional functionality / syntactic sugars
"""

__docformat__ = "restructuredtext en"

# ==============================================================================
import cv2
import numpy
# ==============================================================================


# core
def imreads(image_path_seq, flag='IMREAD_UNCHANGED', real_reading=False):
    """ returns a generator of images and the number of image for an image path
     list. flag is the option flag for imread
    """
    flag = getattr(cv2, flag)
    l = len(image_path_seq)
    if real_reading:
        img = list()
        for f in image_path_seq:
            img.append(cv2.imread(f, flag))
        g = img.__iter__()
    else:
        g = (cv2.imread(f, flag) for f in image_path_seq)
    return g, l


def image_size(image):
    """Return the width, height, number of chanel and number of pixels of an image
    """
    width, height = image.shape[:2]
    channel_number = 1
    if len(image.shape) > 2:
        channel_number = image.shape[2]
    pixels = image.size / channel_number
    return width, height, channel_number, pixels


def cv_mean(image_iterator, n_images):
    """ Compute the element-wise mean over an image iterator containing n_images
    """
    w = 1. / n_images
    start = cv2.addWeighted(image_iterator.next(), w, image_iterator.next(), w,
                            0)
    return reduce(lambda x, y: cv2.addWeighted(x, 1, y, w, 0), image_iterator,
                  start)


def cv_mean2(image_iterator, n_images):
    """ Compute the element-wise mean over an image iterator containing n_images
    """
    if n_images > 1000:
        return None
    img = image_iterator.next()
    img_sum = numpy.empty(img.shape, 'int32')
    img_sum = reduce(lambda x, y: cv2.add(x, numpy.int32(y)), image_iterator,
                     cv2.add(img_sum, numpy.int32(img)))
    return numpy.uint8(numpy.round(numpy.divide(img_sum, n_images)))


def CreateImage(width=100, height=100, channel=1, dtype='uint8'):
    """ Create an empty image (default grayscale)
    """

    dtype = getattr(numpy, dtype)
    if channel <= 1:
        image = numpy.zeros((width, height), dtype)
    else:
        image = numpy.zeros((width, height, channel), dtype)
    return image


def crop(image, y1=0, y2=10, x1=0, x2=10, mask=None):
    """Crops an image by the defined region of interest using OpenCV2 functions.

    :Parameters:
    - `image` specifies a loaded image that should be cropped.
    - `y1` specifies the first y coordinate (upper corners)
    - `y2` specifies the second y coordinate (lower corners, y2 > y1 since
        origin is top left)
    - `x1` specifies the first x coordinate (left corners).
    - `x2` specifies the second x coordinate(right corners, x2 > x1 since origin
        is top left).
    - `mask`  specifies an optional mask to be used for cropping.

    :Returns:
    -`image`, which is the original uncropped image.
    -`cropped`, which is the cropped image created from the given coordinates
     or mask.
    - 'crop_info', which keep track of cropping to allow later uncrop
    """
    if mask is None:
        cropped = image[y1:y2, x1:x2]
    else:
        cropped = mask

    w, h, c, p = image_size(image)
    crop_info = {'w': w, 'h': h, 'y1': y1, 'y2': y2, 'x1': x1, 'x2': x2}
    return cropped, crop_info


def uncrop(cropped, crop_info):
    """ Undo image croping made by crop
    """
    w, h, c, p = image_size(cropped)
    img = CreateImage(crop_info['w'], crop_info['h'], c, str(cropped.dtype))
    if c <= 1:
        img[crop_info['y1']:crop_info['y2'],
            crop_info['x1']:crop_info['x2']] = cropped
    else:
        img[crop_info['y1']:crop_info['y2'],
            crop_info['x1']:crop_info['x2'],
            :] = cropped
    return img


def channel(img, c=0):
    """ return a channel slice of a muti-chanel image
    """
    return img[:, :, c]


# ==============================================================================
# Filtering

def erode(image, kshape='MORPH_CROSS', ksize=3, iterations=1):
    """Erodes an image
    """
    kshape = getattr(cv2, kshape)
    element = cv2.getStructuringElement(kshape, (ksize, ksize))
    eroded = cv2.erode(image, element, iterations=iterations)
    return eroded


def dilate(image, kshape='MORPH_CROSS', ksize=3, iterations=1):
    """Dilates an image
    """
    kshape = getattr(cv2, kshape)
    element = cv2.getStructuringElement(kshape, (ksize, ksize))
    dilated = cv2.dilate(image, element, iterations=iterations)
    return dilated


def open(image, kshape='MORPH_CROSS', ksize=3, iterations=1):
    """Performs image openning
    """
    kshape = getattr(cv2, kshape)
    element = cv2.getStructuringElement(kshape, (ksize, ksize))
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element,
                              iterations=iterations)
    return opened


def close(image, kshape='MORPH_CROSS', ksize=3, iterations=1):
    """Performs image closure
    """
    kshape = getattr(cv2, kshape)
    element = cv2.getStructuringElement(kshape, (ksize, ksize))
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, element,
                              iterations=iterations)
    return closed


def morphological_skeleton(binary_image):
    """Calculates a skeleton from a provided binary image using erosion/dilation
    Input: Binary image
    Output: skeleton image
    """
    size = numpy.size(binary_image)
    skel = numpy.zeros(binary_image.shape, numpy.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(binary_image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary_image, temp)
        skel = cv2.bitwise_or(skel, temp)
        binary_image = eroded.copy()

        zeros = size - cv2.countNonZero(binary_image)
        if zeros == size:
            done = True

    return skel


# (CF) for Sharr/Sobel, docstring/output names seems outdated
def Scharr(image, ddepth=-1, dx=1, dy=0, scale=1, delta=0,
           borderType=cv2.BORDER_DEFAULT):
    ### Added to Wralea.py
    """The function calculates the first x- and y- image derivative using the
     Scharr-Operator as implemented in OpenCV.

     The Scharr Operator works comparable to the Sobel function,
     but is generally believed to provide results with improved accuracy for
      a kernel of size 3x3
    
    :Parameters:
    -`image` specifies the loaded image that should be filtered
    -`ddepth` specifies depth of the output image. By default with ddepth=-1,
    the calculated destination image will have the same depth as the source
    -`dx` is the order of the derivative x (Not used in OpenAlea)
    -`dy` is the order of the derivative y (Not used in OpenAlea)
    -`scale` is an optional scale factor (DEFAULT = 1)
    -`delta` is an optional delta value (DEFAULT = 0)
    -`borderType` defines the pixel extrapolation method

    :Returns:
    -`weightedxygradient` a composite image in which x and y gradients are
    shown with equal weighting.
    -`absolutexgradient` a gradient image only representing the x directional
     gradients converted back to UINT8
    -`absoluteygradient` a gradient image only representing the y directional
     gradients converted back to UINT8

    :Notes:
    """
    # ddepth = cv2.CV_16S
    xgradient = numpy.int_(
        cv2.Scharr(image, ddepth, 0, 1, scale=scale, delta=delta,
                   borderType=borderType))
    ygradient = numpy.int_(
        cv2.Scharr(image, ddepth, 1, 0, scale=scale, delta=delta,
                   borderType=borderType))
    absolutexgradient = numpy.int_(
        cv2.convertScaleAbs(xgradient))  # Optional conversion to UINT8
    absoluteygradient = numpy.int_(
        cv2.convertScaleAbs(ygradient))  # Optional conversion to UINT8
    weightedxygradient = numpy.int_(
        cv2.addWeighted(absolutexgradient, 0.5, absoluteygradient, 0.5, 0))

    return (xgradient,
            ygradient,
            weightedxygradient,
            absolutexgradient,
            absoluteygradient)


def Sobel(image, ddepth=-1, dx=1, dy=0, ksize=5, scale=1, delta=0,
          borderType=cv2.BORDER_DEFAULT):
    """The function calculates the first, second or third image derivative using
     the Sobel operator as implemented in OpenCV.

     :Parameters:
     -`image` specifies the loaded image that should be filtered
    -`ddepth` specifies depth of the output image. By default with ddepth=-1,
    the calculated destination image will have the same depth as the source
    -`dx` is the order of the derivative x (Not used in OpenAlea)
    -`dy` is the order of the derivative y (Not used in OpenAlea)
    -`ksize` defines the size of the extended Sobel Kernel. It can be 1,3,5 or 7
    -`scale` is an optional scale factor (DEFAULT = 1)
    -`delta` is an optional delta value (DEFAULT = 0)
    -`borderType` defines the pixel extrapolation method

    :Returns:
    -`weightedxygradient` a composite image in which x and y gradients are
    shown with equal weighting.
    -`absolutexgradient` a gradient image only representing the x directional
     gradients converted back to UINT8
    -`absoluteygradient` a gradient image only representing the y directional
     gradients converted back to UINT8

    :Notes:
    """
    # ddepth = cv2.CV_16S
    xgradient = numpy.int_(
        cv2.Sobel(image, ddepth, 0, 1, ksize=ksize, scale=scale, delta=delta,
                  borderType=borderType))
    ygradient = numpy.int_(
        cv2.Sobel(image, ddepth, 1, 0, ksize=ksize, scale=scale, delta=delta,
                  borderType=borderType))
    absolutexgradient = numpy.int_(
        cv2.convertScaleAbs(xgradient))  # Optional conversion to UINT8
    absoluteygradient = numpy.int_(
        cv2.convertScaleAbs(ygradient))  # Optional conversion to UINT8
    weightedxygradient = numpy.int_(
        cv2.addWeighted(absolutexgradient, 0.5, absoluteygradient, 0.5, 0))

    return (xgradient,
            ygradient,
            weightedxygradient,
            absolutexgradient,
            absoluteygradient)


# ==============================================================================
# Image transform

def maskimage(image, mask):
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked


# ==============================================================================
# Contour / structural analysis

# (CF) I modified a little the outputs.
def findContours(binaryimage):
    """ Find coutours (object) in the image

    returns contour list, hierarchy, number of contour found and bigest contour
    """
    copied = binaryimage.copy()
    contours, hierarchy = cv2.findContours(copied, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # select the bigest
    id_max = numpy.argmax(map(cv2.contourArea, contours))
    return contours, hierarchy, len(contours), contours[id_max]


def ContoursProperties(contours):
    """ Compute descritive properties of a contour list, as presented in opencv
    python tutorial
    
    Properties returned: 
    centroid, area and perimeter are self-explanatory
    aspect_ratio is the ratio of width to height of bounding rect of the object
    extent  is the ratio of contour area to bounding rectangle area.
    Solidity is the ratio of contour area to its convex hull area.
    Equivalent Diameter is the diameter of the circle whose area is same as the
    contour area. ellipse properties are the Orientation , the Major Axis and
    Minor Axis lengths are the one of the enclosing ellipse
    Extreme Points means topmost, bottommost, rightmost and leftmost points
    of the object.
    """

    def _centroid(cnt):
        M = cv2.moments(cnt)
        area = M['m00']
        if area == 0:
            cx, cy = cnt.mean(axis=0)[0]
        else:
            cx = M['m10'] / area
            cy = M['m01'] / area

        return int(cx), int(cy)

    def _perimeter(cnt):
        return cv2.arcLength(cnt, True)

    def _aspect_ratio(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        return float(w) / h

    def _extent(cnt):
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        res = 0
        if rect_area > 0:
            res = float(area) / rect_area
        return res

    def _solidity(cnt):
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        res = 0
        if hull_area > 0:
            res = float(area) / hull_area
        return res

    def _equivalent_diameter(cnt):
        area = cv2.contourArea(cnt)
        return numpy.sqrt(4 * area / numpy.pi)

    def _ellipse_properties(cnt):
        angle, MA, ma = 0, 0, 0
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)

        except:
            pass
        return angle, MA, ma

    def _extreme_points(cnt):
        leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

        return leftmost, rightmost, topmost, bottommost

    properties = {'contour_id': range(len(contours)),
                  'centroid': map(_centroid, contours),
                  'area': map(cv2.contourArea, contours),
                  'perimeter': map(_perimeter, contours),
                  'aspect_ratio': map(_aspect_ratio, contours),
                  'extent': map(_extent, contours),
                  'solidity': map(_solidity, contours),
                  'diameter_eq': map(_equivalent_diameter, contours),
                  'ellipse': map(_ellipse_properties, contours),
                  'extreme_points': map(_extreme_points, contours)}

    return properties


def colorImageAs(image):
    w, h, c, p = image_size(image)
    if c <= 1:
        dst = CreateImage(w, h, 3)
        for i in range(3):
            dst[:, :, i] = numpy.int_(image)
    else:
        dst = image.copy()
    return dst


# (CF) may be add the fill option ?    
def drawContours(image, contours, contourIdx=-1, colour=(0, 255, 0), line=2):
    """
    """
    dst = colorImageAs(image)
    cv2.drawContours(dst, contours, contourIdx, colour, line)
    return dst


# (CF) new temptative node for fiting/drawing a generalist contour-fited shape.
def fitShape(contour, fit='convexHull'):
    """ fit a shape into a contour
    
    shape is one of convexHull,boundingRect, minAreaReect, fitEllipse or
    minEnclosingCircle
    """
    fun = getattr(cv2, fit)
    shape = fun(contour)
    return shape, fit


def drawShape(image, shape, fit, color=(0, 255, 0), line=2):
    """ draw a shape fitted with fitShape
    """
    img = colorImageAs(image)
    if fit == 'convexHull':
        cv2.drawContours(img, [shape], 0, color, line)
    elif fit == 'boundingRect':
        x, y, w, h = shape
        cv2.rectangle(img, (x, y), (x + w, y + h), color, line)
    elif fit == 'minAreaRect':
        box = cv2.cv.BoxPoints(shape)
        box = numpy.int(box)
        cv2.drawContours(img, [box], 0, color, line)
    elif fit == 'fitEllipse':
        cv2.ellipse(img, shape, color, line)
    elif fit == 'minEnclosingCircle':
        (x, y), radius = shape
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(img, center, radius, color, line)
    else:
        pass
    return img


# (NB) Simplification of convexhullallcontours
def convexhullimage(contours):
    hull = []
    cont = numpy.vstack(contours[i] for i in range(len(contours)))
    hull.append(cv2.convexHull(cont))
    return cont, hull


# (CF) this (nice !) extension needs doc string and probably a better name
def convexhullallcontours(contours):
    number = len(contours)
    status = numpy.zeros((number, 1))

    for i, cnt1 in enumerate(contours):
        x = i
        if i != number - 1:
            for j, cnt2 in enumerate(contours[i + 1:]):
                x += 1
                dist = arecontoursclose(cnt1, cnt2, distancethreshold=150)
                dist = True
                if dist is True:
                    val = min(status[i], status[x])
                    status[x] = status[i] = val
                    # ~ else:
                    # ~ print ("Do nothing")

    maximum = int(status.max()) + 1

    cont = None
    unified = []
    unified2 = []
    for i in xrange(maximum):
        pos = numpy.where(status == i)[0]
        if pos.size != 0:
            cont = numpy.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            hullindices = cv2.convexHull(cont, returnPoints=False)
            unified.append(hull)
            defects = cv2.convexityDefects(cont, hullindices)
            unified2.append(defects)

    return cont, unified, unified2


def arecontoursclose(contour1, contour2, distancethreshold=150):
    """The functions tests, if two provided contours are close to each other,
    being closer than the defined maximum threshold distance.
    
    :Parameters:
    :Returns: 
    """
    row1, row2 = contour1.shape[0], contour2.shape[0]
    for i in xrange(row1):
        for j in xrange(row2):
            distance = numpy.linalg.norm(contour1[i] - contour2[j])
            if abs(distance) < distancethreshold:
                return True
            elif i == row1 - 1 and j == row2 - 1:
                return False

                # def convexityDefects (contour, hull):

# defects = cv2.convexityDefects(contour, hull)
#    return (defects)


# ==============================================================================
# Histogram functions

def grayscalehistogram(images, mask, histSize, ranges):
    ### Added to Wralea.py
    """The function calculates a histogram of a povided gray scale image.

    :Parameters:
    -`images` specifies a single or multiple gray scale images
    -`mask` allows you to provide a mask, which defines the area of interest,
    that should be used for the histogram calculation.
    -`histSize`
    -`ranges`

    :Returns:
    -`histogram`

    :Notes:
    """
    histogram = cv2.calcHist(images, [0], mask, [256], [0, 256])
    return histogram

# ==============================================================================
# Color transform syntactic sugars

def rgb2hsv(image):
    """Converts an image in memory stored in RGB color space to be transformed
     into HSV color space using OpenCV2 function cvtColor.

    :Parameters:
    -`image` specifies the loaded image, that should be transformed:
    Can be 8-Bit unsigned, 16-Bit unsigned or single-precision floating-point.
        
    :Returns:
    -`image_hsv`, which is the transformed image in hsv color space.
    
    :Notes:
    The function rgb2hsv assumes, that you provide an image in RGB channel
    order. OpenCV2 normally uses the BGR channel order. In this case use the
    bgr2hsv function instead.
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return image_hsv


def bgr2hsv(image):
    """Converts an image in memory stored in BGR color space to be transformed
     into HSV color space using OpenCV2 function cvtColor.

    :Parameters:
    -`image` specifies the loaded image, that should be transformed:
    Can be 8-Bit unsigned, 16-Bit unsigned or single-precision floating-point.

    :Returns:
    -`image_hsv`, which is the transformed image in hsv color space.
    
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image_hsv


def bgr2luv(image):
    """Converts an image in memory stored in BGR color space to be transformed
     into LUV color space using OpenCV2 function cvtColor.

    :Parameters:
    -`image` specifies the loaded image, that should be transformed:
    Can be 8-Bit unsigned, 16-Bit unsigned or single-precision floating-point.
    
    :Returns:
    -`image_luv`, which is the transformed image in LUV color space.
    
    :Notes:
    The function bgr2luv assumes, that you provide an image in BGR channel
    order. If the channels are provided in RGB byte channel order, use the
    Phenomenal_opencv2.rgb2luv function instead.
    """
    image_luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    return image_luv


def bgr2rgb(image):  # OK added to wralea.py
    """Converts an image in memory stored in BGR color space to be transformed
     into RGB color space using OpenCV2 function cvtColor.

    :Parameters:
    -`image` specifies the loaded image, that should be transformed:
    Can be 8-Bit unsigned, 16-Bit unsigned or single-precision floating-point.
    
    :Returns:
    -`image_rgb`, which is the transformed image in RGB color space.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def rgb2luv(image, dstCn=0):  # OK added to wralea.py
    """Converts an image in memory stored in RGB color space to be transformed
     into LUV color space using OpenCV2 function cvtColor.

    :Parameters:
    -`image` specifies the loaded image, that should be transformed: Can be
     8-Bit unsigned, 16-Bit unsigned or single-precision floating-point.
    
    -`dstCn` defines the number of channels in the created image. By default
     (dstCn=0) the number of channels is automatically derived from the
      provided input image.
    
    :Returns:
    -`image_luv`, which is the transformed image in LUV color space.
    
    :Notes:
    The function rgb2luv assumes, that you provide an image in RGB channel
    order. If the channels are provided in BGR byte channel order, use the
    Phenomenal_opencv2.bgr2luv function instead.
    """
    image_luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    return image_luv


def luv2bgr(image):  # Ok added to wralea.py
    """Converts an image in memory stored in LUV color space to be transformed
    into BGR color space using OpenCV2 function cvtColor.

    :Parameters:
    -`image` specifies the loaded image, that should be transformed: Can be
     8-Bit unsigned, 16-Bit unsigned or single-precision floating-point.
    
    :Returns:
    -`image_bgr`, which is the transformed image in BGR color space.
    
    :Notes:
    The function luv2bgr assumes, that you want to create an image in the
    penCV2 common BGR channel order. If the channels be created in the RGB
    byte channel order, use the Phenomenal_opencv2.luv2rgb function instead.

    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_LUV2BGR)
    return image_bgr


def luv2rgb(image):  # OK added to wralea.py
    """Converts an image in memory stored in LUV color space to be transformed
     into RGB color space using OpenCV2 function cvtColor.

    :Parameters:
    -`image` specifies the loaded image, that should be transformed: Can be
     8-Bit unsigned, 16-Bit unsigned or single-precision floating-point.
    
 
    :Returns:
    -`image_rgb`, which is the transformed image in RGB color space.
    
    :Notes:
    The function luv2bgr assumes, that you want to create an image in the RGB
     channel order. If the channels be created in the BGR byte channel order
    common in OpenCV2 instead, use the Phenomenal_opencv2.luv2bgr function
    instead.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_LUV2RGB)
    return image_rgb


def bgr2lab(image):  # OK added to wralea.py
    """Converts an image in memory stored in BGR color space to be transformed
     into CIE L*A*B color space using OpenCV2 function cvtColor.

    :Parameters:
    -`image` specifies the loaded image, that should be transformed:
    Can be 8-Bit unsigned, 16-Bit unsigned or single-precision floating-point.
    
  
    :Returns:
    -`image_lab`, which is the transformed image in CIE L*A*B color space.
    
    :Notes:
    The function bgr2lab assumes, that you want to provide an image in the
    OpenCV2 common BGR channel order. If the channels should be provided in
    the RGB byte channel order instead, use the Phenomenal_opencv2.rgb2lab
    function instead.
    """
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return image_lab


def rgb2lab(image):  # OK added to wralea.py
    """Converts an image in memory stored in RGB color space to be transformed
    into CIE L*A*B color space using OpenCV2 function cvtColor.

    :Parameters:
    -`image` specifies the loaded image, that should be transformed: Can be
    8-Bit unsigned, 16-Bit unsigned or single-precision floating-point.
    
  
    :Returns:
    -`image_lab`, which is the transformed image in CIE L*A*B color space.
    
    :Notes:
    The function rgb2lab assumes, that you want to provide an image in the RGB
    channel order. If the channels should be provided in the BGR byte channel
    order commonly used in OpenCV2, use the Phenomenal_opencv2.bgr2lab
    function instead.
    """
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    return image_lab


def lab2bgr(image):  # OK added to wralea.py
    """Converts an image in memory stored in CIE L*A*B color space to be
    transformed into BGR color space using OpenCV2 function cvtColor.

    :Parameters:
    -`image` specifies the loaded image, that should be transformed:
    Can be 8-Bit unsigned, 16-Bit unsigned or single-precision floating-point.
    
   
    :Returns:
    -`image_bgr`, which is the transformed image in BGR color space.
    
    :Notes:
    The function lab2bgr assumes, that you want to create an image in the BGR
    channel order, which is commonly used by OpenCV2. If the channels should be
    created in the RGB byte channel order, use the Phenomenal_opencv2.lab2rgb
    function instead.
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    return image_bgr


def lab2rgb(image):  # OK added to wralea.py
    """Converts an image in memory stored in CIE L*A*B color space to be
    transformed into RGB color space using OpenCV2 function cvtColor.

    :Parameters:
    -`image` specifies the loaded image, that should be transformed:
    Can be 8-Bit unsigned, 16-Bit unsigned or single-precision floating-point.
    
   
    :Returns:
    -`image_rgb`, which is the transformed image in RGB color space.
    
    :Notes:
    The function lab2rgb assumes, that you want to create an image in the RGB
    channel order. If the channels should be created in the BGR byte channel
    order, which is commonly used in OpenCV2, use the
    Phenomenal_opencv2.lab2bgr function instead.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    return image_rgb


def bgr2gray(image):
    """Converts an image in memory stored in BGR color space to be transformed
    into gray value image using the OpenCV2 function cvtColor.

    :Parameters:
    -`image` specifies the loaded image, that should be transformed: Can be
    8-Bit unsigned, 16-Bit unsigned or single-precision floating-point.
    
    :Returns:
    -`gray_image`, which is the transformed image in gray values.
    
    :Notes:
    The function bgr2gray assumes, that you want to provide an image in the BGR
    channel order, which is commonly used in OpenCV2. If the channels should be
    provided in the RGB byte channel order, use the
    Phenomenal_opencv2.rgb2gray function instead.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def rgb2gray(image):
    """Converts an image in memory stored in RGB color space to be transformed
    into gray value image using the OpenCV2 function cvtColor.

    :Parameters:
    -`image` specifies the loaded image, that should be transformed: Can be
    8-Bit unsigned, 16-Bit unsigned or single-precision floating-point.
    
    :Returns:
    -`gray_image`, which is the transformed image in gray values.
    
    :Notes:
    The function rgb2gray assumes, that you want to provide an image in the RGB
    channel order. If the channels should be provided in the BGR byte channel
    order, which is commonly used in OpenCV2, use the
    Phenomenal_opencv2.bgr2gray function instead.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image
