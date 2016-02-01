""" Code coming from Phenomenal_opencv2 module, and candidate to deprecation /deletion
"""

import cv2
import numpy

# (CF) I don't think rexiff reading is to be included : its an independant package

#import pyexiv2 as exif

def readexif(imagefile):
    """Read exif information from an image"""
    metadata = exif.ImageMetadata(imagefile)
    metadata.read()
    #metadata.exif_keys # show all exif keys present in image
    #metadata.iptc_keys # show all iptc keys present in image
    #metadata.xmp_keys # show all xmp keys present in image
    ImageWidthPointer = metadata['Exif.Image.ImageWidth']
    ImageWidth = ImageWidthPointer.raw_value
    ImageLengthPointer = metadata['Exif.Image.ImageLength']
    ImageLength = ImageLengthPointer.raw_value
    BitsPerSamplePointer = metadata['Exif.Image.BitsPerSample']
    BitsPerSample = BitsPerSamplePointer.raw_value
    PhotometricPointer = metadata['Exif.Image.PhotometricInterpretation']
    PhotometricInterpretation = PhotometricPointer.raw_value
    MakePointer = metadata['Exif.Image.Make']
    CameraManufacturer = MakePointer.raw_value    
    model = metadata['Exif.Image.Model']
    cameramodel = model.raw_value
    OrientationPointer = metadata['Exif.Image.Orientation']
    orientation = OrientationPointer.raw_value
    SamplesperpixelPointer = metadata['Exif.Image.SamplesPerPixel']
    samplesperpixel = SamplesperpixelPointer.raw_value
    xresolutionpointer = metadata['Exif.Image.XResolution']
    xresolution = xresolutionpointer.raw_value
    yresolutionpointer = metadata['Exif.Image.YResolution']
    yresolution = yresolutionpointer.raw_value
    resolutionunitPointer = metadata['Exif.Image.ResolutionUnit']
    resolutionunit = resolutionunitPointer.raw_value
    exposure = metadata['Exif.Photo.ExposureTime'] #Load memory address for Exposuretime key
    exposuretime = exposure.raw_value # Access rawdata from exposure memory address  
    flashsettings = metadata['Exif.Photo.Flash']
    flash = flashsettings.raw_value
    #print ("Image width:" +ImageWidth)
    #print ("Image length:" +ImageLength)
    #print ("BitsPerSample:" +BitsPerSample)
    #print ("PhotometricInterpretation:" +PhotometricInterpretation)
    #print ("Camera manufacturer:" +CameraManufacturer)
    #print ("Camera model:" +cameramodel)
    #print ("Exposure time:" +exposuretime)
    #print ("Flash setting:" +flash)
    
    return ImageWidth #(ImageWidth, ImageLength, BitsPerSample, cameramodel, exposuretime, flash)
 
# (CF) I don't think we need a logging module for the opencv package : to me it's rather related to Phenomenal command line interface 
# consequently I remove all printing message from function (I can restore if needed !)

import os
#import pyexiv2 as exif
import logging

def logger_init(outdir):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # create error file handler and set level to error
    handler = logging.FileHandler(os.path.join(outdir, "phenomenal-error.log"),"w", encoding=None, delay="true")
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter("%(asctime)s %(name) %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # create debug file handler and set level to debug
    handler = logging.FileHandler(os.path.join(outdir, "phenomenal-debug.log"),"w")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
  

# (CF) I propose not to incluse skimage skeleton in the package, as it is not opencv. Let's keep it in Phenomenal  
# I have included a simple morphological skeleton based on opencv erosion/dilatation, which is not equivalent (not as good)
# Meanwhile, there exist a c++/python extension for an equivaleent skeletton in opencv: https://github.com/bsdnoobz/zhang-suen-thinning. To be included ?  

import skimage
def skeleton(binaryimage):
    """Calculates a skeleton from a provided binary image skimage morphological thining
    Input: Binary image
    Output: skeleton image
    """

    return numpy.int_(skimage.morphology.skeletonize(binaryimage > 0))


# (CF) These variables were not used in the module
  
RED = [0,0,255]
GREEN = [0,255,0]
BLUE = [255,0,0]
BLACK = [0,0,0]
WHITE = [255,255,255]

#CV_LOAD_IMAGE_UNCHANGED = -1 # flag to load 8 bit, color or gray images. Its DEPRECATED in favour of CV_LOAD_IMAGE_ANYCOLOR
CV_LOAD_IMAGE_GRAYSCALE =  0 # flag to load a 8 bit gray image
CV_LOAD_IMAGE_COLOR     =  1 # flag to load a 8 bit color image 5combine zith andepth for 16 Bit ?)
CV_LOAD_IMAGE_ANYDEPTH  =  2 # flag to load a 8 bit color image in any depth, 
                             # equivalent to CV_LOAD_IMAGE_UNCHANGED but can be modified
                             # with CV_LOAD_IMAGE_ANYDEPTH
CV_LOAD_IMAGE_ANYCOLOR  =  4
# CV_LOAD_IMAGE_ALPHA = <0  #Define a value smaller zero to read an image including the alpha channel.

# (CF) this class is not (and not intended) to be used isn't it ? (Current solution to calls to cv2 options is to make an IEnumStr interface in __wralea__ files and to call getattr(cv2,opt) in python code
    
class cv2Options:
    def __init__(self):
        self.default_border = numpy.int_(cv2.BORDER_DEFAULT)
        #imread opts
        self.__dict__.update({'LOAD_IMAGE_COLOR': cv2.CV_LOAD_IMAGE_COLOR,
                              'LOAD_IMAGE_GRAYSCALE': cv2.CV_LOAD_IMAGE_GRAYSCALE,
                              'LOAD_IMAGE_UNCHANGED':cv2.CV_LOAD_IMAGE_UNCHANGED})
        self.__THRESH_BINARY = numpy.int_(cv2.THRESH_BINARY)
        self.__THRESH_BINARY_INV = numpy.int_(cv2.THRESH_BINARY_INV)
        self.__THRESH_BINARY_INV = numpy.int_(cv2.THRESH_BINARY_INV)
        self.__THRESH_MASK = numpy.int_(cv2.THRESH_MASK)
        self.__THRESH_OTSU = numpy.int_(cv2.THRESH_OTSU)
        self.__THRESH_TOZERO = numpy.int_(cv2.THRESH_TOZERO)
        self.__THRESH_TOZERO_INV = numpy.int_(cv2.THRESH_TOZERO_INV)
        self.__THRESH_TRUNC = numpy.int_(cv2.THRESH_TRUNC)
        self.__FONT_HERSHEY_SIMPLEX = numpy.int_(cv2.FONT_HERSHEY_SIMPLEX)
        self.__FONT_HERSHEY_PLAIN = numpy.int_(cv2.FONT_HERSHEY_PLAIN)
    def get(self,optstring): 
        return getattr(self,optstring)

	

# (CF) This function is a uselees (logging excepted) duplicates: it only calls imread, hat is elewhere in the package.

def loadimage(imagefile, flags=cv2.CV_LOAD_IMAGE_COLOR):
    #Function tested: Yes
    #Documentation completed: Yes
    #Documentation layout complete: Yes
    #Bugs in Layout: Yes: Link rendering does not work
    
    """Wrapping to load an image from file into memory using OpenCV2 function imread.

    :Parameters:
    -`imagefile` - The input image specified by an input image for example in PNG, TIF file format.

    :Returns:

    -`image` in memory, which can be further processed.
    
    :Examples:
    
    >>> import Phenomenal_opencv2 as pcv2
    >>> samplefile ='/data2/pgftp/ARCH2013-05-13/2013-06-06/blob148561'
    >>> destination = tempfile.mkdtemp()
    >>> imagefile = os.path.join(destination,os.path.basename(samplefile))
    >>> image = pcv2.loadimage(imagefile)

    :Supported file formats: The function loadimage currently supports the following file formats provided by the opencv2 function imread:

              - Windows bitmaps: *.bmp, *.dib
              - JPEG image files: *.jpeg, *.jpg, *.jpe
              - JPEG 2000 image files: *.jp2
              - PNG (Portable Networks Graphics) image files: *.png
              - Image files in the Portable image formats: *.pbm, *.pgm, *.ppm
              - Sun rasters: *.sr, *.ras
              - TIFF image files: *.tiff, *.tif

    The type of image will be always determined by the image content (provided through the file and image headers), and not by the given file extension of the image file.

    :Note:
    - Color images are always stored in BGR order of channels, if loaded with the loadimage into memory.
    For further information see the OpenCV documentation at -U{OpenCVimread <http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html>}
    """
    image = cv2.imread(imagefile) #cv2.Loadimage does not exits in cv2
    logging.info("Loaded image to memory") #print "Loaded image to memory!"
    return (image)
    
# (CF) I replace this with cv2.countNonZero wich is faster and compatible with more inputs format (0/1, 0/255, ...)    
    
def area(binaryimage):
    """Calculates the number of Pixels as area of the analysed binaryimage using OpenCV2 functions.

    :Parameters:
    -`binaryimage` specifies the loaded binary image, that should be used to calculate the area of all white pixels in this binary image.
        
    :Returns:
    -`area`, which is the number of all white pixels of the input image. 
    
    :Notes:
    The function assumes, that all binary images are provided as arrays with values being 0 or 255.
    """
    area=binaryimage.sum()/255
    return (area)

# (CF) I replace this with inRange, that does the same job, under its standard generalist name
    
def hsvseg(imagehsv, hsv_min=(0,0,0), hsv_max=(255,255,223)):
    #Function tested: Yes
    #Documentation completed: Yes
    #Documentation layout complete: Yes
    """Uses a HSV treshold algorithm from OpenCV2 to analyse an image in HSV color space and returns a binary image using OpenCV2 functions.

    :Parameters:
    -`imagehsv` specifies the loaded image, that should be segmented.
    -`hsv_min` defines the lower boundaries for the hsv segmentation treshold levels for hue (H), saturation (S) and value (V) as an numpy array in the form [0,0,0]
    -`hsv_max` defines the lower boundaries for the hsv segmentation treshold levels for hue (H), saturation (S) and value (V) as an numpy array in the form [0,0,0]

    :Returns:
    -`binaryimage`, is returned as a binary mask, produced from tresholding.
    
    :Notes:
    
    """    
    hsv_min = numpy.array(hsv_min, numpy.uint8)
    hsv_max = numpy.array(hsv_max, numpy.uint8)
    binaryimage = cv2.inRange(imagehsv, hsv_min, hsv_max)
    logging.info("Binary Image has been calculated from tresholding image in HSV color space.") #print "Binary Image has been calculated from tresholding image in HSV color space."

    #cv2.imwrite('binaryimage.png',binaryimage)

    return (binaryimage)
   

# (CF) This function is buggy (and known to be so) under ipython/Visuale (ie in interactive mode). Suggest to say that Pylab Imshow doea the job and we do not provide interface to open_cv gui

def showimageinwindow(image, windowname="Phenomenal", flags=numpy.int_(cv2.CV_WINDOW_AUTOSIZE)):
    #Function tested: Yes
    #Documentation completed: Yes
    #Documentation layout complete: Yes
    """The function creates a named Window and displays an image in it using the OpenCV2 imshow function.
    The opened window waits for a key to be pressed and is then closed.
    
    :Parameters:
    -`image` specifies the loaded image, which should be displayed.
    -`windowname` defines the name of the window, which is shown in the window caption. The name of the window can be used as a window identifier internally in the function.
        
    :Returns:
    
    :Notes:    
    """
    
    cv2.namedWindow(windowname, cv2.CV_WINDOW_AUTOSIZE)
    cv2.imshow(windowname, image)
    cv2.waitKey()
    cv2.destroyWindow(windowname)
    #The window has to be created for using this function in Windows (Windows 7 issue?)

    
# (CF) same remark as for cv2.imshow

def waitKey (delay=0):
    """
    """
    cv2.waitKey(delay)
   
# (CF) redundant with numpy /python add node. I import cv.add instead, which doeas saturated arithmetics and thus differentiates from simple add
def imageaddition (image1, image2):
    """Addition of two images using the numpy function. (It would be also possible to use cv2.add)
    Note: Images should be of the same depth, size and type at best
    Input: image1, image2
    Output: image
    """
    image= image1 + image2
    return image

#CF redundant with numpy /python minus node. 
def imagesubtraction (image1, image2):
    """Subtraction of two images using the numpy function.
    Note: Images should be of the same depth, size and type at best
    Input: image1, image2
    Output: image
    """
    image= image1 - image2
    return image
  
 # (CF) I replace this with CreateImage that does both rgb/grayscale and let the user choose dtype
def createemptyimage(width, height):
    #note for bugfixing. Some very small number 0 and 1 for example do not seem to work
    image = numpy.zeros((width, height), numpy.uint8)
    return image
    
# (CF) I replace this with CreateImage that does both rgb/grayscale    
def createemptyimagergb(width, height):
    #note for bugfixing. Some very small number 0 and 1 for example do not seem to work
    image = numpy.zeros((width, height, 3), numpy.uint8)
    return image  
   
# (CF) I replace this with uncrop
def copycroptoemptyimage (crop, emptyimage, y1, y2, x1, x2):
    empty = emptyimage.copy()
    empty[y1:y2, x1:x2,:] = crop
    return empty
    
# (CF) I replace this with uncrop
def copycroptoemptyimagebinary (crop, emptyimage, y1, y2, x1, x2):
    empty = emptyimage.copy()
    empty[y1:y2, x1:x2] = crop
    return empty     

# (CF) I replace this with channel
def splithsvtohuegray (imagehsv):
    gray = imagehsv[:,:,0]
    return gray    
    
# (CF) I merge this with bilateralFilter
def bilateralFilter_node(image, d, sigmaColor, sigmaSpace, borderType):
    btype = eval('.'.join(('cv2',borderType))) #Note: The borderType is defined in the wralea script
    print btype
    return bilateralFilter(image, d=5, sigmaColor=int(5), sigmaSpace=int(5), borderType=numpy.int_(btype))
#bilateralFilter_node.__doc__ = bilateralFilter.__doc__

# (CF) I merge this with Laplacian
def Laplacian_node(image, borderType):
    btype = eval('.'.join(('cv2',borderType))) #Note: The borderType is defined in the wralea script
    #print btype
    return Laplacian(image, ddepth=-1, ksize=5, scale=1, delta=0, borderType=numpy.int_(btype))
#Laplacian_node.__doc__ = Laplacian.__doc__

# (CF) I merge this with Scharr
def Scharr_node(image, dx, dy, borderType):
    btype = eval('.'.join(('cv2',borderType))) #Note: The borderType is defined in the wralea script
    #print btype
    return Scharr(image, ddepth=-1, dx=1, dy=0, scale=1, delta=0, borderType=numpy.int_(btype))
#Scharr_node.__doc__ = Scharr.__doc__

# (CF) I merge this with Sobel
def Sobel_node(image, dx, dy, borderType):
    btype = eval('.'.join(('cv2',borderType))) #Note: The borderType is defined in the wralea script
    return Sobel(image, ddepth=-1, dx=1, dy=0, ksize=5, scale=1, delta=0, borderType=numpy.int_(btype))
#Sobel_node.__doc__ = Sobel.__doc__


# (CF) I merge this with cornerEigenValsAndVecs
def cornerEigenValsAndVecs_node(image, dx, dy, borderType):
    btype = eval('.'.join(('cv2',borderType))) #Note: The borderType is defined in the wralea script
    return cornerEigenValsAndVecs(image, blockSize=-1, ksize=5, borderType=numpy.int_(btype))
#cornerEigenValsAndVecs_node.__doc__ = cornerEigenValsAndVecs.__doc__


# (CF) I merge this with findChessboardCorners
def findChessboardCorners_node(image, patternSize, use_adaptive):
    if use_adaptive:
        return findChessboardCorners(image, patternSize, cv2.cv.CV_CALIB_CB_ADAPTIVE_THRESH)
    else:
        return findChessboardCorners(image, patternSize)
#findChessboardCorners_node.__doc__ = findChessboardCorners.__doc__

# (CF) I merge this with drawChessboardCorners
def drawChessboardCorners_node(image, patternSize, corners, patternWasFound):
    return drawChessboardCorners(image, patternSize, corners, patternWasFound)
#drawChessboardCorners_node.__doc__ = drawChessboardCorners.__doc__

# (CF) replaced by adaptativeThreshold
def adaptiveThresholdMean (grayimage, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, inversebinary="False", blockSize=5, C=0):
    ### Added to Wralea.py including an integer bug 
    """The function uses an adaptive Tresholding mechanism provided by the OpenCV2 library. The function thereby transforms a grayscale image to a binary image.

    :Parameters:
    -`grayimage` specifies the loaded image, that should be segmented. The source image has to be provided as a 8-Bit single channel image.
    -`maxValue`
    -`adaptiveMethod`
    -`thresholdType` specifies if an binary or the inverse of the binary image should be created. The thresholdType therefore must be either cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV.

    
    :Returns:
    -`binaryimage`, is returned as a binary mask, produced from tresholding.
    
    :Notes:
    """
    binaryimage = cv2.adaptiveThreshold(grayimage, maxValue, adaptiveMethod, inversebinary, blockSize, C)
    return (binaryimage)

# (CF) replaced by drawShape
def boundingbox (binary, contours):
    x,y,w,h = cv2.boundingRect(contours)
    boundingbox = numpy.int_ (cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2))
    return boundingbox
    
def rotatedboundingbox (binary, contours):    
    rect = numpy.int_ (cv2.minAreaRect(contours))
    box = numpy.int_ (cv2.boxPoints(rect))
    return box, rect
    
     
    

# (CF) The following nodes were not really deprecated, I only removed them from the python module as they mostly consist of a direct call to cv2 functions, 
# and kept them in the __wralea___ using cv2 as nodeclass and node 'desc' as a small doc indication

    
def cv_add(image1, image2):
    """ addition of two images using saturation arithmetics
    """
    return cv2.add(image1,image2)

def cv_addWeighted(image1, w1, image2, w2, offset=0):
    """ return a blended image : w1 * image1 + w2 * image2 + offset
    """
    return cv2.addWeighted(image1, w1, image2, w2, offset)
  
def absdiff (image1, image2):
    difference = cv2.absdiff(image1, image2)
    return difference

def imwrite(imagefile, image):
    cv2.imwrite(imagefile, image)
    return imagefile

def medianBlur (image, ksize=5):
    """
    """
    blured = cv2.medianBlur(image, ksize)
    return (blured)
 
def fitEllipse(binaryimage):
    """
    """
    fitellipse = numpy.int_(cv2.fitEllipse(binaryimage))
    return (fitellipse)
    
def minAreaRect(binaryimage):
    """Calculate the rotated rectangle of the minimum area that encloses the input set of point (for example in a binary image).
    """
    minarearect = cv2.minAreaRect(binaryimage)
    return (minarearect)
    
def contourArea (cnt, oriented="False"):
    """
    """
    cntarea=cv2.contourArea(cnt, oriented)
    return (cntarea)
    
def convexHull(listofpoints):
    """Calculates the convexhull of an provided binary using opencv2 functions.
    Input: List of points
    Output: Image of the convexhull
    """
    
    convexhull = cv2.convexHull(listofpoints) 
    return (convexhull)
