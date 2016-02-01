# -*- python -*-

"""OpenCV2 Nodes binding for OpenAlea.

These bindings offers additional doc to cv2 functions and/or ease the import as visualea nodes
"""

__docformat__ = "restructuredtext en"

import cv2
import numpy

#________________ core

def imread(imagefile, flag='CV_LOAD_IMAGE_UNCHANGED'):
    """Load an image from file into memory.

    :Parameters:
    -`imagefile` - The input image specified by an input image for example in PNG, TIF file format.
    - `flag` - Storage mode for the image. One of 'CV_LOAD_IMAGE_UNCHANGED' (default), 'CV_LOAD_IMAGE_COLOR', 'CV_LOAD_IMAGE_GRAYSCALE'
    :Returns:

    -`image` in memory, which can be further processed.
    
    :Examples:
    
    >>> import openalea.opencv.cv2_nodes as cvnodes
    >>> samplefile ='blob148561'
    >>> image = cvnode.imread(imagefile)

    :Supported file formats: The function imread currently supports the following file formats provided by the opencv2 function imread:

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

    flag = getattr(cv2, flag)
    return cv2.imread(imagefile, flags=flag)
    

# (CF) doc to be completed    
def resize(image, dsize=None, fx=0.5, fy=0.5, interpolation='INTER_LINEAR'): 
    """Resizes an image.
   
   fx,fy = scale across horizontal / vertical
   interpolation method:
   INTER_NEAREST - a nearest-neighbor interpolation
   INTER_LINEAR - a bilinear interpolation (used by default)
   INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
   INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
   INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood
    """
    interpolation = getattr(cv2,interpolation)
    return cv2.resize(image, dsize=dsize, fx=fx, fy=fy, interpolation=interpolation)
    
    
def putText(image, text='', org=(450,450), fontFace='FONT_HERSHEY_SIMPLEX', fontScale=2, color=(255,0,0), thickness=3):
    fontFace = getattr(cv2, fontFace)
    cv2.putText(image, text, org, fontFace, fontScale, color, thickness)
    return image

def cvtColor(image,flag='COLOR_BGR2GRAY'):
    """Converts an image from one color space to another
    """
    flag = getattr(cv2,flag)
    return cv2.cvtColor(image, flag)
    
#_________________________________Filtering


def bilateralFilter (image, d=5, sigmaColor=5, sigmaSpace=5, borderType='BORDER_DEFAULT'):
    ### Added to Wralea.py
    """The function applies a bilateral filter to an image. 
	    
    :Parameters:
    -`image` specifies the loaded image that should be filtered
    -`d` diameter of each pixel neighbourhood
    -`sigmaColor` 
    -`SigmaSpace`
    -`borderType` defines the pixel extrapolation method

    :Returns:
    -`filteredimage`

    :Notes:
    """    
    #ddepth = cv2.CV_16S
    bordertype = getattr(cv2,bordertype)
    filteredimage = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace, borderType=borderType)
    return filteredimage
    


def Laplacian (image, ddepth=-1, ksize=5, scale=1, delta=0, borderType='BORDER_DEFAULT'):
    ### Added to Wralea.py
    """The function calculates the Laplacian of an image, as implemented in OpenCV. 
	    
    :Parameters:
    -`image` specifies the loaded image that should be filtered
    -`ddepth` specifies depth of the output image. By default with ddepth=-1, the calculated destination image will have the same depth as the source
    -`ksize` defines the aperture size
    -`scale` is an optional scale factor (DEFAULT = 1)
    -`delta` is an optional delta value (DEFAULT = 0)
    -`borderType` defines the pixel extrapolation method

    :Returns:
    -`filteredimage`

    :Notes:
    """    
    #ddepth = cv2.CV_16S
    bordertype = getattr(cv2,bordertype)
    laplacian = cv2.Laplacian(image, ddepth, ksize=ksize, scale=scale, delta=delta, borderType=borderType)
    return laplacian
    

    
    
#___________________________________________image transform    
    
def distanceTransform (binaryimage):
    """Calculates from the sourceimage for every pixel of the source image the distant to the closest zero pixel using the OpenCV2 distance transform.
    """
    distancemap = numpy.int_ (cv2.distanceTransform(binaryimage, distanceType=cv2.cv.CV_DIST_L1, maskSize=5))
    return (distancemap)

# (CF) threshold types should be documented
def threshold(image, thresh=122, maxval=255, type='THRESH_BINARY',otsu = False):
    """applies fixed-level thresholding to a single-channel array
    """
    type = getattr(cv2,type)
    if otsu:
        type = type + cv2.THRESH_OTSU
        thresh = 0
    retval, binary = cv2.threshold(image, thresh, maxval, type=type)
    return binary	    
    
# (CF) Doc to be completed    
def adaptiveThreshold (grayimage, maxValue=255, adaptiveMethod='ADAPTIVE_THRESH_GAUSSIAN_C', thresholdType='THRESH_BINARY', blockSize=5, C=0):
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
    adaptiveMethod = getattr(cv2, adaptiveMethod)
    thresholdType = getattr(cv2, thresholdType)
    binaryimage = cv2.adaptiveThreshold(grayimage, maxValue, adaptiveMethod, thresholdType, blockSize, C)
    return (binaryimage)
    
    
#Feature detection
    
def cannyEdge (image, threshold1, treshold2, apertureSize=3, L2gradient="False"):
    """Calculates edges in an input image using the Canny86 algorithm provided by OpenCV.
    Input: Image
    Output: outputmap
    """
    outmap=cv2.Canny(image, treshold1, treshold2, apertureSize, L2gradient)
    return outmap

def cornerHarris (image, blockSize=3, ksize=3, k=0.04, borderType=cv2.BORDER_DEFAULT):
    """Calculates edges using the Harris edge detector provided by OpenCV2.
    Input: Image
    Output: Outputmap image in CV_32FC1
    """
    outmap = cv2.cornerHarris(image, blockSize, ksize, k)
    return outmap

	
def cornerEigenValsAndVecs(image, blockSize=3, ksize=5, borderType='BORDER_DEFAULT'):
    borderType = getattr(cv2, borderType)
    eigenvv = numpy.int_ (cv2.CornerEigenValsAndVecs(image, blockSize=blockSize, ksize=ksize, borderType=borderType))
    return (eigenvv)


def cornerSubPix (image, corners, searchWindow=(11,11), zeroZone=(-1,-1), criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
    """The function refines locations of corners previously identified. Iteratively the function tries to find the sub-pixel accurate location of all provided corners.

    :Parameters:
    -`image` specifies the loaded image, that should be processed.
    -`corners` defines the corners already identified in a previous step.
    -`searchWindow` defines the size (sidelength) of the SearchWindow, in which the sub-pixel accurate location of a corner should be investigated.
    -`zeroZone` defines deaed region in the middle of the SearchWindow, in which .
    
    :Returns:
    -`subpixelcorners`, returns the subpixel accurate locations o the found corners.
    
    :Notes:
    
    """

    img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.cornerSubPix(img, corners, searchWindow, zeroZone, criteria)
    return (corners)

#__________Camera Calibration

def findChessboardCorners(image, patternSize):
    #(found, corners) = numpy.int_ (cv2.findChessboardCorners(image, patternSize, flags=flags))    
    (found, corners) = cv2.findChessboardCorners(image, patternSize, flags=cv2.cv.CV_CALIB_CB_ADAPTIVE_THRESH)
    return (found, corners)
 
def drawChessboardCorners(image, patternSize, corners, patternWasFound):
    img = image.copy()
    cv2.drawChessboardCorners(img, patternSize, corners, patternWasFound)
    return img

   
#def calibrateCamera (objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags, criteria):    

    
def cornerSubPix_node(image, corners, searchWindow, zeroZone, criteria):
    return cornerSubPix(image, corners, searchWindow, zeroZone, criteria)
cornerSubPix_node.__doc__ = cornerSubPix.__doc__

def convertPointsToHomogeneous(image):
    """The function is part of the camera calibration library of OpenCV and converts points from Euclidean space to homogeneous space.
    The input tuple is thereby expanded by adding another dimension set to 1 for each point part of the tuple.
    
    """
    outputimage = cv2.convertPointsToHomogeneous(image)
    return outputimage
   
def convertPointsFromHomogeneous(image):
    """The function is part of the camera calibration library of OpenCV and converts points from homogeneous space to Euclidean space.
    Conversion is done by perspective projection as described in the OpenCV documentation.
    
    """
    outputimage = cv2.convertPointsFromHomogeneous(image)
    return outputimage

def convertPointsHomogeneous(image):
    """The function is part of the camera calibration library of OpenCV and converts points from and to homogenous coordinates. 
    See also functions convertPointsToHomogenous and convertPointsToHomogenous.
    """
    outputimage = cv2.convertPointsHomogeneous(image)
    return outputimage   

## VideoAnalysis Need to be added to wralea:


def calcOpticalFlowPyrLK (prevImg, nextImg, prevPts, winSize, maxLevel, criteria, flags, minEigThreshold):
    """
    """
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, winSize, maxLevel, criteria, flags, minEigThreshold)
    return (nextPts, status, err)

def buildOpticalFlowPyramid (img, winSize, maxLevel):
    """
    """
    retval, pyramid = cv2.buildOpticalFlowPyramid(img, winSize, maxLevel)
    return (retval, pyramid)


# Photographic repair/denoising


def inpaint (image, mask, flags=cv2.INPAINT_NS):
    ### Added to Wralea.py
    """The function can be used to restore the image content in a provided region of interest using the neighborhood to this region.

    :Parameters:
    -`image` specifies the loaded image that should be processsed. The image can be either a 8-Bit 1- or 3-channel image.
    -`mask` specifies a mask by a provided 1-channel 8-Bit image, in which non-zero pixels define the regions of interest, which should be restored.
    -`flags`

    :Returns:
    -`restoredimage`, which is the image with the restored pixels.

    :Notes:
    """
    restoredimage = cv2.inpaint(image, mask, flags)
    return restoredimage


def fastNlMeansDenoising(image, h=float(3), templateWindowSize=7, searchWindowSize=21):
    """The function performs denoising using the Non-local Means Denoising algorithm provided by the OpenCV2 library.
    
    :Parameters:
    :Returns:
    :Notes:
    """
    denoisedimage= numpy.int_ (cv2.fastNlMeansDenoising(image, h, templateWindowSize, searchWindowSize))
    return (denoisedimage)

def fastNlMeansDenoisingColored(image, h=float(3), hColor=float(3), templateWindowSize=7, searchWindowSize=21):
    """The function performs denoising using the Non-local Means Denoising algorithm provided by the OpenCV2 library.
    
    :Parameters:
    :Returns:
    :Notes:
    """
    denoisedimage= numpy.int_ (cv2.fastNlMeansDenoisingColored(image, h, hColor, templateWindowSize, searchWindowSize))
    return (denoisedimage)    
    
    
