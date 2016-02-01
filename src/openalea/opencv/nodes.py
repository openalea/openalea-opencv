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
OpenCV2 Nodes binding for OpenAlea.

These bindings offers additional doc to cv2 functions and/or ease the import
as visualea nodes
"""
# ==============================================================================
__docformat__ = "restructuredtext en"

import cv2
import numpy
# ==============================================================================
# Core


def imread(imagefile, flag='CV_LOAD_IMAGE_UNCHANGED'):
    """Load an image from file into memory.

    :Parameters:
    -`imagefile` - The input image specified by an input image for example in
    PNG, TIF file format.

    - `flag` - Storage mode for the image. One of 'CV_LOAD_IMAGE_UNCHANGED'
    (default), 'CV_LOAD_IMAGE_COLOR', 'CV_LOAD_IMAGE_GRAYSCALE'

    :Returns:
    -`image` in memory, which can be further processed.
    
    :Examples:
    
    >>> from openalea.opencv.nodes import imread
    >>> sample_file ='blob148561.png'
    >>> image = imread(sample_file)

    :Supported file formats: The function imread currently supports the
    following file formats provided by the opencv2 function imread:

              - Windows bitmaps: *.bmp, *.dib
              - JPEG image files: *.jpeg, *.jpg, *.jpe
              - JPEG 2000 image files: *.jp2
              - PNG (Portable Networks Graphics) image files: *.png
              - Image files in the Portable image formats: *.pbm, *.pgm, *.ppm
              - Sun rasters: *.sr, *.ras
              - TIFF image files: *.tiff, *.tif

    The type of image will be always determined by the image content
    (provided through the file and image headers), and not by the given file
    extension of the image file.

    :Note:
    - Color images are always stored in BGR order of channels, if loaded with
    the loadimage into memory.

    For further information see the OpenCV documentation at -U {OpenCVimread
    <http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html>}
    """

    flag = getattr(cv2, flag)
    return cv2.imread(imagefile, flags=flag)


# ==============================================================================
# (CF) doc to be completed
def resize(image, dsize=None, fx=0.5, fy=0.5, interpolation='INTER_LINEAR'):
    """Resizes an image.
   
   fx,fy = scale across horizontal / vertical
   interpolation method:
   INTER_NEAREST - a nearest-neighbor interpolation
   INTER_LINEAR - a bilinear interpolation (used by default)
   INTER_AREA - resampling using pixel area relation. It may be a preferred
   method for image decimation, as it gives moire-free results.
   But when the image is zoomed, it is similar to the INTER_NEAREST method.
   INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
   INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood
    """
    interpolation = getattr(cv2, interpolation)
    return cv2.resize(image, dsize=dsize, fx=fx, fy=fy,
                      interpolation=interpolation)


def putText(image, text='', org=(450, 450), font_face='FONT_HERSHEY_SIMPLEX',
            font_scale=2, color=(255, 0, 0), thickness=3):
    font_face = getattr(cv2, font_face)
    cv2.putText(image, text, org, font_face, font_scale, color, thickness)
    return image


def cvtColor(image, flag='COLOR_BGR2GRAY'):
    """Converts an image from one color space to another
    """
    flag = getattr(cv2, flag)
    return cv2.cvtColor(image, flag)

# ==============================================================================
# Filtering


def bilateralFilter(image, d=5, sigma_color=5, sigma_space=5,
                    border_type='BORDER_DEFAULT'):
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
    # ddepth = cv2.CV_16S
    border_type = getattr(cv2, border_type)
    return cv2.bilateralFilter(
        image, d, sigma_color, sigma_space, borderType=border_type)


def Laplacian(image, ddepth=-1, k_size=5, scale=1, delta=0,
              border_type='BORDER_DEFAULT'):
    """
    The function calculates the Laplacian of an image, as implemented in OpenCV.

    :Parameters:
    -`image` specifies the loaded image that should be filtered
    -`ddepth` specifies depth of the output image. By default with ddepth=-1,
    the calculated destination image will have the same depth as the source
    -`ksize` defines the aperture size
    -`scale` is an optional scale factor (DEFAULT = 1)
    -`delta` is an optional delta value (DEFAULT = 0)
    -`borderType` defines the pixel extrapolation method

    :Returns:
    -`filteredimage`

    :Notes:
    """
    # ddepth = cv2.CV_16S
    border_type = getattr(cv2, border_type)
    return cv2.Laplacian(image,
                         ddepth,
                         ksize=k_size,
                         scale=scale,
                         delta=delta,
                         borderType=border_type)


# ==============================================================================
# Image transform
def distanceTransform(binary_image):
    """Calculates from the sourceimage for every pixel of the source image the
     distant to the closest zero pixel using the OpenCV2 distance transform.
    """
    return numpy.int_(cv2.distanceTransform(
        binary_image, distanceType=cv2.cv.CV_DIST_L1, maskSize=5))


# ==============================================================================
# (CF) threshold types should be documented
def threshold(image,
              thresh=122,
              max_val=255,
              thresh_type='THRESH_BINARY',
              otsu=False):
    """Applies fixed-level thresholding to a single-channel array
    """
    thresh_type = getattr(cv2, thresh_type)
    if otsu:
        thresh_type = thresh_type + cv2.THRESH_OTSU
        thresh = 0
    retval, binary = cv2.threshold(image, thresh, max_val, type=thresh_type)
    return binary

# ==============================================================================
# (CF) Doc to be completed    
def adaptiveThreshold(gray_image,
                      max_value=255,
                      adaptive_method='ADAPTIVE_THRESH_GAUSSIAN_C',
                      threshold_type='THRESH_BINARY',
                      block_size=5,
                      c=0):
    """
    The function uses an adaptive Tresholding mechanism provided by the OpenCV2
    library. The function thereby transforms a grayscale image to a binary
    image.

    :Parameters:
    -`grayimage` specifies the loaded image, that should be segmented.
    The source image has to be provided as a 8-Bit single channel image.
    -`maxValue`
    -`adaptiveMethod`
    -`thresholdType` specifies if an binary or the inverse of the binary image
    should be created. The thresholdType therefore must be either
    cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV.

    
    :Returns:
    -`binaryimage`, is returned as a binary mask, produced from tresholding.
    
    :Notes:
    """
    adaptive_method = getattr(cv2, adaptive_method)
    threshold_type = getattr(cv2, threshold_type)
    return cv2.adaptiveThreshold(
        gray_image, max_value, adaptive_method, threshold_type, block_size, c)


# Feature detection

def cannyEdge(image,
              threshold_1,
              threshold_2,
              aperture_size=3,
              l2_gradient="False"):
    """Calculates edges in an input image using the Canny86 algorithm provided
    by OpenCV.

    Input: Image
    Output: outputmap
    """
    return cv2.Canny(
        image, threshold_1, threshold_2, aperture_size, l2_gradient)


def cornerHarris(image,
                 block_size=3,
                 k_size=3,
                 k=0.04,
                 borderType=cv2.BORDER_DEFAULT):
    """Calculates edges using the Harris edge detector provided by OpenCV2.
    Input: Image
    Output: Outputmap image in CV_32FC1
    """
    return cv2.cornerHarris(image, block_size, k_size, k)


def cornerEigenValsAndVecs(image,
                           block_size=3,
                           k_size=5,
                           border_type='BORDER_DEFAULT'):

    border_type = getattr(cv2, border_type)
    return numpy.int_(cv2.CornerEigenValsAndVecs(
        image, blockSize=block_size, ksize=k_size, borderType=border_type))


def cornerSubPix(image,
                 corners,
                 search_window=(11, 11),
                 zero_zone=(-1, -1),
                 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                           30,
                           0.001)):
    """The function refines locations of corners previously identified.
    Iteratively the function tries to find the sub-pixel accurate location of
    all provided corners.

    :Parameters:
    -`image` specifies the loaded image, that should be processed.
    -`corners` defines the corners already identified in a previous step.
    -`searchWindow` defines the size (sidelength) of the SearchWindow, in which
     the sub-pixel accurate location of a corner should be investigated.

    -`zeroZone` defines deaed region in the middle of the SearchWindow, in
    which.
    
    :Returns:
    -`subpixelcorners`, returns the subpixel accurate locations o the found
     corners.
    
    :Notes:
    """

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.cornerSubPix(img, corners, search_window, zero_zone, criteria)
    return corners

# ==============================================================================
# Camera Calibration


def findChessboardCorners(image, pattern_size):
    return cv2.findChessboardCorners(
        image, pattern_size, flags=cv2.cv.CV_CALIB_CB_ADAPTIVE_THRESH)


def drawChessboardCorners(image, pattern_size, corners, pattern_was_found):
    img = image.copy()
    cv2.drawChessboardCorners(img, pattern_size, corners, pattern_was_found)
    return img


def cornerSubPix_node(image, corners, search_window, zero_zone, criteria):
    return cornerSubPix(image, corners, search_window, zero_zone, criteria)


cornerSubPix_node.__doc__ = cornerSubPix.__doc__


def convertPointsToHomogeneous(image):
    """The function is part of the camera calibration library of OpenCV and
     converts points from Euclidean space to homogeneous space.

    The input tuple is thereby expanded by adding another dimension set to 1
    for each point part of the tuple.
    """
    return cv2.convertPointsToHomogeneous(image)


def convertPointsFromHomogeneous(image):
    """The function is part of the camera calibration library of OpenCV and
    converts points from homogeneous space to Euclidean space.

    Conversion is done by perspective projection as described in the OpenCV
    documentation.
    """
    return cv2.convertPointsFromHomogeneous(image)


def convertPointsHomogeneous(image):
    """The function is part of the camera calibration library of OpenCV and
     converts points from and to homogenous coordinates.

    See also functions convertPointsToHomogenous and convertPointsToHomogenous.
    """
    return cv2.convertPointsHomogeneous(image)

# ==============================================================================
## VideoAnalysis Need to be added to wralea:


def calcOpticalFlowPyrLK(prev_img, next_img, prev_pts, win_size, max_level,
                         criteria, flags, min_eig_threshold):
    """
    :Returns:
        nextPts, status, err
    """
    return cv2.calcOpticalFlowPyrLK(prev_img,
                                    next_img,
                                    prev_pts,
                                    win_size,
                                    max_level,
                                    criteria,
                                    flags,
                                    min_eig_threshold)


def buildOpticalFlowPyramid(img, win_size, max_level):
    """
    :Returns:
        retval, pyramid
    """
    return cv2.buildOpticalFlowPyramid(img, win_size, max_level)


# ==============================================================================
# Photographic repair/denoising


def inpaint(image, mask, flags=cv2.INPAINT_NS):
    ### Added to Wralea.py
    """The function can be used to restore the image content in a provided
    region of interest using the neighborhood to this region.

    :Parameters:
    -`image` specifies the loaded image that should be processsed.
    The image can be either a 8-Bit 1- or 3-channel image.

    -`mask` specifies a mask by a provided 1-channel 8-Bit image, in which
    non-zero pixels define the regions of interest, which should be restored.

    -`flags`

    :Returns:
    -`restoredimage`, which is the image with the restored pixels.

    :Notes:
    """
    return cv2.inpaint(image, mask, flags)


def fastNlMeansDenoising(image,
                         h=float(3),
                         template_window_size=7,
                         search_window_size=21):
    """The function performs denoising using the Non-local Means Denoising
    algorithm provided by the OpenCV2 library.
    
    :Parameters:
    :Returns:
    :Notes:
    """
    return numpy.int_(cv2.fastNlMeansDenoising(
        image, h, template_window_size, search_window_size))


def fastNlMeansDenoisingColored(image,
                                h=float(3),
                                h_color=float(3),
                                template_window_size=7,
                                search_window_size=21):
    """The function performs denoising using the Non-local Means Denoising
     algorithm provided by the OpenCV2 library.
    
    :Parameters:
    :Returns:
    :Notes:
    """
    return numpy.int_(cv2.fastNlMeansDenoisingColored(
        image, h, h_color, template_window_size, search_window_size))
