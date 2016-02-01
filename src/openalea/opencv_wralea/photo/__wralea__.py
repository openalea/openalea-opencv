# This file has been automatically generated by pkg_builder

from openalea.core import *

__name__ = 'openalea.opencv.photo'
__version__ = '1.0.0'
__license__ = 'CeCILL-C'
__author__ = 'M.Mielewczik, C.Fournier, C. Pradal'
__institutes__ = None
__description__ = ''
__url__ = 'http://openalea.gforge.inria.fr'

__editable__ = 'True'
__icon__ = 'inpainticon.png'
__alias__ = []


__all__ = ['opencv2_inpaint']


               
opencv2_inpaint = Factory(name='inpaint',
                authors=' (M.Mielewczik)',
                description='',
                category='Image Processing',
                nodemodule='openalea.opencv.nodes',
                nodeclass='inpaint',
                #inputs=[{'interface': IFileStr, 'name': 'Image'}],
                outputs=({'interface': None, 'name': 'out'},),
                widgetmodule=None,
                widgetclass=None,
               )


opencv2_fastNlMeansDenoising = Factory(name='fastNlMeansDenoising',
                authors=' (M.Mielewczik)',
                description='',
                category='Image Processing',
                nodemodule='openalea.opencv.nodes',
                nodeclass='fastNlMeansDenoising',
                inputs=[{'interface': IFileStr, 'name': 'Image'},
                        {'interface': IFloat(1,3), 'name':'h', 'value':(3)},
                        {'interface': IInt(1,9), 'name':'templateWindowSize', 'value':(7)},
                        {'interface': IInt(1,100), 'name':'searchWindowSize', 'value':(21)}],
                outputs=({'interface': None, 'name': 'image'},),
                widgetmodule=None,
                widgetclass=None,
               )    

__all__.append('opencv2_fastNlMeansDenoising')

opencv2_fastNlMeansDenoisingColored = Factory(name='fastNlMeansDenoisingColored',
                authors=' (M.Mielewczik)',
                description='',
                category='Image Processing',
                nodemodule='openalea.opencv.nodes',
                nodeclass='fastNlMeansDenoisingColored',
                inputs=[{'interface': IFileStr, 'name': 'Image'},
                        {'interface': IFloat(1,3), 'name':'h', 'value':(3)},
                        {'interface': IFloat(1,3), 'name':'hColor', 'value':(3)},
                        {'interface': IInt(1,9), 'name':'templateWindowSize', 'value':(7)},
                        {'interface': IInt(1,21), 'name':'searchWindowSize', 'value':(21)}],
                outputs=({'interface': None, 'name': 'image'},),
                widgetmodule=None,
                widgetclass=None,
               )    

__all__.append('opencv2_fastNlMeansDenoisingColored')


 
    

