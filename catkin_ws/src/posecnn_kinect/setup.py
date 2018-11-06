## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
from distutils.extension import Extension



# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=[
        'kinect_package', 'datasets', 'fcn', 'utils', 'rpn_layer', 'normals', 'networks', 'backprojecting_layer', 'projecting_layer',
        'average_distance_loss', 'computing_flow_layer', 'computing_label_layer', 'gradient_reversal_layer', 'hard_label_layer',
        'hough_voting_gpu_layer', 'roi_pooling_layer', 'rpn_layer', 'triplet_loss', 'hough_voting_layer', 'nms', 'libsynthesizer'
    ],
    package_dir={'': 'src'},
    package_data={
        'utils': ['cython_bbox.so'],
        'normals': ['gpu_normals.so'],
        'backprojecting_layer' : ['backprojecting.so'],
        'projecting_layer' : ['projecting.so'],
        '' : ['*.so']
    },
    include_package_data=True,
)

setup(**setup_args)
