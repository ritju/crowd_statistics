from setuptools import setup
from glob import glob
import os

package_name = 'crowd_statistics'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, 'crowd_statistics/nanodet', 'crowd_statistics/model', "crowd_statistics/nanodet/data", 'crowd_statistics/nanodet/data/dataset', 
                'crowd_statistics/nanodet/data/transform', 'crowd_statistics/nanodet/evaluator', 'crowd_statistics/nanodet/model', 
                'crowd_statistics/nanodet/model/arch', 'crowd_statistics/nanodet/model/backbone', 
                'crowd_statistics/nanodet/model/fpn', 'crowd_statistics/nanodet/model/head',
                'crowd_statistics/nanodet/model/loss', 'crowd_statistics/nanodet/model/module',
                'crowd_statistics/nanodet/model/weight_averager', 'crowd_statistics/nanodet/optim', 
                'crowd_statistics/nanodet/trainer', 'crowd_statistics/nanodet/util', 'crowd_statistics/nanodet/model/head/assigner',
                'crowd_statistics/Insightface_pytorch','crowd_statistics/Insightface_pytorch/mtcnn_pytorch','crowd_statistics/Insightface_pytorch/mtcnn_pytorch/src',
                'crowd_statistics/Insightface_pytorch/mtcnn_pytorch/src/weights','crowd_statistics/Insightface_pytorch/work_space',
                'crowd_statistics/libcudash',],
    include_package_data = True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='540399793@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "crowd_statistics_node = crowd_statistics.crowd_statistics_node:main",
            "dabai_crowd_statistics_node = crowd_statistics.dabai_nanodet_detect:main",
        ],
    },
)
