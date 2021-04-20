from setuptools import setup, Extension, find_packages
import numpy
from Cython.Build import cythonize


extensions = [
    Extension("priority_collections.priority_heap", ["priority_collections/priority_heap.pyx"], include_dirs=[numpy.get_include()]),
]


compiler_directives = {
    "language_level": 3,
    "embedsignature": True
}


setup(
    name="priority_collections",
    description='NGV Architecture Cython Building Modules',
    author='Eleftherios Zisis',
    author_email='eleftherios.zisis@epfl.ch',
    packages=find_packages(),
    ext_modules=cythonize(extensions, compiler_directives=compiler_directives),
    include_package_data=True,
    use_scm_version=True,
    setup_requires=[
        'numpy>=1.13',
        'setuptools_scm'
    ],
    install_requires=[
        'numpy>=1.13'
    ],
)

