from setuptools import setup, dist, Extension, find_packages

# https://github.com/pypa/pip/issues/5761
dist.Distribution().fetch_build_eggs(['Cython>=0.29.0', 'numpy>=1.10'])

import numpy


compiler_args = ["-DNDEBUG", "-O3"]
macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]


extensions = [
    Extension("priority_collections.priority_heap", ["priority_collections/priority_heap.pyx"], define_macros=macros, extra_compile_args=compiler_args)
]


setup(
    name="priority_collections",
    description='NGV Architecture Cython Building Modules',
    author='Eleftherios Zisis',
    author_email='eleftherios.zisis@epfl.ch',
    packages=find_packages(),
    ext_modules=extensions,
    include_dirs=[numpy.get_include()],
    include_package_data=True,
    use_scm_version=True,
    setup_requires=[
        'setuptools>=18.0',
        'setuptools_scm',
        'numpy>=1.19',
        'cython'
    ],
    install_requires=[
        'numpy>=1.19'
    ],
)

