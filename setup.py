from setuptools import setup, dist, Extension, find_packages

# https://github.com/pypa/pip/issues/5761
dist.Distribution().fetch_build_eggs(['Cython>=0.15.1', 'numpy>=1.10'])

extensions = [
    Extension("priority_collections.priority_heap", ["priority_collections/priority_heap.pyx"])
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

