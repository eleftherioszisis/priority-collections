from setuptools import setup, Extension, find_packages


extensions = [
    Extension("priority_collections.priority_heap", ["priority_collections/priority_heap.pyx"])
]


setup(
    name="priority_collections",
    description='NGV Architecture Cython Building Modules',
    author='Eleftherios Zisis',
    author_email='eleftherios.zisis@epfl.ch',
    packages=find_packages(),
    ext_modules=build_ext_modules(modules),
    include_package_data=True,
    use_scm_version=True,
    setup_requires=[
        'setuptools>=18.0',
        'setuptools_scm',
        'cython'
    ],
    install_requires=[
        'numpy>=1.13'
    ],
)

