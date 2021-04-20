from setuptools import setup, Extension, find_packages


modules = {
    "priority_collections.priority_heap": ["priority_collections/priority_heap"]
}


def build_ext_modules(modules):

    # check if cython is available
    try:
        from Cython.Build import cythonize
        has_cython = True
    except ImportError:
        has_cython = False

    ext = 'pyx' if has_cython else 'c'

    extensions = [
        Extension(name, [f'{path}.{ext}' for path in paths]) for name, paths in modules.items()
    ]

    if has_cython:
        print('Building .pyx sources using cython')
        return cythonize(
            module_list=extensions,
            compiler_directives={
                "language_level": 3,
                "embedsignature": True
            })

    print('Building .c sources without cython', extensions)
    return extensions


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
        'setuptools_scm'
    ],
    install_requires=[
        'numpy>=1.13'
    ],
)

