from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='cg_cpp',
      ext_modules=[cpp_extension.CppExtension('cg_cpp', ['cg.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
