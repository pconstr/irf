from distutils.core import setup, Extension

module1 = Extension('irf',
                    sources = ['irfmodule.cpp','randomForest.cpp','MurmurHash3.cpp'])

setup (name = 'irf',
       version = '0.1',
       description = 'Incremental Random Forest',
       ext_modules = [module1])
