/* Copyright 2010-2011 Carlos Guerreiro
 * Licensed under the MIT license */

#include <Python.h>
#include "structmember.h"

#include "randomForest.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <vector>
#include <map>

#include <cstdlib>
#include "MurmurHash3.h"

using namespace std;
using namespace IncrementalRandomForest;

struct IRF {
  PyObject_HEAD
  Forest* forest;

  IRF(void) {
    forest = 0;
  }
  ~IRF(void) {
    if(forest)
      destroy(forest);
  }
};

static void IRF_dealloc(IRF* self) {
  self->~IRF();
  self->ob_type->tp_free((PyObject*)self);
}

static PyObject* IRF_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  IRF *self;

  PyObject* firstArg = PyTuple_GetItem(args, 0);

  bool fromFile = firstArg && PyString_Check(firstArg);

  char* fname;

  if(fromFile) {
    if(!PyArg_ParseTuple(args, "s",
                         &fname))
      return 0;

    ifstream inF(fname);
    if(!inF.is_open()) {
      // FIXME: raise exception?
      return 0;
    }

    self = new (type->tp_alloc(type, 0)) IRF();
    if(self) {
      self->forest = load(inF);
    }
  } else {
    int nTrees;
    if(!PyArg_ParseTuple(args, "i", &nTrees))
      return 0;

    self = new (type->tp_alloc(type, 0)) IRF();
    if(self) {
      self->forest = create(nTrees);
    }
  }

  return (PyObject *)self;
}

static int IRF_init(IRF *self, PyObject *args, PyObject *kwds) {
  return 0;
}

static PyMemberDef IRF_members[] = {
  {NULL}  /* Sentinel */
};

static PyObject* IRF_commit(IRF* self) {
  commit(self->forest);
  return Py_BuildValue("");
}

static PyObject* IRF_validate(IRF* self) {
  return PyBool_FromLong(validate(self->forest));
}

static PyObject* IRF_asJSON(IRF* self) {
  stringstream ss;
  asJSON(self->forest, ss);
  ss.flush();
  return Py_BuildValue("s", ss.str().c_str());
}

static PyObject* IRF_save(IRF* self, PyObject* args) {
  char* fname;

  if(!PyArg_ParseTuple(args, "s",
                       &fname)) {
    return 0;
  }

  ofstream outS(fname);
  if(!outS.is_open())
    return PyBool_FromLong(false);

  return PyBool_FromLong(save(self->forest, outS));
}

static PyObject* packFeatures(Sample* s) {
  PyObject* d = PyDict_New();
  for(map<int, float>::const_iterator it = s->xCodes.begin(); it != s->xCodes.end(); ++it) {
    PyObject* k = Py_BuildValue("i", it->first);
    PyObject* v = Py_BuildValue("f", it->second);
    PyDict_SetItem(d, k, v);
    Py_DECREF(k);
    Py_DECREF(v);
  }
  return d;
}

static bool extractFeatures(PyObject* features, Sample* s) {
  PyObject *key, *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(features, &pos, &key, &value)) {
    long k = PyInt_AsLong(key);
    if(k == -1 && PyErr_Occurred() != 0) {
      return false;
    }
    double v = PyFloat_AsDouble(value);
    if(v == -1 && PyErr_Occurred() != 0) {
      return false;
    }
    s->xCodes[k] = v;
  }
  return true;
}

static PyObject* IRF_classify(IRF* self, PyObject* args) {
  PyObject* features;
  if(!PyArg_ParseTuple(args, "O",
                       &features
                       ))
    return 0;

  Sample s;
  extractFeatures(features, &s);

  return Py_BuildValue("f", classify(self->forest, &s));
}

static PyObject* IRF_remove(IRF* self, PyObject* args) {
  char* sampleId;
  if(!PyArg_ParseTuple(args, "s",
                       &sampleId))
    return 0;
  return PyBool_FromLong(remove(self->forest, sampleId));
}

static PyObject* IRF_add(IRF* self, PyObject* args) {
  char* sampleId;
  PyObject* features;
  float target;

  if(!PyArg_ParseTuple(args, "sOf",
                       &sampleId,
                       &features,
                       &target)) {
    return 0;
  }

  Sample* s = new Sample();

  s->suid = sampleId;
  s->y = target;

  if(!extractFeatures(features, s)) {
    cerr << "failed to extract features!" << endl;
    delete s;
    return 0;
  }

  return PyBool_FromLong(add(self->forest, s));
}

static PyObject* IRF_samples(IRF* self, PyObject* args);

static PyMethodDef IRF_methods[] = {
  {"commit", (PyCFunction)IRF_commit, METH_NOARGS,
   "Commit pending changes"
  },
  {"asJSON", (PyCFunction)IRF_asJSON, METH_NOARGS,
   "Encode as JSON"
  },
  {"save", (PyCFunction)IRF_save, METH_VARARGS,
   "Save forest to file"
  },
  {"validate", (PyCFunction)IRF_validate, METH_NOARGS,
   "Validate forest"
  },
  {"classify", (PyCFunction)IRF_classify, METH_VARARGS,
   "Classify according to features"
  },
  {"add", (PyCFunction)IRF_add, METH_VARARGS,
   "Add a sample"
  },
  {"remove", (PyCFunction)IRF_remove, METH_VARARGS,
   "Remove a sample"
  },
  {"samples", (PyCFunction)IRF_samples, METH_NOARGS,
   "Get stored samples"
  },
  {NULL}  /* Sentinel */
};

static PyTypeObject IRFType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /*ob_size*/
  "irf.IRF",             /*tp_name*/
  sizeof(IRF), /*tp_basicsize*/
  0,                         /*tp_itemsize*/
  (destructor)IRF_dealloc,   /*tp_dealloc*/
  0,                         /*tp_print*/
  0,                         /*tp_getattr*/
  0,                         /*tp_setattr*/
  0,                         /*tp_compare*/
  0,                         /*tp_repr*/
  0,                         /*tp_as_number*/
  0,                         /*tp_as_sequence*/
  0,                         /*tp_as_mapping*/
  0,                         /*tp_hash */
  0,                         /*tp_call*/
  0,                         /*tp_str*/
  0,                         /*tp_getattro*/
  0,                         /*tp_setattro*/
  0,                         /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT,        /*tp_flags*/
  "IRF objects",           /* tp_doc */
  0,                   /* tp_traverse */
  0,                   /* tp_clear */
  0,                   /* tp_richcompare */
  0,                   /* tp_weaklistoffset */
  0,                   /* tp_iter */
  0,                   /* tp_iternext */
  IRF_methods,             /* tp_methods */
  IRF_members,             /* tp_members */
  0,                         /* tp_getset */
  0,                         /* tp_base */
  0,                         /* tp_dict */
  0,                         /* tp_descr_get */
  0,                         /* tp_descr_set */
  0,                         /* tp_dictoffset */
  (initproc)IRF_init,      /* tp_init */
  0,                         /* tp_alloc */
  IRF_new,                 /* tp_new */
};

struct SampleIter {
  PyObject_HEAD
  SampleWalker* walker;

  void setRange(SampleWalker* w) {
    delete walker;
    walker = w;
  }

  SampleIter(SampleWalker* w) {
    walker = w;
  }

  SampleIter(void) {
    walker = 0;
  }

  ~SampleIter(void) {
    delete walker;
  }
};

static void SampleIter_dealloc(SampleIter* self) {
  self->~SampleIter();
  self->ob_type->tp_free((PyObject*)self);
}

static PyObject* SampleIter_new(PyTypeObject *type, PyObject *ars, PyObject *kwds) {
  SampleIter* self;

  self = new (type->tp_alloc(type, 0)) SampleIter();
  if( self != NULL) {
  }
  return (PyObject*)self;
}

static int SampleIter_init(SampleIter* self, PyObject* args, PyObject* kwds) {
  return 0;
}

PyObject* SampleIter_iter(PyObject *self) {
  Py_INCREF(self);
  return self;
}

PyObject* SampleIter_iternext(PyObject *self) {
  SampleIter *p = (SampleIter*)self;
  if(p->walker->stillSome()) {
    Sample* s = p->walker->get();
    return Py_BuildValue("(sNf)", s->suid.c_str(), packFeatures(s), s->y);
  } else {
    /* Raising of standard StopIteration exception with empty value. */
    PyErr_SetNone(PyExc_StopIteration);
    return NULL;
  }
}

static PyTypeObject SampleIterType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /*ob_size*/
  "irf.SampleIter",            /*tp_name*/
  sizeof(SampleIter),       /*tp_basicsize*/
  0,                         /*tp_itemsize*/
  (destructor)SampleIter_dealloc,                         /*tp_dealloc*/
  0,                         /*tp_print*/
  0,                         /*tp_getattr*/
  0,                         /*tp_setattr*/
  0,                         /*tp_compare*/
  0,                         /*tp_repr*/
  0,                         /*tp_as_number*/
  0,                         /*tp_as_sequence*/
  0,                         /*tp_as_mapping*/
  0,                         /*tp_hash */
  0,                         /*tp_call*/
  0,                         /*tp_str*/
  0,                         /*tp_getattro*/
  0,                         /*tp_setattro*/
  0,                         /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_ITER,
  /* tp_flags: Py_TPFLAGS_HAVE_ITER tells python to
     use tp_iter and tp_iternext fields. */
  "Internal Sample iterator objects",           /* tp_doc */
  0,  /* tp_traverse */
  0,  /* tp_clear */
  0,  /* tp_richcompare */
  0,  /* tp_weaklistoffset */
  SampleIter_iter,  /* tp_iter: __iter__() method */
  SampleIter_iternext,  /* tp_iternext: next() method */
  0, /* tp_methods */
  0, /* tp_members */
  0,                         /* tp_getset */
  0,                         /* tp_base */
  0,                         /* tp_dict */
  0,                         /* tp_descr_get */
  0,                         /* tp_descr_set */
  0,                         /* tp_dictoffset */
  (initproc)SampleIter_init,      /* tp_init */
  0,                         /* tp_alloc */
  SampleIter_new,                 /* tp_new */
};

static PyObject* IRF_load(PyObject* self, PyObject* args) {
  IRF* p;

  char* fname;
  if(!PyArg_ParseTuple(args, "s",
                       &fname))
    return 0;

  p = (IRF*) PyObject_CallObject((PyObject*) &IRFType, args);

  if (!p) return NULL;

  return (PyObject*) p;
}

static PyMethodDef module_methods[] = {
  {"load", (PyCFunction)IRF_load, METH_VARARGS,
   "load random forest from file"
  },
  {NULL}  /* Sentinel */
};

static PyObject* IRF_samples(IRF* self, PyObject* args) {
  SampleIter *p;

  PyObject *argList = Py_BuildValue("()");
  p = (SampleIter*) PyObject_CallObject((PyObject*) &SampleIterType, argList);
  Py_DECREF(argList);

  if (!p) return NULL;

  /* I'm not sure if it's strictly necessary. */
  if (!PyObject_Init((PyObject *)p, &SampleIterType)) {
    Py_DECREF(p);
    return NULL;
  }

  p->setRange(getSamples(self->forest));

  return (PyObject *)p;
}

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC initirf(void) {
  PyObject* m;

  if (PyType_Ready(&IRFType) < 0)
    return;
  if (PyType_Ready(&SampleIterType) < 0)
    return;

  m = Py_InitModule3("irf", module_methods,
                     "Incremental Random Forest.");

  Py_INCREF(&IRFType);
  PyModule_AddObject(m, "IRF", (PyObject *)&IRFType);
  PyModule_AddObject(m, "SampleIter", (PyObject *)&SampleIterType);
}
