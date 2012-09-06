/* Copyright 2012, Igalia S.L.
 * Author Carlos Guerreiro cguerreiro@igalia.com
 * Licensed under the MIT license */

#include <cstdio>
#include <sstream>

#include <v8.h>
#include <node.h>
#include <node_buffer.h>

#include "randomForest.h"

using namespace v8;
using namespace node;
using namespace std;
using namespace IncrementalRandomForest;

class IRF: ObjectWrap {
private:
  Forest* f;

  static void setFeatures(Sample* s, Local<Object>& features) {
    Local<Array> featureNames = features->GetOwnPropertyNames();
    int featureCount = featureNames->Length();
    int i;
    for(i = 0; i < featureCount; ++i) {
      // FIXME: verify that this is an integer
      Local<Integer> n = featureNames->Get(i)->ToInteger();
      // FIXME: verify that this is a number
      Local<Number> v = features->Get(n->Value())->ToNumber();
      s->xCodes[n->Value()] = v->Value();
    }
  }

  static void getFeatures(Sample* s, Local<Object>& features) {
    map<int, float>::const_iterator it;
    char key[16];
    for(it = s->xCodes.begin(); it != s->xCodes.end(); ++it) {
      sprintf(key, "%d", it->first);
      features->Set(String::New(key), Number::New(it->second));
    }
  }

public:
  IRF(uint32_t count) : ObjectWrap() {
    f = create(count);
  }

  IRF(Forest* withF) : ObjectWrap(), f(withF) {
  }

  ~IRF() {
    destroy(f);
  }

  static void init(Handle < Object > target, Handle<Value> (*func)(const Arguments&), Persistent<FunctionTemplate>& ct, const char* name) {
    Local<FunctionTemplate> t = FunctionTemplate::New(func);
    ct = Persistent<FunctionTemplate>::New(t);
    ct->InstanceTemplate()->SetInternalFieldCount(1);
    Local<String> nameSymbol = String::NewSymbol(name);
    ct->SetClassName(nameSymbol);
    NODE_SET_PROTOTYPE_METHOD(ct, "add", add);
    NODE_SET_PROTOTYPE_METHOD(ct, "remove", remove);
    NODE_SET_PROTOTYPE_METHOD(ct, "classify", classify);
    NODE_SET_PROTOTYPE_METHOD(ct, "classifyPartial", classifyPartial);
    NODE_SET_PROTOTYPE_METHOD(ct, "asJSON", asJSON);
    NODE_SET_PROTOTYPE_METHOD(ct, "statsJSON", statsJSON);
    NODE_SET_PROTOTYPE_METHOD(ct, "each", each);
    NODE_SET_PROTOTYPE_METHOD(ct, "commit", commit);
    NODE_SET_PROTOTYPE_METHOD(ct, "toBuffer", toBuffer);
    target->Set(nameSymbol, ct->GetFunction());
  }

  static Handle<Value> fromBuffer(const Arguments& args) {
    if(args.Length() != 1) {
      return ThrowException(Exception::Error(String::New("add takes 3 arguments")));
    }

    if(!Buffer::HasInstance(args[0]))
      return ThrowException(Exception::Error(String::New("argument must be a Buffer")));

    Local<Object> o = args[0]->ToObject();

    cerr << Buffer::Length(o) << endl;

    stringstream ss(Buffer::Data(o));

    IRF* ih = new IRF(load(ss));
    ih->Wrap(args.This());
    return args.This();
  }

  static Handle<Value> New(const Arguments& args) {
    HandleScope scope;

    if (!args.IsConstructCall()) {
      return ThrowException(Exception::TypeError(String::New("Use the new operator to create instances of this object.")));
    }

    IRF* ih;
    if(args.Length() >= 1) {
      if(args[0]->IsNumber()) {
        uint32_t count = args[0]->ToInteger()->Value();
        ih = new IRF(count);
      } else if(Buffer::HasInstance(args[0])) {
        Local<Object> o = args[0]->ToObject();
        stringstream ss(Buffer::Data(o));
        ih = new IRF(load(ss));
      } else {
        return ThrowException(Exception::Error(String::New("argument 1 must be a number (number of trees) or a Buffer (to create from)")));
      }
    } else
      ih = new IRF(1);

    ih->Wrap(args.This());
    return args.This();
  }

  static Handle<Value> add(const Arguments& args) {
    HandleScope scope;

    if(args.Length() != 3) {
      return ThrowException(Exception::Error(String::New("add takes 3 arguments")));
    }

    Local<String> suid = *args[0]->ToString();
    if(suid.IsEmpty())
      return ThrowException(Exception::Error(String::New("argument 1 must be a string")));

    if(!args[1]->IsObject())
      return ThrowException(Exception::Error(String::New("argument 2 must be a object")));
    Local<Object> features = *args[1]->ToObject();

    if(!args[2]->IsNumber())
      return ThrowException(Exception::Error(String::New("argument 3 must be a number")));
    Local<Number> y = *args[2]->ToNumber();

    IRF* ih = ObjectWrap::Unwrap<IRF>(args.This());
    Sample* s = new Sample();
    s->suid = *String::AsciiValue(suid);
    s->y = y->Value();
    setFeatures(s, features);

    return scope.Close(Boolean::New(IncrementalRandomForest::add(ih->f, s)));
  }

  static Handle<Value> remove(const Arguments& args) {
    HandleScope scope;

    if(args.Length() != 1) {
      return ThrowException(Exception::Error(String::New("remove takes 1 argument")));
    }

    Local<String> suid = *args[0]->ToString();
    if(suid.IsEmpty())
      return ThrowException(Exception::Error(String::New("argument 1 must be a string")));

    IRF* ih = ObjectWrap::Unwrap<IRF>(args.This());

    return scope.Close(Boolean::New(IncrementalRandomForest::remove(ih->f, *String::AsciiValue(suid))));
  }

  static Handle<Value> classify(const Arguments& args) {
    HandleScope scope;

    if(args.Length() != 1) {
      return ThrowException(Exception::Error(String::New("classify takes 1 argument")));
    }

    if(!args[0]->IsObject())
      return ThrowException(Exception::Error(String::New("argument 1 must be a object")));
    Local<Object> features = *args[0]->ToObject();

    IRF* ih = ObjectWrap::Unwrap<IRF>(args.This());

    IncrementalRandomForest::Sample s;
    setFeatures(&s, features);

    return scope.Close(Number::New(IncrementalRandomForest::classify(ih->f, &s)));
  }

  static Handle<Value> classifyPartial(const Arguments& args) {
    HandleScope scope;

    if(args.Length() != 2) {
      return ThrowException(Exception::Error(String::New("classifyPartial takes 2 argument")));
    }

    if(!args[0]->IsObject())
      return ThrowException(Exception::Error(String::New("argument 1 must be a object")));
    Local<Object> features = *args[0]->ToObject();


    if(!args[1]->IsNumber())
      return ThrowException(Exception::Error(String::New("argument 2 must be a number")));
    Local<Number> nTrees = *args[1]->ToNumber();

    IRF* ih = ObjectWrap::Unwrap<IRF>(args.This());

    IncrementalRandomForest::Sample s;
    setFeatures(&s, features);

    return scope.Close(Number::New(IncrementalRandomForest::classifyPartial(ih->f, &s, nTrees->Value())));
  }

  static Handle<Value> asJSON(const Arguments& args) {
    HandleScope scope;

    if(args.Length() != 0) {
      return ThrowException(Exception::Error(String::New("toJSON takes 0 arguments")));
    }

    IRF* ih = ObjectWrap::Unwrap<IRF>(args.This());

    stringstream ss;
    IncrementalRandomForest::asJSON(ih->f, ss);
    ss.flush();

    return scope.Close(String::New(ss.str().c_str()));
  }

  static Handle<Value> statsJSON(const Arguments& args) {
    HandleScope scope;

    if(args.Length() != 0) {
      return ThrowException(Exception::Error(String::New("statsJSON takes 0 arguments")));
    }

    IRF* ih = ObjectWrap::Unwrap<IRF>(args.This());

    stringstream ss;
    IncrementalRandomForest::statsJSON(ih->f, ss);
    ss.flush();

    return scope.Close(String::New(ss.str().c_str()));
  }

  static Handle<Value> each(const Arguments& args) {
    HandleScope scope;

    if(args.Length() != 1) {
      return ThrowException(Exception::Error(String::New("each takes 1 argument")));
    }
    if (!args[0]->IsFunction()) {
      return ThrowException(Exception::TypeError(String::New("argument must be a callback function")));
    }
    // There's no ToFunction(), use a Cast instead.
    Local<Function> callback = Local<Function>::Cast(args[0]);

    Local<Value> k = Local<Value>::New(Undefined());
    Local<Value> v = Local<Value>::New(Undefined());

    const unsigned argc = 3;
    Local<Value> argv[argc] = { v };

    IRF* ih = ObjectWrap::Unwrap<IRF>(args.This());

    SampleWalker* walker = getSamples(ih->f);

    Local<Object> globalObj = Context::GetCurrent()->Global();
    Local<Function> objectConstructor = Local<Function>::Cast(globalObj->Get(String::New("Object")));

    while(walker->stillSome()) {
      Sample* s = walker->get();
      argv[0] = Local<String>::New(String::New(s->suid.c_str()));
      Local<Object> features = Object::New();
      getFeatures(s, features);
      argv[1] = features;
      argv[2] = Local<Number>::New(Number::New(s->y));
      TryCatch tc;
      Local<Value> ret = callback->Call(Context::GetCurrent()->Global(), argc, argv);
      if(ret.IsEmpty() || ret->IsFalse())
        break;
    }

    delete walker;

    return Undefined();
  }

  static Handle<Value> commit(const Arguments& args) {
    HandleScope scope;

    if(args.Length() != 0) {
      return ThrowException(Exception::Error(String::New("commit takes 0 arguments")));
    }

    IRF* ih = ObjectWrap::Unwrap<IRF>(args.This());

    IncrementalRandomForest::commit(ih->f);

    return scope.Close(Undefined());
  }

  static Handle<Value> toBuffer(const Arguments& args) {
    HandleScope scope;

    if(args.Length() != 0) {
      return ThrowException(Exception::Error(String::New("save takes 0 arguments")));
    }

    IRF* ih = ObjectWrap::Unwrap<IRF>(args.This());
    stringstream ss(stringstream::out | stringstream::binary);
    save(ih->f, ss);
    ss.flush();

    Buffer* out = Buffer::New(const_cast<char*>(ss.str().c_str()), ss.tellp());

    return scope.Close(out->handle_);
  }
};

static Persistent<FunctionTemplate> irf_ct;

void RegisterModule(Handle<Object> target) {
  IRF::init(target, IRF::New, irf_ct, "IRF");
}

NODE_MODULE(irf, RegisterModule);
