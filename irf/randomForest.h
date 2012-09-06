/* Copyright 2010-2011 Carlos Guerreiro
 * Licensed under the MIT license */

#ifndef PCONSTR_RANDOMFOREST_H
#define PCONSTR_RANDOMFOREST_H

#include <map>
#include <iostream>
#include <vector>

namespace IncrementalRandomForest {

  struct DecisionTreeNode;

  struct Sample {
    std::string suid;
    float y;
    std::map<int, float> xCodes;
  };

  // FIXME: should be opaque
  struct TreeState {
    unsigned int seed;
    TreeState(void) : seed(1) {
    }
  };

  class SampleWalker {
  private:
  public:
    virtual ~SampleWalker(void) {
    }
    virtual bool stillSome(void) const = 0;
    virtual Sample* get(void) = 0;
  };

  class Forest;

  Forest* create(int nTrees);
  void destroy(Forest* rf);
  Forest* load(std::istream& forestS);
  bool save(Forest* rf, std::ostream& outS);
  void asJSON(Forest* rf, std::ostream& outS);
  bool add(Forest* rf, Sample* s);
  bool remove(Forest* rf, const char* sId);
  void commit(Forest* rf);
  float classify(Forest* rf, Sample* s);
  float classifyPartial(Forest* rf, Sample* s, int n);
  bool validate(Forest* rf);
  SampleWalker* getSamples(Forest* rf);
}

#endif
