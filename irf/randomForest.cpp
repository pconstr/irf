/* Copyright 2010-2011 Carlos Guerreiro 
 * Licensed under the MIT license */

#include <algorithm>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <utility>

#include <map>
#include <vector>
#include <set>
#include <stack>

#include <fstream>
#include <sstream>
#include <cstdlib>

#include "randomForest.h"
#include "MurmurHash3.h"

#include <limits>

#include <google/sparse_hash_map>

using namespace std;
using google::sparse_hash_map;


namespace IncrementalRandomForest {

  static const unsigned int maxCodesToConsider = 30;
  static const unsigned int maxCodesToKeep = 40;
  static const unsigned int minEvidence = 2;
  static const unsigned int maxUnsplit = 30;
  static const unsigned int minBalanceSplit = 10;
  static const float minProbDiff = 0;
  static const float minEntropyGain = 0.01;

  template <class T>
  static inline string to_string (const T& t)
  {
    stringstream ss;
    ss << t;
    return ss.str();
  }
  
  static void printSample(ostream& out, Sample* s) {
    out << s->y;
    out << " " << s->xCodes.size();
    map<int, float>::const_iterator itCodes;
    for(itCodes = s->xCodes.begin(); itCodes != s->xCodes.end(); ++itCodes) {
      out << " " << itCodes->first << " " << itCodes->second;
    }
    out << endl;
  }

  static float entropyBinary(int c0, int c1) {
    float h = 0;
    const int n = c0 + c1;
    if(c0 > 0) {
      float p0 = (float) c0 / n;
      h -= p0 * log(p0);
    }
    if(c1 > 0) {
      float p1 = (float) c1 / n;
      h -= p1 * log(p1);
    }
    return h;
  }

  static void findUsedCodes(SampleWalker& sw, set<int>& uc) {
    map<int, float>::const_iterator itC;
    while(sw.stillSome()) {
      Sample* s = sw.get();
      for(itC = s->xCodes.begin(); itC != s->xCodes.end(); ++itC)
	uc.insert(itC->first);
    }
  }

  static void splitListByTarget(const vector<Sample*>& sl, vector<Sample*>& sl0, vector<Sample*>& sl1) {
    vector<Sample*>::const_iterator it;
    for(it = sl.begin(); it != sl.end(); ++it) {
      Sample* s = *it;
      if(s->y > 0.5)
	sl1.push_back(s);
      else
	sl0.push_back(s);
    }
  }

  class VectorSampleWalker : public SampleWalker {
  private:
    vector<Sample*>::const_iterator itCurr;
    vector<Sample*>::const_iterator itEnd;
  public:
    VectorSampleWalker(const vector<Sample*>& sv) :
      itCurr(sv.begin()), itEnd(sv.end()) {
    }
    virtual bool stillSome(void) const {
      return itCurr != itEnd;
    }
    virtual Sample* get(void) {
      return *itCurr++;
    }  
  };
  
  static void splitListAgainstCode(SampleWalker& sw, int c, vector<Sample*>& sl0, vector<Sample*>& sl1) {
    while(sw.stillSome()) {
      Sample* s = sw.get();
      map<int, float>::const_iterator itCode = s->xCodes.find(c);
      bool cc;
      if(itCode != s->xCodes.end()) {
	cc = itCode->second > 0.5;
      } else
	cc = false;
      
      if(cc)
	sl1.push_back(s);
      else
	sl0.push_back(s);
    }
  }

  typedef unsigned int CodeRankType;
  
  struct DecisionCounts {
    unsigned int c0p;
    unsigned int c1p;
    CodeRankType rank;

    DecisionCounts(void) {
      c0p = 0;
      c1p = 0;
      rank = 0; // needs to be set afterwards
    }

    bool enoughEvidence(DecisionTreeNode* dt) const;
    
    bool isZeroFor(DecisionTreeNode* dt) const;
    
    void print(ostream& outs, DecisionTreeNode* dt) const;
    
    float entropy(DecisionTreeNode* dt) const;
  };

  static bool operator == (const DecisionCounts& dc1, const DecisionCounts& dc2) {
    return (dc1.c0p == dc2.c0p) && (dc1.c1p == dc2.c1p);
  }
  
  static bool operator != (const DecisionCounts& dc1, const DecisionCounts& dc2) {
    return !(dc1 == dc2);
  }

  struct DecisionTreeInternal;
  struct DecisionTreeLeaf;
  
  struct DecisionTreeNode {
    int code; // iff code == -1 it's a leaf node
    unsigned int c0;
    unsigned int c1;
    sparse_hash_map<int, DecisionCounts> decisionCountMap;
    unsigned long id;
    pair<CodeRankType, int> minValidRank;
    DecisionTreeNode() : decisionCountMap() {
      decisionCountMap.set_deleted_key(-1);
      minValidRank = make_pair(0U, 0);
    }
    DecisionTreeInternal* checkInternal(void);
    DecisionTreeLeaf* checkLeaf(void);
    bool checkType(DecisionTreeInternal**, DecisionTreeLeaf**);
  };
  
  struct DecisionTreeInternal : public DecisionTreeNode {
    DecisionTreeNode* negative;
    DecisionTreeNode* positive;
  };
  
  struct DecisionTreeLeaf : public DecisionTreeNode {
    float value;
    vector<Sample*> samples;
  };

  DecisionTreeInternal* DecisionTreeNode::checkInternal(void) {
    if(code != -1)
      return static_cast<DecisionTreeInternal*>(this);
    else
      return 0;
  }

  DecisionTreeLeaf* DecisionTreeNode::checkLeaf(void) {
    if(code == -1)
      return static_cast<DecisionTreeLeaf*>(this);
    else
      return 0;
  }

  bool DecisionTreeNode::checkType(DecisionTreeInternal** ni, DecisionTreeLeaf** nl) {
    if(code == -1) {
      *ni = 0;
      *nl = static_cast<DecisionTreeLeaf*>(this);
      return true;
    } else {
      *ni = static_cast<DecisionTreeInternal*>(this);
      *nl = 0;
      return false;
    }
  }

  bool DecisionCounts::enoughEvidence(DecisionTreeNode* dt) const {
    const unsigned int c0n = dt->c0 - c0p;
    const unsigned int c1n = dt->c1 - c1p;
    return ((c0n + c1n) >= minEvidence) && ((c0p + c1p) >= minEvidence);
  }

  void DecisionCounts::print(ostream& outs, DecisionTreeNode* dt) const {
    const unsigned int c0n = dt->c0 - c0p;
    const unsigned int c1n = dt->c1 - c1p;
    outs << "    c0n = " << c0n << endl;
    outs << "    c1n = " << c1n << endl;
    outs << "    c0p = " << c0p << endl;
    outs << "    c1p = " << c1p << endl;
    outs << "    rank = " << rank << endl;
  }

  float DecisionCounts::entropy(DecisionTreeNode* dt) const {
    // FIXME: redundant computation of c0n and c1n all over the place
    const unsigned int c0n = dt->c0 - c0p;
    const unsigned int c1n = dt->c1 - c1p;
    float hn = entropyBinary(c0n, c1n);
    float hp = entropyBinary(c0p, c1p);
    int cp = c0p + c1p;
    int cn = c0n + c1n;
    return (hn * cn + hp * cp) / (cn + cp);
  }
  
  static DecisionTreeLeaf* makeLeaf(TreeState& ts, float v) {
    DecisionTreeLeaf* n = new DecisionTreeLeaf();
    n->code = -1;
    n->value = v;
    n->c0 = 0;
    n->c1 = 0;
    n->id = rand_r(&ts.seed);
    return n;
  }
  
  static DecisionTreeInternal* makeInternal(TreeState& ts, int c, DecisionTreeNode* n0, DecisionTreeNode* n1) {
    DecisionTreeInternal* n = new DecisionTreeInternal();
    n->code = c;
    n->negative = n0;
    n->positive = n1;
    n->id = rand_r(&ts.seed);
    return n;
  }

  bool DecisionCounts::isZeroFor(DecisionTreeNode* dt) const {
    const unsigned int c0n = dt->c0 - c0p;
    const unsigned int c1n = dt->c1 - c1p;
    return (c0n == dt->c0) && (c1n == dt->c1) && (c0p == 0) && (c1p == 0);
  }

  static void destroyDecisionTreeNode(DecisionTreeNode* dt) {
    DecisionTreeInternal* ni;
    DecisionTreeLeaf* nl;
    if(!dt->checkType(&ni, &nl)) {
      destroyDecisionTreeNode(ni->negative);
      destroyDecisionTreeNode(ni->positive);
      delete ni;
    } else
      delete nl;
  }

  static DecisionTreeNode* emptyDecisionTree(TreeState& ts) {
    return makeLeaf(ts, 0);
  }
  
  static CodeRankType codeRankInNode(int code, unsigned long nodeId) {
    char s[64];
    // FIXME: this is possibly quite slow
    sprintf(s, "%d%lu", code, nodeId);
    uint32_t out;
    MurmurHash3_x86_32(s, strlen(s), 42, &out);
    return out;
  }

  static void updateValue(DecisionTreeLeaf* l) {
    int n =  l->c0 + l->c1;
    if(n == 0)
      l->value = 1; // assume positive, FIXME: does this make sense?
    else
      l->value = (float)l->c1 / n;
  }
  
  class TreeSampleWalker;
  
  static void computeDecisionCounters(DecisionTreeNode* dt,
				      const TreeSampleWalker& origSW,
				      sparse_hash_map<int, DecisionCounts>& decisionCountMap,
				      unsigned int& outC0,
				      unsigned int& outC1,
				      pair<CodeRankType, int>& minValidRank);

  static pair<CodeRankType, int> findMinRankToConsider(const sparse_hash_map<int, DecisionCounts>& dcMap) {
    sparse_hash_map<int, DecisionCounts>::const_iterator mapIt;
    pair<CodeRankType, int> minRankToConsider;
    minRankToConsider.first = 0; minRankToConsider.second = 0; // FIXME: is this necessary?
    if(dcMap.size() > maxCodesToConsider) {
      set<pair<CodeRankType, int> > ranks;
      for(mapIt = dcMap.begin(); mapIt != dcMap.end(); ++mapIt) {
	ranks.insert(make_pair(mapIt->second.rank, mapIt->first));
	if(ranks.size() > maxCodesToConsider)
	  ranks.erase(ranks.begin());
      }
      minRankToConsider = *(ranks.begin());
    }
    return minRankToConsider;
  }

  static int findMinEntropyCode(float currentEntropy, DecisionTreeNode* dt) {
    float minEntropy = 10;
    int minEntropyCode = -1;

    sparse_hash_map<int, DecisionCounts>::const_iterator mapIt;

    // FIXME: this is possibly too inneficient. keep DCs sorted by rank in a vector instead?
    pair<CodeRankType, int> minRankToConsider = findMinRankToConsider(dt->decisionCountMap);

    for(mapIt = dt->decisionCountMap.begin(); mapIt != dt->decisionCountMap.end(); ++mapIt) {
      const DecisionCounts& dc = mapIt->second;

      if(make_pair(dc.rank, mapIt->first) >= minRankToConsider &&
	 dc.enoughEvidence(dt)) {
	
	float ah = dc.entropy(dt);
	
	if(ah < minEntropy) {
	  minEntropy = ah;
	  minEntropyCode = mapIt->first;
	}
      }
    }
    
    if(minEntropyCode == -1)
      return -1;
    
    if(minEntropy < currentEntropy)
      return minEntropyCode;
    
    return -1;
  }
  
  static void splitNode(TreeState& ts, DecisionTreeInternal* dt, int minEntropyCode, SampleWalker& sw);
  
  class TreeSampleWalker : public SampleWalker {
  private:
    stack<DecisionTreeNode*> st;
    vector<Sample*>::iterator itCurr, itEnd;
    bool hasSome;
    
    bool stackDown(DecisionTreeNode* n) {
      DecisionTreeInternal* ni;
      DecisionTreeLeaf* nl;
      while(!n->checkType(&ni, &nl)) {
	st.push(ni);
	n = ni->negative;
      }
      itCurr = nl->samples.begin();
      itEnd = nl->samples.end();
      if(itCurr != itEnd) {
	st.push(nl);
	return true;
      } else {
	return false;
      }
    }
  private:
    void advanceToNext(void) {
      while(!st.empty()) {
	DecisionTreeInternal* ni = st.top()->checkInternal();
	st.pop();
	if(stackDown(ni->positive))
	  break;
      }
    }
  public:
    TreeSampleWalker(DecisionTreeNode* n): st() {
      if(!stackDown(n)) {
	advanceToNext();
      }
    }
    virtual bool stillSome(void) const {
      return !st.empty();
    }
    virtual Sample* get(void) {
      Sample* s = *itCurr;
      
      ++itCurr;
      if(itCurr == itEnd) {
	st.pop(); // pop leaf
	
	advanceToNext();
      }
      
      return s;
    }
  };
  
  // samples
  static void setupLeafFromSamples(DecisionTreeLeaf* dt) {
    // FIXME: probably done already as we can only call this on a leaf
    dt->code = -1;
    
    computeDecisionCounters(dt,
			    TreeSampleWalker(dt),
			    dt->decisionCountMap,
			    dt->c0,
			    dt->c1,
			    dt->minValidRank);
    updateValue(dt);
  }

  static void computeDecisionCounters(DecisionTreeNode* dt,
				      const TreeSampleWalker& origSW,
				      sparse_hash_map<int, DecisionCounts>& decisionCountMap,
				      unsigned int& outC0,
				      unsigned int& outC1,
				      pair<CodeRankType, int>& minValidRank) {
    
    minValidRank = make_pair(0U, 0);
    
    set<int> usedCodes;
    TreeSampleWalker sw2 = origSW;
    findUsedCodes(sw2, usedCodes);
    
    set<int>::const_iterator ucIt;
    
    int c0 = 0;
    int c1 = 0;
    
    vector<Sample*>::const_iterator sIt;
    
    decisionCountMap.clear();
    
    for(TreeSampleWalker sw = origSW; sw.stillSome();) {
      Sample* s = sw.get();
      bool classIn = s->y >= 0.5;
      if(classIn)
	++c1;
      else
	++c0;
    }
    outC0 = c0;
    outC1 = c1;
    
    set<pair<CodeRankType, int> > ranks;
    for(ucIt = usedCodes.begin(); ucIt != usedCodes.end(); ++ucIt) {
      const int code = *ucIt;
      CodeRankType rank = codeRankInNode(code, dt->id);
      ranks.insert(make_pair(rank, code));
      if(ranks.size() > maxCodesToKeep) {
	minValidRank = max(minValidRank, make_pair(ranks.begin()->first, code + 1));
	ranks.erase(ranks.begin());
      }
    }
    
    set<pair<CodeRankType, int> >::const_iterator rIt;
    for(rIt = ranks.begin(); rIt != ranks.end(); ++rIt) {
      const int code = rIt->second;
      DecisionCounts& dc = decisionCountMap[code];
      dc.rank = codeRankInNode(code, dt->id);
      for(TreeSampleWalker sw=origSW; sw.stillSome();) {
	Sample* s = sw.get();
	bool classIn = s->y > 0.5;
	
	map<int, float>::const_iterator itCodeInS = s->xCodes.find(code);
	bool hasCode = itCodeInS != s->xCodes.end() && itCodeInS->second > 0.5;
	
	if(classIn) {
	  if(hasCode) {
	    ++(dc.c1p);
	  }
	} else {
	  if(hasCode) {
	    ++(dc.c0p);
	  }
	}
      }
    }
  }
  
  static DecisionTreeNode* splitLeafIfPossible(TreeState& ts, DecisionTreeNode* dt) {
    float currentEntropy = entropyBinary(dt->c0, dt->c1);
    int minEntropyCode = findMinEntropyCode(currentEntropy, dt);
    
    bool shouldBeSplit = minEntropyCode != -1;
    
    if(shouldBeSplit) {
      TreeSampleWalker sw(dt);
      DecisionTreeInternal* newInternal = makeInternal(ts, minEntropyCode, 0, 0);
      newInternal->c0 = dt->c0;
      newInternal->c1 = dt->c1;
      newInternal->minValidRank = dt->minValidRank;
      newInternal->decisionCountMap = dt->decisionCountMap;
      newInternal->id = dt->id;
      splitNode(ts, newInternal, minEntropyCode, sw);
      destroyDecisionTreeNode(dt);
      return newInternal;
    } else
      return dt;
  }
  
  static void splitNode(TreeState& ts, DecisionTreeInternal* dt, int minEntropyCode, SampleWalker& sw) {
    DecisionTreeLeaf* dtn = makeLeaf(ts, 0);
    DecisionTreeLeaf* dtp = makeLeaf(ts, 0);
    
    if(dt->decisionCountMap.find(minEntropyCode) == dt->decisionCountMap.end()) {
      cerr << " code " << minEntropyCode << " not found in decisionCountMap!" << endl;
      exit(1);
    }
    
    splitListAgainstCode(sw, minEntropyCode, dtn->samples, dtp->samples);
    
    setupLeafFromSamples(dtn);
    setupLeafFromSamples(dtp);
    
    if(dt->negative) {
      // resplit
      destroyDecisionTreeNode(dt->negative);
    }
    dt->negative = dtn;
    if(dt->positive) {
      // resplit
      destroyDecisionTreeNode(dt->positive);
    }
    dt->positive = dtp;
    
    dt->code = minEntropyCode;
    
    dt->negative = splitLeafIfPossible(ts, dt->negative);
    dt->positive = splitLeafIfPossible(ts, dt->positive);
  }
  
  static void updateDecisionCounters(DecisionTreeNode* dt, Sample* s, int addedBefore0, int addedBefore1, int direction = 1) {
    sparse_hash_map<int, DecisionCounts>::iterator dcIt;
    for(dcIt = dt->decisionCountMap.begin(); dcIt != dt->decisionCountMap.end();) {
      const int code = dcIt->first;
      DecisionCounts& dc = dcIt->second;
      map<int, float>::const_iterator codeIt = s->xCodes.find(code);
      if(codeIt == s->xCodes.end()) {
	// code not used in sample
      } else {
	// code used in sample
	if(s->y >= 0.5) {
	  if(codeIt->second >= 0.5) {
	    (dc.c1p) += direction;
	  }
	} else {
	  if(codeIt->second >= 0.5) {
	    (dc.c0p) += direction;
	  }
	}
      }
      
      if(direction < 0 && dc.c0p == 0 && dc.c1p == 0) {
	sparse_hash_map<int, DecisionCounts>::iterator toErase = dcIt;
	++dcIt;
	dt->decisionCountMap.erase(toErase);
      } else
	++dcIt;
    }
    
    if(direction < 0)
      return;
    
    // FIXME: unnecessary if it's clear no new codes will be added
    set<pair<CodeRankType, int> > ranks;
    for(dcIt = dt->decisionCountMap.begin(); dcIt != dt->decisionCountMap.end(); ++dcIt)
      ranks.insert(make_pair(dcIt->second.rank, dcIt->first));
    
    map<int, float>::const_iterator codeIt;
    for(codeIt = s->xCodes.begin(); codeIt != s->xCodes.end(); ++codeIt) {
      sparse_hash_map<int, DecisionCounts>::iterator dcIt = dt->decisionCountMap.find(codeIt->first);
      if(dcIt == dt->decisionCountMap.end()) {
	
	CodeRankType newRank = codeRankInNode(codeIt->first, dt->id);
	
	bool doInsert = make_pair(newRank, codeIt->first) >= dt->minValidRank;
	
	if(doInsert) {
	  dcIt = dt->decisionCountMap.insert(make_pair(codeIt->first, DecisionCounts())).first;
	  DecisionCounts& dc = dcIt->second;
	  dc.rank = newRank;
	  
	  if(s->y >= 0.5) {
	    if(codeIt->second >= 0.5) {
	      (dc.c1p) += direction;
	    }
	  } else {
	    if(codeIt->second >= 0.5) {
	      (dc.c0p) += direction;
	    }
	  }
	  
	  ranks.insert(make_pair(dc.rank, codeIt->first));
	  
	  if(ranks.size() > maxCodesToKeep) {
	    int toDrop = ranks.begin()->second;
	    dt->minValidRank = max(dt->minValidRank, make_pair(ranks.begin()->first, ranks.begin()->second + 1));
	    ranks.erase(ranks.begin());
	    dt->decisionCountMap.erase(toDrop);
	  }
	}
      }
    }
  }
  
  static void printDCs(const sparse_hash_map<int, DecisionCounts>& dc, DecisionTreeNode* dt) {
    set<pair<CodeRankType, int> > ranks;
    sparse_hash_map<int, DecisionCounts>::const_iterator it;
    for(it = dc.begin(); it != dc.end(); ++it) {
      ranks.insert(make_pair(it->second.rank, it->first));
    }
    set<pair<CodeRankType, int> >::const_reverse_iterator rIt;
    for(rIt = ranks.rbegin(); rIt != ranks.rend(); ++rIt)
      cerr << " " << rIt->first << "," << rIt->second;
    cerr << endl;
  }
  
  static void printNodeSamples(DecisionTreeNode* dt) {
    TreeSampleWalker sw(dt);
    while(sw.stillSome()) {
      printSample(cerr, sw.get());
    }
  }
  
  static bool compareDCsDir(const sparse_hash_map<int, DecisionCounts>& dcM1,
			    const sparse_hash_map<int, DecisionCounts>& dcM2,
			    DecisionTreeNode* dt,
			    const char* tag1,
			    const char* tag2) {
    bool valid = true;
    
    pair<CodeRankType, int> minR1 = findMinRankToConsider(dcM1);
    pair<CodeRankType, int> minR2 = findMinRankToConsider(dcM2);
    
    sparse_hash_map<int, DecisionCounts>::const_iterator itDC;
    
    int countIn = 0;
    for(itDC = dcM1.begin(); itDC != dcM1.end(); ++itDC) {
      int code = itDC->first;
      const DecisionCounts& dc = itDC->second;
      
      if(make_pair(dc.rank, code) >= minR1) {
	++countIn;
	sparse_hash_map<int, DecisionCounts>::const_iterator itDC2 = dcM2.find(code);
	if(itDC2 == dcM2.end()) {
	  if(!dc.isZeroFor(dt)) {
	    cerr << "ERROR: non-zero DC for code " << code << " not found: (" << tag1 << " in " << tag2 << ") in " << (long)dt << " : " << endl;
	    cerr << "minValidRank = " << dt->minValidRank.first << " , " << dt->minValidRank.second << endl;
	    cerr << "minR1 = (" << minR1.first << "," << minR1.second << ")     minR2 = (" << minR2.first << "," << minR2.second << ")" << endl;
	    dc.print(cerr, dt);
	    valid = false;
	  }
	} else {
	  const DecisionCounts& dc2 = itDC2->second;
	  if(make_pair(dc2.rank, code) < minR2) {
	    cerr << "ERROR: non-zero DC for code " << code << " found: (" << tag1 << " in " << tag2 << ") but not in top N in " << (long) dt << endl;
	    valid = false;
	  }
	  if(dc != dc2) {
	    cerr << "ERROR: DCs for code " << code << " don't match: (" << tag1 << " in " << tag2 << ") in " << (long) dt << " : " << endl;
	    dc.print(cerr, dt);
	    dc2.print(cerr, dt);
	    valid = false;
	  }
	}
      }
    }
    
    if(!valid) {
      cerr << "there were " << countIn << " codes over minimum out of a total of " << dcM1.size() << endl;
      cerr << tag1 << " : " << endl;
      printDCs(dcM1, dt);
      cerr << tag2 << " : " << endl;
      printDCs(dcM2, dt);
      printNodeSamples(dt);
    }
    
    return valid;
  }
  
  static bool compareDCs(const sparse_hash_map<int, DecisionCounts>& dcM1,
			 const sparse_hash_map<int, DecisionCounts>& dcM2,
			 DecisionTreeNode* dt,
			 const char* tag1,
			 const char* tag2) {
    bool valid = true;
    
    if(!compareDCsDir(dcM1, dcM2, dt, tag1, tag2))
      valid = false;
    if(!compareDCsDir(dcM2, dcM1, dt, tag2, tag1))
      valid = false;
    
    return valid;
  }
  
  static void insertLeafSamples(vector<Sample*>& v, SampleWalker& sw) {
    while(sw.stillSome())
      v.push_back(sw.get());
  }
  
  static void collectRecursive(DecisionTreeNode* dt, vector<Sample*>& v) {
    DecisionTreeInternal *ni;
    DecisionTreeLeaf* nl;
    if(dt->checkType(&ni, &nl)) {
      copy(nl->samples.begin(), nl->samples.end(), back_inserter(v));
    } else {
      collectRecursive(ni->negative, v);
      collectRecursive(ni->positive, v);
    }
  }
  
  static bool validateWalker(DecisionTreeNode* dt) {
    vector<Sample*> vRec;
    collectRecursive(dt, vRec);
    vector<Sample*> vWalker;
    TreeSampleWalker sw(dt);
    while(sw.stillSome())
      vWalker.push_back(sw.get());
    if(vWalker != vRec) {
      cerr << "ERROR: vWalker != vRec" << endl;
      return false;
    } else {
      return true;
    }
  }
  
  static bool validateDecisionTree(TreeState& ts, DecisionTreeNode* dt) {
    
    bool valid = true;
    
    DecisionTreeInternal* ni;
    DecisionTreeLeaf* nl;
    dt->checkType(&ni, &nl);
    
    // make sure there are no multiple versions of the same post
    
    if(!validateWalker(dt))
      valid = false;
    
    if(nl) {
      vector<Sample*>::const_iterator itS;
      map<string, Sample*> suidMap;
      for(itS = nl->samples.begin(); itS != nl->samples.end(); ++itS) {
	Sample* s = *itS;
	const string& suid = s->suid;
	
	map<string, Sample*>::const_iterator itSS = suidMap.find(suid);
	if(itSS != suidMap.end()) {
	  cerr << "ERROR: multiple occurances of post " << suid << endl;
	  valid = false;
	}
	suidMap[s->suid] = s;
      }
    }
    
    // validate counters;
    
    if(nl) {
      if(nl->samples.size() != (nl->c0 + nl->c1)) {
	cerr << "ERROR: c0 + c1 != #samples" << endl;
	valid = false;
      }
    }
    
    sparse_hash_map<int, DecisionCounts>::const_iterator itDC;
    for(itDC = dt->decisionCountMap.begin(); itDC != dt->decisionCountMap.end(); ++itDC) {
      const DecisionCounts& dc = itDC->second;
      
      const unsigned int c0n = dt->c0 - dc.c0p;
      const unsigned int c1n = dt->c1 - dc.c1p;
      
      if(c0n < 0) {
	cerr << "ERROR: c0n < 0: " << c0n << endl;
	valid = false;
      }
      if(c1n < 0) {
	cerr << "ERROR: c1n < 0: " << c1n << endl;
	valid = false;
      }
      if(dc.c0p < 0) {
	cerr << "ERROR: c0p < 0: " << dc.c0p << endl;
	valid = false;
      }
      if(dc.c1p < 0) {
	cerr << "ERROR: c1p < 0: " << dc.c1p << endl;
	valid = false;
      }
      
      if((dc.c0p + c0n) != dt->c0) {
	cerr << "ERROR: c0p + c0n != c0 : " << dc.c0p << " + " << c0n << " != " << dt->c0 << endl;
	valid = false;
      }
      
      if((dc.c1p + c1n) != dt->c1) {
	cerr << "ERROR: c1p + c1n != c1 : " << dc.c1p << " + " << c1n << " != " << dt->c1 << endl;
	valid = false;
      }
    }
    
    // validate counters against samples
    
    {
      sparse_hash_map<int, DecisionCounts> computedDCs;
      unsigned int computedC0, computedC1;
      pair<CodeRankType, int> computedMinValidRank;
      
      // FIXME: validate minValidRank
      
      TreeSampleWalker sw(dt);
      computeDecisionCounters(dt, sw, computedDCs, computedC0, computedC1 ,computedMinValidRank);
      if(computedC0 != dt->c0) {
	cerr << "ERROR: c0 != computedC0 : " << dt->c0 << " != " << computedC0 << endl;
	valid = false;
      }
      if(computedC1 != dt->c1) {
	cerr << "ERROR: c1 != computedC1 : " << dt->c1 << " != " << computedC1 << endl;
	valid = false;
      }
      if(!compareDCs(dt->decisionCountMap, computedDCs, dt, "stored", "computed")) {
	cerr << "bang bang bang" << endl;
	cerr << "dt = " << (long) dt << endl;
	cerr << "minValidRank = " << dt->minValidRank.first << " , " << dt->minValidRank.second << endl;
	cerr << dt->decisionCountMap.size() << " DCs in stored" << endl;
	cerr << computedDCs.size() << " DCs in computed" << endl;
	cerr << "split code was " << dt->code << endl;
	valid = false;
      }
    }
    
    if(ni) {
      
      sparse_hash_map<int, DecisionCounts>::const_iterator itDC;
      itDC = dt->decisionCountMap.find(dt->code);
      if(itDC != dt->decisionCountMap.end()) {
	const DecisionCounts& dc = itDC->second;
	
	const unsigned int c0n = dt->c0 - dc.c0p;
	const unsigned int c1n = dt->c1 - dc.c1p;
	
	if(ni->negative->c0 != c0n) {
	  cerr << "ERROR: negative->c0 != c0n : " << ni->negative->c0 << " != " << c0n << endl;
	  valid = false;
	}
	if(ni->negative->c1 != c1n) {
	  cerr << "ERROR: negative->c1 != c1n : " << ni->negative->c1 << " != " << c1n << endl;
	valid = false;
	}
	if(ni->positive->c0 != dc.c0p) {
	  cerr << "ERROR: positive->c0 != c0p : " << ni->positive->c0 << " != " << dc.c0p << endl;
	  valid = false;
	}
	if(ni->positive->c1 != dc.c1p) {
	  cerr << "ERROR: positive->c1 != c1p : " << ni->positive->c1 << " != " << dc.c1p << endl;
	  valid = false;
	}
	
      } else {
	cerr << "ERROR: code upon node is split not found in decisionCountMap" << endl;
	valid = false;
      }
      
      if(!valid)
	cerr << "validating negative branch now" << endl;
      if(!validateDecisionTree(ts, ni->negative))
	valid = false;
      if(!valid)
	cerr << "validating positive branch now" << endl;
      if(!validateDecisionTree(ts, ni->positive))
	valid = false;
      
      if((ni->negative->c0 + ni->positive->c0) != ni
	 ->c0) {
	cerr << "ERROR: negative->c0 + positive->c0 != c0 : " << ni->negative->c0 << " + " << ni->positive->c0 << " != " << ni->c0 << endl;
	valid = false;
      }
      if((ni->negative->c1 + ni->positive->c1) != dt->c1) {
	cerr << "ERROR: negative->c1 + positive->c1 != c1 : " << ni->negative->c1 << " + " << ni->positive->c1 << " != " << ni->c1 << endl;
	valid = false;
      }
    }
    
    return valid;
  }
  
  static void updateDecisionTreeSamples(DecisionTreeNode* dt, const vector<Sample*>& batchAdd, const vector<Sample*>& batchRemove) {
    DecisionTreeInternal* ni;
    DecisionTreeLeaf* nl;
    
    dt->checkType(&ni, &nl);
    
    if(nl) {
      vector<Sample*>::const_iterator bIt;
      for(bIt = batchRemove.begin(); bIt != batchRemove.end(); ++bIt) {
	Sample* s = *bIt;
	vector<Sample*>::iterator sIt = find(nl->samples.begin(), nl->samples.end(), s);
	if(sIt != nl->samples.end())
	  nl->samples.erase(sIt);
	else
	  cerr << "ERROR: could not find sample to remove!" << endl;
      }
      nl->samples.insert(nl->samples.end(), batchAdd.begin(), batchAdd.end());
    }
    
    // FIXME: these splits will have to be done again in when walking the tree to update (split/unsplit) nodes
    // how to avoid the duplicate work?
    if(ni) {
      vector<Sample*> aN, aP;
      VectorSampleWalker swAdd(batchAdd);
      splitListAgainstCode(swAdd, ni->code, aN, aP);
      
      vector<Sample*> rN, rP;
      VectorSampleWalker swRemove(batchRemove);
      splitListAgainstCode(swRemove, ni->code, rN, rP);
      
      if(aN.size() > 0 || rN.size() > 0)
	updateDecisionTreeSamples(ni->negative, aN, rN);
      
      if(aP.size() > 0 || rP.size() > 0)
	updateDecisionTreeSamples(ni->positive, aP, rP);
      
    } 
  }
  
  static DecisionTreeNode* updateDecisionTreeNode(TreeState& ts, DecisionTreeNode* dt, const vector<Sample*>& batchAdd, const vector<Sample*>& batchRemove) {
    DecisionTreeInternal* ni;
    DecisionTreeLeaf* nl;
    
    dt->checkType(&ni, &nl);
    
    { 
      // removals
      
      vector<Sample*>::const_iterator bIt;
      int addedBefore0 = 0;
      int addedBefore1 = 0;
      for(bIt = batchRemove.begin(); bIt != batchRemove.end(); ++bIt) {
	updateDecisionCounters(dt, *bIt, addedBefore0, addedBefore1, -1);
	if((*bIt)->y >= 0.5)
	  ++addedBefore1;
	else
	  ++addedBefore0;
      }

      vector<Sample*> b0, b1;
      
      // FIXME: more efficient just to count them!
      splitListByTarget(batchRemove, b0, b1);
      
      (dt->c1) += -1 * b1.size();
      (dt->c0) += -1 * b0.size();
      
    }
    
  {
    // additions
    
    vector<Sample*>::const_iterator bIt;
    int addedBefore0 = 0;
    int addedBefore1 = 0;
    for(bIt = batchAdd.begin(); bIt != batchAdd.end(); ++bIt) {
      updateDecisionCounters(dt, *bIt, addedBefore0, addedBefore1);
      if((*bIt)->y >= 0.5)
	++addedBefore1;
      else
	++addedBefore0;
    }
    
    vector<Sample*> b0, b1;
    
    // FIXME: more efficient just to count them!
    splitListByTarget(batchAdd, b0, b1);
    
    (dt->c1) += b1.size();
    (dt->c0) += b0.size();
  }
  
  if((dt->decisionCountMap.size() < maxCodesToConsider)
     && ((dt->minValidRank.first != 0) || (dt->minValidRank.second != 0))) {
    computeDecisionCounters(dt,
			    TreeSampleWalker(dt),
			    dt->decisionCountMap,
			    dt->c0,
			    dt->c1,
			    dt->minValidRank);
  }
  
  float currentEntropy = entropyBinary(dt->c0, dt->c1);
  int minEntropyCode = findMinEntropyCode(currentEntropy, dt);
  
  bool shouldBeSplit = minEntropyCode != -1;
  
  if(nl) {
    // update leaf node
    
    if(shouldBeSplit) {
      
      // time to split
      
      DecisionTreeInternal* newInternal = makeInternal(ts, minEntropyCode, 0, 0);
      newInternal->c0 = dt->c0;
      newInternal->c1 = dt->c1;
      newInternal->minValidRank = dt->minValidRank;
      newInternal->decisionCountMap = dt->decisionCountMap;
      newInternal->id = dt->id;
      VectorSampleWalker sw(nl->samples);
      splitNode(ts, newInternal, minEntropyCode, sw);
      
      destroyDecisionTreeNode(nl);
      
      return newInternal;
    } else {
      // staying a leaf
      
      updateValue(nl);
      
      return nl;
    }
  } else {
    // update internal node

    if(!shouldBeSplit) {
      DecisionTreeLeaf* newLeaf = makeLeaf(ts, 0);
      newLeaf->id = dt->id;
      
      TreeSampleWalker sw(dt);

      insertLeafSamples(newLeaf->samples, sw);
      
      setupLeafFromSamples(newLeaf);
      
      destroyDecisionTreeNode(dt);
      
      return newLeaf;
    } else {
      
      if(minEntropyCode != ni->code) {
	TreeSampleWalker sw(dt);
	splitNode(ts, ni, minEntropyCode, sw);
      } else {
	vector<Sample*> aN, aP;
	VectorSampleWalker swAdd(batchAdd);
	
	splitListAgainstCode(swAdd, ni->code, aN, aP);
	
	vector<Sample*> rN, rP;
	VectorSampleWalker swRemove(batchRemove);
	
	splitListAgainstCode(swRemove, ni->code, rN, rP);
	
	if(aN.size() > 0 || rN.size() > 0) {
	  ni->negative = updateDecisionTreeNode(ts, ni->negative, aN, rN);
	}
	if(aP.size() > 0 || rP.size() > 0) {
	  ni->positive = updateDecisionTreeNode(ts, ni->positive, aP, rP);
	}
      }
      
      return dt;
    }
  }
  }
  
  static DecisionTreeNode* updateDecisionTree(TreeState& ts, DecisionTreeNode* dt, const vector<Sample*>& batchAdd, const vector<Sample*>& batchRemove) {
    
    for(vector<Sample*>::const_iterator it1 = batchAdd.begin(); it1 != batchAdd.end(); ++it1) {
      vector<Sample*>::const_iterator it2 = find(batchRemove.begin(), batchRemove.end(), *it1);
      if(it2 != batchRemove.end()) {
	cerr << "sample in batchAdd also in batchRemove!!!" << endl;
	exit(1);
      }
    }
    for(vector<Sample*>::const_iterator it1 = batchRemove.begin(); it1 != batchRemove.end(); ++it1) {
      vector<Sample*>::const_iterator it2 = find(batchAdd.begin(), batchAdd.end(), *it1);
      if(it2 != batchAdd.end()) {
	cerr << "sample in batchRemove also in batchAdd!!!" << endl;
	exit(1);
      }
    }
    
    updateDecisionTreeSamples(dt, batchAdd, batchRemove);
    DecisionTreeNode* n = updateDecisionTreeNode(ts, dt, batchAdd, batchRemove);
    return n;
  }
  
  static DecisionTreeNode* loadDecisionTreeNodeForForest(TreeState& ts, istream& forestS, map<long, Sample*>& sampleMap) {
    int nodeCode;
    forestS >> nodeCode;
    
    DecisionTreeNode* n = 0;
    DecisionTreeLeaf* nl = 0;
    DecisionTreeInternal* ni = 0;
    
    if(nodeCode == -1) {
      nl = makeLeaf(ts, 0);
      n = nl;
    } else {
      ni = makeInternal(ts, nodeCode, 0, 0);
      n = ni;
    }
    
    forestS >> n->id;
    forestS >> n->minValidRank.first >> n->minValidRank.second;
    forestS >> n->c0 >> n->c1;
    
    int countDC;
    forestS >> countDC;
    
    n->decisionCountMap.resize(countDC);
    
    for(int i = 0; i < countDC; ++i) {
      int code;
      forestS >> code;
      DecisionCounts dc;
      // FIXME: no need to be backwards compatible after complete deployment
      unsigned int dummy;
      forestS >> dummy >> dummy >> dc.c0p >> dc.c1p >> dc.rank;
      // not loading empty DC
      if(!(dc.c0p == 0 && dc.c1p == 0))
	n->decisionCountMap[code] = dc;
    }
    
    if(nl) {
      int countSamples;
      forestS >> countSamples;
      nl->samples.resize(countSamples);
      for(int i = 0;  i< countSamples; ++i) {
	long sampleId;
	forestS >> sampleId;
	if(sampleMap.find(sampleId) == sampleMap.end()) {
	  cerr << "unknown sample!" << endl;
	  exit(1);
	}
	nl->samples[i] = sampleMap[sampleId];
      }
      
      forestS >> nl->value;
    } else {
      ni->negative = loadDecisionTreeNodeForForest(ts, forestS, sampleMap);
      ni->positive = loadDecisionTreeNodeForForest(ts, forestS, sampleMap);
    }
    return n;
  }
  
  static void saveDecisionTreeNodeInForest(DecisionTreeNode* dt, ostream& forestS) {
    DecisionTreeInternal* ni;
    DecisionTreeLeaf* nl;
    
    forestS << dt->code << endl;
    forestS << dt->id << endl;
    forestS << dt->minValidRank.first << " " << dt->minValidRank.second << endl;
    forestS << dt->c0 << " " << dt->c1 << endl;
    forestS << dt->decisionCountMap.size() << endl;
    sparse_hash_map<int, DecisionCounts>::const_iterator dcIt;
    for(dcIt = dt->decisionCountMap.begin(); dcIt != dt->decisionCountMap.end(); ++dcIt) {
      forestS << dcIt->first << endl;
      const DecisionCounts& dc = dcIt->second;
      forestS << 0 << " " << 0 << " " << dc.c0p << " " << dc.c1p << " " << dc.rank << endl;
    }
    
    dt->checkType(&ni, &nl);
    
    if(nl) {
      forestS << nl->samples.size() << endl;
      vector<Sample*>::const_iterator sIt;
      for(sIt = nl->samples.begin(); sIt != nl->samples.end(); ++sIt)
	forestS << (long)(*sIt) << endl;
    }
    
    if(nl) {
      forestS << nl->value << endl;
    } else {
      saveDecisionTreeNodeInForest(ni->negative, forestS);
      saveDecisionTreeNodeInForest(ni->positive, forestS);
    }
  }
  
  static void saveDecisionTreeInForest(TreeState& ts, DecisionTreeNode* dt, ostream& forestS) {
    saveDecisionTreeNodeInForest(dt, forestS);
  }
  
  void outputDecisionTree(TreeState& ts, DecisionTreeNode* dt, ostream& outS) {
    DecisionTreeInternal* ni;
    DecisionTreeLeaf* nl;
    
    if(dt->checkType(&ni, &nl)) {
      outS << nl->value;
    } else {
      outS << "[";
      outS << ni->code << ",";
      outputDecisionTree(ts, ni->negative, outS);
      outS << ",";
      outputDecisionTree(ts, ni->positive, outS);
      outS << "]";
    }
  }
  
  static void loadRandomForest(TreeState& ts, istream& forestS, vector<DecisionTreeNode*>& forest, map<string, Sample*>& samples) {
    forestS >> ts.seed;
    int nTrees;
    forestS >> nTrees;
    int nSamples;
    forestS >> nSamples;
    map<long, Sample*> sampleMap;
    for(int i = 0; i < nSamples; ++i) {
      long sampleId;
      forestS >> sampleId;
      Sample* s = new Sample();
      forestS >> s->suid;
      forestS >> s->y;
      int countSampleCodes;
      forestS >> countSampleCodes;
      for(int j = 0; j < countSampleCodes; ++j) {
	int code;
	float value;
	forestS >> code >> value;
	s->xCodes[code] = value;
      }
      sampleMap[sampleId] = s;
      samples[s->suid] = s;
    }
    for(int i = 0; i < nTrees; ++i) {
      forest.push_back(loadDecisionTreeNodeForForest(ts, forestS, sampleMap));     
    }
  }
  
  static float evaluateSampleAgainstDecisionTree(TreeState& ts, Sample* s, DecisionTreeNode* dt) {
    DecisionTreeNode* dtn = dt;
    
    DecisionTreeInternal* ni;
    DecisionTreeLeaf* nl;
    
    while(!dtn->checkType(&ni, &nl)) {
      float y;
      
      map<int, float>::const_iterator xCodeIt = s->xCodes.find(dtn->code);
      if(xCodeIt != s->xCodes.end())
	y = xCodeIt->second;
      else
	y = 0;
      
      if(y >= 0.5) {
	dtn = ni->positive;
      } else {
	dtn = ni->negative;
      }
    }
    
    return nl->value;
  }

  static bool sampleInTree(const Sample* sp, int t) {
    char s[64];
    sprintf(s, "%d%s", t, sp->suid.c_str());
    uint32_t out;
    MurmurHash3_x86_32(s, strlen(s), 42, &out);
    return (out % 3) < 2; // 2 in 3 chance
  }

  class MapSampleWalker : public SampleWalker {
  private:
    map<string, Sample*>::const_iterator itCurr;
    map<string, Sample*>::const_iterator itEnd;
  public:
    MapSampleWalker(const map<string, Sample*>& sm) :
      itCurr(sm.begin()), itEnd(sm.end()) {
    }
    virtual bool stillSome(void) const {
      return itCurr != itEnd;
    }
    virtual Sample* get(void) {
      Sample* ret = itCurr->second;
      ++itCurr;
      return ret;
    }
  };
  
  class Forest {
  private:
    map<string, Sample*> samples;
    map<string, Sample*> toAdd;
    map<string, Sample*> toRemove;
    vector<DecisionTreeNode*> forest;
    bool changesToCommit;
    TreeState ts;
  public:
    Forest(istream& forestS) {
      map<long, Sample*> sampleMap;
      loadRandomForest(ts, forestS, forest, samples);
      changesToCommit = false;
    }

    Forest(int nTrees) {
      for(int i=0; i < nTrees; ++i)
	forest.push_back(emptyDecisionTree(ts));
      changesToCommit = false;
    }

    ~Forest(void) {
      for(vector<DecisionTreeNode*>::iterator itTree = forest.begin();
	  itTree != forest.end();
	  ++itTree) {
	destroyDecisionTreeNode(*itTree);
      }
      map<string, Sample*>::iterator itAdd;
      for(itAdd = toAdd.begin(); itAdd != toAdd.end(); ++itAdd) {
	delete itAdd->second;
      }
      map<string, Sample*>::iterator itMap;
      for(itMap = samples.begin(); itMap != samples.end(); ++itMap) {
	delete itMap->second;
      }
    }

    bool add(Sample* s) {
      changesToCommit = true;
      map<string, Sample*>::iterator itAdd = toAdd.find(s->suid);
      
      bool added = false;
      if(itAdd != toAdd.end()) {
	delete itAdd->second;
      } else {
	added = true;
	map<string, Sample*>::iterator itRemove = toRemove.find(s->suid);
	map<string, Sample*>::iterator itMap = samples.find(s->suid);
	
	if(itRemove == toRemove.end()) {
	  // no remove record
	  if(itMap != samples.end()) {
	    toRemove[itMap->first] = itMap->second;
	  }
	}
      }
      
      toAdd[s->suid] = s;
      return added;
    }

    bool remove(const char* sId) {
      map<string, Sample*>::iterator itAdd = toAdd.find(sId);
      if(itAdd != toAdd.end()) {
	delete itAdd->second;
	toAdd.erase(itAdd);
	changesToCommit = true;
	return true;
      }
      
      map<string, Sample*>::iterator itRemove = toRemove.find(sId);
      if(itRemove != toRemove.end())
	return false;
      
      map<string, Sample*>::iterator itMap = samples.find(sId);
      if(itMap == samples.end())
	return false;
      
      changesToCommit = true;
      
      toRemove[sId] = itMap->second;
      
      return true;
    }

    void commit(void) {
      if(!changesToCommit)
	return;
      
      map<string, Sample*>::iterator sIt;
      
      int treeId = 0;
      for(vector<DecisionTreeNode*>::iterator itTree = forest.begin();
	  itTree != forest.end();
	  ++itTree, ++treeId) {
	vector<Sample*> treeAdd, treeRemove;
	for(sIt = toRemove.begin(); sIt != toRemove.end(); ++sIt) {
	  if(sampleInTree(sIt->second, treeId))
	    treeRemove.push_back(sIt->second);
	}
	for(sIt = toAdd.begin(); sIt != toAdd.end(); ++sIt) {
	  if(sampleInTree(sIt->second, treeId))
	    treeAdd.push_back(sIt->second);
	}
	*itTree = updateDecisionTree(ts, *itTree, treeAdd, treeRemove);
      }
      
      for(sIt = toRemove.begin(); sIt != toRemove.end(); ++sIt) {
	delete sIt->second;
	samples.erase(sIt->first);
      }
      for(sIt = toAdd.begin(); sIt != toAdd.end(); ++sIt)
	samples[sIt->first] = sIt->second;
      
      toAdd.clear();
      toRemove.clear();
      changesToCommit = false;
    }

    void asJSON(ostream& outS) {
      commit();
      outS << "[";
      for(vector<DecisionTreeNode*>::iterator itTree = forest.begin();
	  itTree != forest.end();
	  ++itTree) {
	if(itTree != forest.begin())
	  outS << ",";
	outputDecisionTree(ts, *itTree, outS);
      }
      outS << "]";
    }

    bool save(ostream& outS) {
      commit();
      outS << ts.seed << endl;
      outS << forest.size() << endl;
      
      outS << samples.size() << endl;
      map<string, Sample*>::const_iterator sIt;
      for(sIt = samples.begin(); sIt != samples.end(); ++sIt) {
	const Sample* s = sIt->second;
	outS << (long)s << endl;
	outS << s->suid << endl;
	outS << s->y << endl;
	map<int, float>::const_iterator codeIt;
	outS << s->xCodes.size() << endl;
	for(codeIt = s->xCodes.begin(); codeIt != s->xCodes.end(); ++codeIt)
	  outS << codeIt->first << " " << codeIt->second << endl;
      }
      
      for(vector<DecisionTreeNode*>::iterator itTree = forest.begin();
	  itTree != forest.end();
	  ++itTree) {
	saveDecisionTreeInForest(ts, *itTree, outS);
      }
      return true;
    }

    float classify(Sample* s) {
      commit();
      double v = 0;
      for(vector<DecisionTreeNode*>::iterator itTree = forest.begin();
	  itTree != forest.end();
	  ++itTree) {
	double dv = evaluateSampleAgainstDecisionTree(ts, s, *itTree);
	v += dv;
      }
      return v / forest.size();
    }

    bool validate(void) {
      for(vector<DecisionTreeNode*>::iterator itTree = forest.begin();
	  itTree != forest.end();
	  ++itTree) {
	if(!validateDecisionTree(ts, *itTree))
	  return false;
      }
      return true;
    }

    SampleWalker* getSamples(void) {
      commit();
      return new MapSampleWalker(samples);
    }
  };
  
  /* visible outside module */

  Forest* create(int nTrees) {
    return new Forest(nTrees);
  }

  void destroy(Forest* rf) {
    delete rf;
  }

  Forest* load(istream& forestS) {
    return new Forest(forestS);
  }

  bool save(Forest* rf, ostream& outS) {
    return rf->save(outS);
  }

  void asJSON(Forest* rf, ostream& outS) {
    rf->asJSON(outS);
  }

  bool add(Forest* rf, Sample* s) {
    return rf->add(s);
  }

  bool remove(Forest* rf, const char* sId) {
    return rf->remove(sId);
  }

  void commit(Forest* rf) {
    rf->commit();
  }

  float classify(Forest* rf, Sample* s) {
    return rf->classify(s);
  }

  bool validate(Forest* rf) {
    return rf->validate();
  }

  SampleWalker* getSamples(Forest* rf) {
    return rf->getSamples();
  }
}
