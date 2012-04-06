Incremental Random Forest
=========================

An implementation in C++ (with [node.js](http://nodejs.org) and Python bindings) of a variant of [Leo Breiman's Random Forests](http://stat-www.berkeley.edu/users/breiman/RandomForests/cc_home.htm)

The forest is maintained incrementally as samples are added or removed - rather than fully rebuilt from scratch every time - to save effort.

It is not a streaming implementation, all the samples are stored and will be reseen when required to recursively rebuild invalidated subtrees. The effort to update each individual tree can vary substantially but the overall effort to update the forest is averaged across the trees so tends not to vary so much.

IRF is licensed under the MIT license.

Features and limitations
------------------------

* Sparse feature vectors
* Samples can be added, removed and changed
* Learning can be performed lazily or initiated explicitly
* The forest can be serialized to JSON for transmission/storage
* The forest needs to fit fully in RAM, performance suffers dramatically when swapping
* Currently only binary classification - 0 or 1. The classifier estimates the probability of belonging to class 1, as a float from 0 to 1
* Currently only binary features: y >= 0.5 is considered 1, otherwise 0

Node.js setup
-----
`npm install irf`

Node.js usage
-------------

```javascript
var irf = require('irf');

var f = new irf.IRF(99); // create forest of 99 trees

f.add('1', {1:1, 3:1, 5:1}, 0); // add a sample identified as '1' with the given feature values, classified as 0
f.add('2', {1:0, 3:0, 4:1}, 0); // features are stored sparsely, when a value is not given it will be taken as 0
f.add('3', {2:0, 3:0, 5:0}, 0); // but 0s can also be given explicitly
// ...

var y = f.classify({1:1, 3:1, 5:1}); // classify feature vector
                                     // the forest will be lazily updated before classification
f.commit();                          // but you can force an update at any time
                                     // you get a probability estimate from 0 to 1 for belong to class 1
var c = Math.round(y);               // round to nearest to get class (0 or 1)

f.remove('8'); // remove a sample
f.add('8', {1:0, 2:0, 3:0, 4:0, 5:1}, 0); // and add it again with new values

console.log(f.asJSON()); // serialize to json (for classification, not suitable for incremental update)

f.each(function(suid, features, y) {
    // ...
});

var b = f.toBuffer();    // serialize (complete) to buffer
var f2 = new irf.IRF(b); // construct from buffer contents
```

Python setup
-----
    cd irf
    python setup.py install

Python usage
------------

```python
import irf

f = irf.IRF(99) # create forest of 99 trees

f.add('1', {1:1, 3:1, 5:1}, 0) # add a sample identified as '1' with the given feature values, classified as 0
f.add('2', {1:0, 3:0, 4:1}, 0) # features are stored sparsely, when a value is not given it will be taken as 0
f.add('3', {2:0, 3:0, 5:0}, 0) # but 0s can also be given explicitly
# ...

y = f.classify({1:1, 2:1, 5:1}); print y, int(round(y)) # classify feature vector, round to nearest to get class

f.save('simple.rf') # save forest to file

f = irf.load('simple.rf') # load forest from file

f.remove('8') # remove a sample
f.add('8', {1:0, 2:0, 3:0, 4:0, 5:1}, 0) # and add it again with new values

y = f.classify({1:1, 2:1, 5:1}); print y, int(round(y)) # the forest will be lazily updated before classification
# f.commit() # but you can force it

for (sId, x, y) in f.samples(): # iterate through samples in the forest, in lexicographic ID order
    print sId, x, y # and print them
```

C++ usage
---------

_to be written_

Dependencies
------------

System:

* STL

Included:

* MurmurHash3 (from [smhasher](http://code.google.com/p/smhasher/))

External:

* [google sparse hash](http://goog-sparsehash.sourceforge.net/)


Tests
-----

* simple.py - trivial made up data to illustrate how to use the API
* mushrooms.js, mushrooms.py - using the [mushrooms dataset](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#mushrooms) collected by [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/) from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/)
