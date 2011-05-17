#!/usr/bin/python

import irf

f = irf.IRF(99) # create forest of 99 trees

f.add('1', {1:1, 3:1, 5:1}, 0) # add a sample identified as '1' with the given feature values, classified as 0 
f.add('2', {1:0, 3:0, 4:1}, 0) # features are stored sparsely, when a value is not given it will be taken as 0
f.add('3', {2:0, 3:0, 5:0}, 0) # but 0s can also be given explicitly
f.add('4', {1:0, 2:0, 3:0, 5:0}, 0)
f.add('5', {1:1, 2:1, 3:1, 4:1, 5:1}, 0)
f.add('6', {2:1, 3:1, 4:0}, 1)
f.add('7', {1:1, 2:1, 3:0, 4:1}, 1)
f.add('8', {1:0, 2:0, 3:0, 4:1, 5:1}, 1)
f.add('9', {1:0, 2:0, 3:1, 4:0, 5:1}, 1)
f.add('10', {1:0, 3:1, 4:1, 5:0, 6:1}, 1)
f.add('11', {1:1, 3:0, 5:1, 7:1}, 1)
f.add('12', {1:0, 3:1, 5:1, 8:1}, 1)
f.add('13', {1:0, 3:0, 4:1, 7:1}, 0)
f.add('14', {1:1, 3:1, 5:1, 8:1}, 0)
f.add('15', {1:1, 4:1, 8:1}, 1)

y = f.classify({1:1, 2:1, 5:1}); print y, int(round(y)) # classify feature vector, round to nearest to get class
y = f.classify({3:1, 2:1, 5:1}); print y, int(round(y))
y = f.classify({1:1, 3:1, 5:1}); print y, int(round(y))
y = f.classify({2:1, 5:1});      print y, int(round(y))

f.save('simple.rf') # save forest to file

f = irf.load('simple.rf') # load forest from file

f.remove('8') # remove a sample
f.add('8', {1:0, 2:0, 3:0, 4:0, 5:1}, 0) # and add it again with new values

y = f.classify({1:1, 2:1, 5:1}); print y, int(round(y)) # the forest will be lazily updated before classification
y = f.classify({3:1, 2:1, 5:1}); print y, int(round(y)) # it is also possible for force it with commit()
y = f.classify({1:1, 3:1, 5:1}); print y, int(round(y))
y = f.classify({2:1, 5:1});      print y, int(round(y))

for (sId, x, y) in f.samples(): # iterate through samples in the forest, in lexicographic ID order
    print sId, x, y # and print them
