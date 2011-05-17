#!/usr/bin/python

import irf

def printCounts(counts):
    total = counts[0][0]+ counts[0][1]+ counts[1][0]+ counts[1][1]
    print "total = ", total
    print "  correct negatives: ", counts[0][0]
    print "  false   negatives: ", counts[1][0]
    print "  correct positives: ", counts[1][1]
    print "  false   positives: ", counts[0][1]

def test(rf, testing):
    counts = [[0,0],[0,0]]
    for instance in testing:
        c = int(rf.classify(instance[1]) >= 0.5)
        counts[c][instance[2]] = counts[c][instance[2]] + 1
    return counts

def main():
    rf = irf.IRF(99)

    print 'reading...'
    f = open('mushrooms')
    instanceID = 0
    testing = []
    classValues = {'1':0, '2':1}
    for rawL in f.readlines():
        l = rawL.strip()
        values = l.split(' ')
        c = classValues[values[0]]
        features = {}
        for kCv in values[1:]:
            k, v = kCv.split(':')
            features[int(k)] = int(v)
        instance = (str(instanceID), features, c)
        if instanceID % 5 <= 3:
            testing.append(instance)
        else:
            rf.add(*instance)
        instanceID = instanceID + 1

    print 'learning...'
    rf.commit()

    print 'classifying...'
    counts = test(rf, testing)
    printCounts(counts)

    print 'saving...'
    rf.save('mushrooms.rf')

    print 'loading...'
    rf = irf.load('mushrooms.rf')

    print 'classifying...'
    counts = test(rf, testing)
    printCounts(counts)

    print '.'

if __name__ == "__main__":
    main()
