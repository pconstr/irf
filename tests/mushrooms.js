#!/usr/bin/env node

'use strict';

var fs = require('fs');
var irf = require('../index.js');
var carrier = require('carrier');

var rf = new irf.IRF(99);
var classValues = {'1': 0, '2':1 };

console.log('reading...');

var c = carrier.carry(fs.createReadStream('mushrooms'));

var instanceID = 0;
var testing = [];

function test(rf, testing) {
  var counts = [[0, 0], [0, 0]];
  testing.forEach(function(instance) {
    var y = rf.classify(instance[1]) >= 0.5 ? 1 : 0;
    counts[y][instance[2]]++
  });
  return counts;
}

function printCounts(counts) {
  var total = counts[0][0]+ counts[0][1]+ counts[1][0]+ counts[1][1]
  console.log("total =", total);
  console.log("  correct negatives:", counts[0][0]);
  console.log("  false   negatives:", counts[1][0]);
  console.log("  correct positives: ", counts[1][1]);
  console.log("  false   positives: ", counts[0][1]);
}

c.on('line', function(line) {
  var data = line.split(' ');

  var y = classValues[data[0]];

  var features = {};
  data.slice(1).forEach(function(t) {
    if(t !== '') {
      var elems = t.split(':');
      features[elems[0]] = elems[1];
    }
  });

  var iid = String(instanceID);
  if(instanceID % 5 <= 3) {
    testing.push([iid, features, y]);
  } else {
    rf.add(iid, features, y);
  }

  instanceID++;
});

c.on('end', function() {
  console.log('learning...');
  rf.commit();
  console.log('classifying...');
  var counts = test(rf, testing);
  printCounts(counts);
  console.log('saving...');
  fs.writeFileSync('mushrooms.rf', rf.toBuffer());
  console.log('loading...');
  rf = new irf.IRF(fs.readFileSync('mushrooms.rf'));
  console.log('classifying...');
  var counts = test(rf, testing);
  printCounts(counts);
  console.log('.');
});
