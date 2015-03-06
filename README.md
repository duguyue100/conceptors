# Conceptors

This repo aims to provide a python implementation of H. Jaeger's conceptor network. The original work can be found from [here](http://minds.jacobs-university.de/conceptors).

## Requirements

This package is developed under following system and packages:

+ Ubuntu 14.04
+ python 2.7.9
+ numpy 1.9.1
+ scipy 0.14.0
+ matplotlib u'1.4.0'

These packages can be found and installed by using Anaconda 2.1.0.

## Updates

+ Structure sketch [DONE]
+ First working version [2015-02-26]
+ Refined version of conceptor network [2015-02-27]
+ Autoconceptors [UNTESTED 2015-03-05]
+ Random Feature Conceptors [TODO]
+ Conceptor visualization [2015-02-27] [A test is added in the code, not in a function, visualization seems fine]
+ Conceptor I/O: saving and loding in file [TODO]
+ Now the naive conceptor can accept multidimensional input instead of 1-d input [2014-02-28]
+ Japanese Vowels recognition test [TODO]
+ Conceptors network fixed [2015-03-05]
+ Update logical operators (AND, OR, NOT) [UNTESTED, 2015-03-05]

## Notes

+ Based on my reading so far, there is really no training for the network. The readout weights and target weights are calculated analytically.

+ I did a test of using two patterns, then it's able to recall barely. (this implementation is purly based on the `test.m` of the file) [2015-02-26] [This is not true anymore, the tests I ran is a messy recall, it's supposed to be like that way. the new version refined the test results]

+ The final objective is to realize _Random Feature Conceptor_ network, this network gives a biological plausible solution to realize conceptors.

+ I was testing the conceptor network and this time, I used a 2-d signal instead of 1-d. The signal is made up by two sine waves that have different frequencies. Turns out the network output can almost match the first dimension, and it's failed to reconstruct the second dimension. (This problem is fixed, I mis-calculated one equation in the updating function).

+ The reconstruction of autoconceptor is not as expected.

## Contacts
Yuhuang Hu  
Advanced Robotic Lab  
Department of Artificial Intelligence  
Faculty of Computer Science & IT  
University of Malaya  
Email:duguyue100@gmail.com