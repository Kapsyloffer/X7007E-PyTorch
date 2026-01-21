Transformer, 100,000 items, 100 test
Results:
#1: 24 / 44 (45% reduction)

#2: 8 / 44 (82% reduction)

#3: 37 / 44 (16% reduction)

AVG: 47.66% reduction!!

BUGG: offset maxxar på 200, borde maxxa på takt + drift - size
Overfitting? Stämmer alla constraints?

PointerNetwork

Reduction med 85% from the get-go då vi overfittar, datan ser fine ut, inga constraints brutna.

PointerNetwork 100 test, 500000 training, unseeded randomized input

#1: 83/367 (77% reduction)

#2 75/346 (78% reduction)

#3 64/340 (82% reduction)

Avg: 78.96% reduction

PointNetwork 10 test, 500000 training, blablabla

#1: 1/25 (96%)

#2: 0/34 (100%)

#3: 2(20 (90%)

AVG: 95.3% reduction

Notes: Allt ser ut som det ska, constraints respekteras, datan är randomizad, unseeded, have we done it?
