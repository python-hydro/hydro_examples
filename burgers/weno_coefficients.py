import numpy


# Coefficients of order r=2
# On smooth solutions this should converge with order r=3
C_2 = numpy.array([ 1,  2 ]) / 3
a_2 = numpy.array([
                   [ 3, -1],
                   [ 1,  1],
                  ]) / 2
sigma_2 = numpy.array([
                        [
                          [ 1,  0],
                          [-2,  1]
                        ],
                        [
                          [ 1,  0],
                          [-2,  1]
                        ]
                      ])

# Coefficients of order r=3
# On smooth solutions this should converge with order r=5
C_3 = numpy.array([ 1,  6,  3 ]) / 10
a_3 = numpy.array([
                   [ 11,  -7,   2],
                   [  2,   5,  -1],
                   [ -1,   5,   2],
                  ]) / 6
sigma_3 = numpy.array([
                        [
                          [ 10,   0,   0],
                          [-31,  25,   0],
                          [ 11, -19,   4]
                        ],
                        [
                          [  4,   0,   0],
                          [-13,  13,   0],
                          [  5, -13,   4]
                        ],
                        [
                          [  4,   0,   0],
                          [-19,  25,   0],
                          [ 11, -31,  10]
                        ]
                      ]) / 3

# Coefficients of order r=4
# On smooth solutions this should converge with order r=7
C_4 = numpy.array([  1,  12,  18,   4 ]) / 35
a_4 = numpy.array([
                   [ 25, -23,  13,  -3],
                   [  3,  13,  -5,   1],
                   [ -1,   7,   7,  -1],
                   [  1,  -5,  13,   3],
                  ]) / 12
sigma_4 = numpy.array([
                        [
                          [  2107,      0,      0,      0],
                          [ -9402,  11003,      0,      0],
                          [  7042, -17246,   7043,      0],
                          [ -1854,   4642,  -3882,    547]
                        ],
                        [
                          [   547,      0,      0,      0],
                          [ -2522,   3443,      0,      0],
                          [  1922,  -5966,   2843,      0],
                          [  -494,   1602,  -1642,    267]
                        ],
                        [
                          [   267,      0,      0,      0],
                          [ -1642,   2843,      0,      0],
                          [  1602,  -5966,   3443,      0],
                          [  -494,   1922,  -2522,    547]
                        ],
                        [
                          [   547,      0,      0,      0],
                          [ -3882,   7043,      0,      0],
                          [  4642, -17246,  11003,      0],
                          [ -1854,   7042,  -9402,   2107]
                        ]
                      ]) / 240

# Coefficients of order r=5
# On smooth solutions this should converge with order r=9
C_5 = numpy.array([  1,  20,  60,  40,   5 ]) / 126
a_5 = numpy.array([
                   [ 137, -163,  137,  -63,   12],
                   [  12,   77,  -43,   17,   -3],
                   [  -3,   27,   47,  -13,    2],
                   [   2,  -13,   47,   27,   -3],
                   [  -3,   17,  -43,   77,   12],
                  ]) / 60
sigma_5 = numpy.array([
                        [
                          [  107918,        0,        0,        0,        0],
                          [ -649501,  1020563,        0,        0,        0],
                          [  758823, -2462076,  1521393,        0,        0],
                          [ -411487,  1358458, -1704396,   482963,        0],
                          [   86329,  -288007,   364863,  -208501,    22658]
                        ],
                        [
                          [   22658,        0,        0,        0,        0],
                          [ -140251,   242723,        0,        0,        0],
                          [  165153,  -611976,   406293,        0,        0],
                          [  -88297,   337018,  -464976,   138563,        0],
                          [   18079,   -70237,    99213,   -60871,     6908]
                        ],
                        [
                          [    6908,        0,        0,        0,        0],
                          [  -51001,   104963,        0,        0,        0],
                          [   67923,  -299076,   231153,        0,        0],
                          [  -38947,   179098,  -299076,   104963,        0],
                          [    8209,   -38947,    67923,   -51001,     6908]
                        ],
                        [
                          [    6908,        0,        0,        0,        0],
                          [  -60871,   138563,        0,        0,        0],
                          [   99213,  -464976,   406293,        0,        0],
                          [  -70237,   337018,  -611976,   242723,        0],
                          [   18079,   -88297,   165153,  -140251,    22658]
                        ],
                        [
                          [   22658,        0,        0,        0,        0],
                          [ -208501,   482963,        0,        0,        0],
                          [  364863, -1704396,  1521393,        0,        0],
                          [ -288007,  1358458, -2462076,  1020563,        0],
                          [   86329,  -411487,   758823,  -649501,   107918]
                        ]
                      ]) / 5040

# Coefficients of order r=6
# On smooth solutions this should converge with order r=11
C_6 = numpy.array([   1,   30,  150,  200,   75,    6 ]) / 462
a_6 = numpy.array([
                   [ 147, -213,  237, -163,   62,  -10],
                   [  10,   87,  -63,   37,  -13,    2],
                   [  -2,   22,   57,  -23,    7,   -1],
                   [   1,   -8,   37,   37,   -8,    1],
                   [  -1,    7,  -23,   57,   22,   -2],
                   [   2,  -13,   37,  -63,   87,   10],
                  ]) / 60
sigma_6 = numpy.array([
                        [
                          [   6150211,          0,          0,          0,          0,          0],
                          [ -47460464,   94851237,          0,          0,          0,          0],
                          [  76206736, -311771244,  260445372,          0,          0,          0],
                          [ -63394124,  262901672, -444003904,  190757572,          0,          0],
                          [  27060170, -113206788,  192596472, -166461044,   36480687,          0],
                          [  -4712740,   19834350,  -33918804,   29442256,  -12950184,    1152561]
                        ],
                        [
                          [   1152561,          0,          0,          0,          0,          0],
                          [  -9117992,   19365967,          0,          0,          0,          0],
                          [  14742480,  -65224244,   56662212,          0,          0,          0],
                          [ -12183636,   55053752,  -97838784,   43093692,          0,          0],
                          [   5134574,  -23510468,   42405032,  -37913324,    8449957,          0],
                          [   -880548,    4067018,   -7408908,    6694608,   -3015728,     271779]
                        ],
                        [
                          [    271779,          0,          0,          0,          0,          0],
                          [  -2380800,    5653317,          0,          0,          0,          0],
                          [   4086352,  -20427884,   19510972,          0,          0,          0],
                          [  -3462252,   17905032,  -35817664,   17195652,          0,          0],
                          [   1458762,   -7727988,   15929912,  -15880404,    3824847,          0],
                          [   -245620,    1325006,   -2792660,    2863984,   -1429976,     139633]
                        ],
                        [
                          [    139633,          0,          0,          0,          0,          0],
                          [  -1429976,    3824847,          0,          0,          0,          0],
                          [   2863984,  -15880404,   17195652,          0,          0,          0],
                          [  -2792660,   15929912,  -35817664,   19510972,          0,          0],
                          [   1325006,   -7727988,   17905032,  -20427884,    5653317,          0],
                          [   -245620,    1458762,   -3462252,    4086352,   -2380800,     271779]
                        ],
                        [
                          [    271779,          0,          0,          0,          0,          0],
                          [  -3015728,    8449957,          0,          0,          0,          0],
                          [   6694608,  -37913324,   43093692,          0,          0,          0],
                          [  -7408908,   42405032,  -97838784,   56662212,          0,          0],
                          [   4067018,  -23510468,   55053752,  -65224244,   19365967,          0],
                          [   -880548,    5134574,  -12183636,   14742480,   -9117992,    1152561]
                        ],
                        [
                          [   1152561,          0,          0,          0,          0,          0],
                          [ -12950184,   36480687,          0,          0,          0,          0],
                          [  29442256, -166461044,  190757572,          0,          0,          0],
                          [ -33918804,  192596472, -444003904,  260445372,          0,          0],
                          [  19834350, -113206788,  262901672, -311771244,   94851237,          0],
                          [  -4712740,   27060170,  -63394124,   76206736,  -47460464,    6150211]
                        ]
                      ]) / 120960

# Coefficients of order r=7
# On smooth solutions this should converge with order r=13
C_7 = numpy.array([   1,   42,  315,  700,  525,  126,    7 ]) / 1716
a_7 = numpy.array([
                   [ 1089, -1851,  2559, -2341,  1334,  -430,    60],
                   [   60,   669,  -591,   459,  -241,    74,   -10],
                   [  -10,   130,   459,  -241,   109,   -31,     4],
                   [    4,   -38,   214,   319,  -101,    25,    -3],
                   [   -3,    25,  -101,   319,   214,   -38,     4],
                   [    4,   -31,   109,  -241,   459,   130,   -10],
                   [  -10,    74,  -241,   459,  -591,   669,    60],
                  ]) / 420
sigma_7 = numpy.array([
                        [
                          [    7177657304,              0,              0,              0,              0,              0,              0],
                          [  -68289277071,   166930543737,              0,              0,              0,              0,              0],
                          [  140425750893,  -698497961463,   739478564460,              0,              0,              0,              0],
                          [ -158581758572,   797280592452, -1701893556420,   985137198380,              0,              0,              0],
                          [  102951716988,  -521329653333,  1119254208255, -1301580166020,   431418789360,              0,              0],
                          [  -36253275645,   184521097818,  -397822832973,   464200620612,  -308564463663,    55294430841,              0],
                          [    5391528799,   -27545885877,    59577262788,   -69700128812,    46430779053,   -16670007831,     1258225940]
                        ],
                        [
                          [    1258225940,              0,              0,              0,              0,              0,              0],
                          [  -12223634361,    31090026771,              0,              0,              0,              0,              0],
                          [   25299603603,  -132164397513,   143344579860,              0,              0,              0,              0],
                          [  -28498553012,   151212114012,  -332861569020,   195601143380,              0,              0,              0],
                          [   18375686988,   -98508059523,   219064013505,  -259838403420,    86959466460,              0,              0],
                          [   -6414710427,    34632585198,   -77574968883,    92646554652,   -62392325913,    11250068787,              0],
                          [     945155329,    -5128661355,    11548158588,   -13862429972,     9380155443,    -3397272201,      257447084]
                        ],
                        [
                          [     257447084,              0,              0,              0,              0,              0,              0],
                          [   -2659103847,     7257045753,              0,              0,              0,              0,              0],
                          [    5684116173,   -32164185663,    36922302360,              0,              0,              0,              0],
                          [   -6473137292,    37531128132,   -88597133220,    54531707180,              0,              0,              0],
                          [    4158865908,   -24530177853,    59045150655,   -74236325220,    25788772260,              0,              0],
                          [   -1432622085,     8555779674,   -20891234853,    26694456132,   -18869146983,     3510366201,              0],
                          [     206986975,    -1247531949,     3078682188,    -3982402892,     2854088973,    -1077964287,       84070496]
                        ],
                        [
                          [      84070496,              0,              0,              0,              0,              0,              0],
                          [    -969999969,     2927992563,              0,              0,              0,              0,              0],
                          [    2283428883,   -14296379553,    18133963560,              0,              0,              0,              0],
                          [   -2806252532,    18083339772,   -47431870620,    32154783380,              0,              0,              0],
                          [    1902531828,   -12546315963,    33820678305,   -47431870620,    18133963560,              0,              0],
                          [    -676871859,     4550242446,   -12546315963,    18083339772,   -14296379553,     2927992563,              0],
                          [      99022657,     -676871859,     1902531828,    -2806252532,     2283428883,     -969999969,       84070496]
                        ],
                        [
                          [      84070496,              0,              0,              0,              0,              0,              0],
                          [   -1077964287,     3510366201,              0,              0,              0,              0,              0],
                          [    2854088973,   -18869146983,    25788772260,              0,              0,              0,              0],
                          [   -3982402892,    26694456132,   -74236325220,    54531707180,              0,              0,              0],
                          [    3078682188,   -20891234853,    59045150655,   -88597133220,    36922302360,              0,              0],
                          [   -1247531949,     8555779674,   -24530177853,    37531128132,   -32164185663,     7257045753,              0],
                          [     206986975,    -1432622085,     4158865908,    -6473137292,     5684116173,    -2659103847,      257447084]
                        ],
                        [
                          [     257447084,              0,              0,              0,              0,              0,              0],
                          [   -3397272201,    11250068787,              0,              0,              0,              0,              0],
                          [    9380155443,   -62392325913,    86959466460,              0,              0,              0,              0],
                          [  -13862429972,    92646554652,  -259838403420,   195601143380,              0,              0,              0],
                          [   11548158588,   -77574968883,   219064013505,  -332861569020,   143344579860,              0,              0],
                          [   -5128661355,    34632585198,   -98508059523,   151212114012,  -132164397513,    31090026771,              0],
                          [     945155329,    -6414710427,    18375686988,   -28498553012,    25299603603,   -12223634361,     1258225940]
                        ],
                        [
                          [    1258225940,              0,              0,              0,              0,              0,              0],
                          [  -16670007831,    55294430841,              0,              0,              0,              0,              0],
                          [   46430779053,  -308564463663,   431418789360,              0,              0,              0,              0],
                          [  -69700128812,   464200620612, -1301580166020,   985137198380,              0,              0,              0],
                          [   59577262788,  -397822832973,  1119254208255, -1701893556420,   739478564460,              0,              0],
                          [  -27545885877,   184521097818,  -521329653333,   797280592452,  -698497961463,   166930543737,              0],
                          [    5391528799,   -36253275645,   102951716988,  -158581758572,   140425750893,   -68289277071,     7177657304]
                        ]
                      ]) / 59875200

C_all = {
          2 : C_2,
          3 : C_3,
          4 : C_4,
          5 : C_5,
          6 : C_6,
          7 : C_7
        }

a_all = {
          2 : a_2,
          3 : a_3,
          4 : a_4,
          5 : a_5,
          6 : a_6,
          7 : a_7
        }

sigma_all = {
              2 : sigma_2,
              3 : sigma_3,
              4 : sigma_4,
              5 : sigma_5,
              6 : sigma_6,
              7 : sigma_7
            }

