############################################################################################################
############################################################################################################

Résultats sur le val set avec plusieurs méthodes de pre-processing :

Method: Contrast, Param: 1.5, Accuracy: 0.15, Avg Correct Score: 0.85, Avg Incorrect Score: 0.59
Method: Contrast, Param: 2.0, Accuracy: 0.15, Avg Correct Score: 0.85, Avg Incorrect Score: 0.58
Method: Contrast, Param: 2.5, Accuracy: 0.14, Avg Correct Score: 0.85, Avg Incorrect Score: 0.58
Method: Binarization, Param: 100, Accuracy: 0.12, Avg Correct Score: 0.78, Avg Incorrect Score: 0.57
Method: Binarization, Param: 128, Accuracy: 0.12, Avg Correct Score: 0.82, Avg Incorrect Score: 0.56
Method: Binarization, Param: 150, Accuracy: 0.12, Avg Correct Score: 0.84, Avg Incorrect Score: 0.56
Method: Blur, Param: 1, Accuracy: 0.15, Avg Correct Score: 0.84, Avg Incorrect Score: 0.58
Method: Blur, Param: 2, Accuracy: 0.12, Avg Correct Score: 0.82, Avg Incorrect Score: 0.56
Method: Blur, Param: 3, Accuracy: 0.09, Avg Correct Score: 0.80, Avg Incorrect Score: 0.54
Method: Sharpen, Param: None, Accuracy: 0.15, Avg Correct Score: 0.84, Avg Incorrect Score: 0.59
Method: Equalize, Param: None, Accuracy: 0.14, Avg Correct Score: 0.84, Avg Incorrect Score: 0.58
Method: Resize, Param: 1.5, Accuracy: 0.16, Avg Correct Score: 0.85, Avg Incorrect Score: 0.59
Method: Resize, Param: 2.0, Accuracy: 0.15, Avg Correct Score: 0.84, Avg Incorrect Score: 0.58
Method: Resize, Param: 2.5, Accuracy: 0.15, Avg Correct Score: 0.84, Avg Incorrect Score: 0.58

Conclusion : 

Ne pas utiliser de binarisation - Ne pas utiliser de blur - combiner resize 1.5 et contrast 1.5 ? (non concluant)

## UNIQUEMENT RESIZE AVEC UN PARAMÈTRE DE 1,5 ##

############################################################################################################
############################################################################################################

Nombre de réponses sur le test set (a servi à ajuster les mots associés aux fromages) :


0            BRIE DE MELUN     26
1                CAMEMBERT     45
2                 EPOISSES     54
3          FOURME D’AMBERT     17
4                 RACLETTE     10
5                  MORBIER     11
6           SAINT-NECTAIRE      7
7   POULIGNY SAINT- PIERRE     21
8                ROQUEFORT     23
9                    COMTÉ      9
10                  CHÈVRE      0
11                PECORINO     26
12              NEUFCHATEL     20
13                 CHEDDAR     37
14      BÛCHETTE DE CHÈVRE     23
15                PARMESAN     33
16         SAINT- FÉLICIEN      8
17               MONT D’OR     82
18                 STILTON     23
19                SCARMOZA     20
20                 CABECOU     16
21                BEAUFORT     15
22                 MUNSTER     32
23               CHABICHOU     21
24          TOMME DE VACHE      7
25               REBLOCHON     31
26                EMMENTAL     20
27                    FETA     39
28            OSSAU- IRATY     46
29               MIMOLETTE     20
30               MAROILLES     23
31                 GRUYÈRE     22
32                 MOTHAIS     23
33                VACHERIN     20
34              MOZZARELLA     46
35          TÊTE DE MOINES     68
36           FROMAGE FRAIS     43


