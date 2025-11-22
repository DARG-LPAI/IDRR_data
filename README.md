# IDRR_data

The data of IDRR (Implicit Discourse Relation Recognition), including PDTB2, PDTB3 and CoNLL16. 

To get dataframe easily.

~~~sh
cd IDRR_data
pip install -e .
~~~

## Data

* columns of csv

~~~ py
'arg1', 'arg2', 'conn1', 'conn2', 
'conn1sense1', 'conn1sense2', 'conn2sense1', 'conn2sense2',
'relation', 'split', 'data_id'
~~~

* get dataframe from IDRRDataFrames in dataframe.py 

* new columns

~~~ py
'index', 'arg1', 'arg2', 'conn1', 'conn2', 
'conn1sense1', 'conn1sense2', 'conn2sense1', 'conn2sense2', 
'relation', 'split', 'data_id', 
'label11', 'label11id', 'label12', 'label12id', 
'label21', 'label21id', 'label22', 'label22id', 
'ans_word1', 'ans_word1id', 'ans_word2', 'ans_word2id'
~~~

## Preprocess

1. connXsenseY -> labelXY, labelXYid
2. connX -> ans_wordX, ans_wordXid
3. filter relation ['Explicit', 'Implicit']
4. filter split ['train', 'dev', 'test', 'blind-test']


## Raw Resource

* PDTB2

[https://github.com/cgpotts/pdtb2](https://github.com/cgpotts/pdtb2)

[https://catalog.ldc.upenn.edu/LDC2008T05](https://catalog.ldc.upenn.edu/LDC2008T05)

* PDTB3

[https://github.com/najoungkim/pdtb3](https://github.com/najoungkim/pdtb3)

[https://catalog.ldc.upenn.edu/LDC2019T05](https://catalog.ldc.upenn.edu/LDC2019T05)

* CoNLL16

[https://github.com/attapol/conll16st](https://github.com/attapol/conll16st)

[https://www.cs.brandeis.edu/~clp/conll16st/dataset.html](https://www.cs.brandeis.edu/~clp/conll16st/dataset.html)

* Preprocess

[https://github.com/najoungkim/pdtb3/blob/master/preprocess/preprocess_pdtb3.py](https://github.com/najoungkim/pdtb3/blob/master/preprocess/preprocess_pdtb3.py)
