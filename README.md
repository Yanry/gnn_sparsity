# LightGCN
```
$ cd LightGCN
$ python lightgcn_main.py --dataset amazon-book --adj_type pre --Ks [20]
```
output: LightGCN/output

# Graph_Transformer_Networks
Dataset: Download datasets (DBLP, ACM, IMDB) from this [link](https://drive.google.com/file/d/1Nx74tgz_-BDlqaFO75eQG6IkndzI92j4/view?usp=sharing) and extract data.zip into data folder.
``` 
$ mkdir data
$ cd data
$ unzip data.zip
$ cd ..
```
```
$ cd Graph_Transformer_Networks
$ python fastgtn_main.py --dataset ACM/DBLP/IMDB --non_local
```
**Notice:** --non_local must be True

output: Graph_Transformer_Networks/output