# Predicting Instance Type Assertions in Knowledge Graphs Using Stochastic Neural Networks
Tensorflow-based implementation of [Ridle](#link.pdf) (<ins>R</ins>elation - <ins>I</ins>nstance <ins>D</ins>istribution <ins>Le</ins>arning) for supervised node classification on (directed) relational graphs, as published in [CIKM 2021](https://www.cikm2021.org/).


![PCA projections of learned entity representations](https://github.com/TobiWeller/Ridle/blob/main/graph.png?raw=true| width=100)
PCA projections for the learned entity representations. Popular classes from cross-domain KGs were selected forvisualization. Ridle (top) allows for a better separation of the instances into their respective classes in comparison to thestate-of-the-art RDF2Vec approach (bottom).


## Setup
To use this package, you must install the following dependencies first: 
- python (>=3.7)
- Tensorflow (>=2.1)
- numpy (>=1.18)
- pandas (>=1.2.3)


## Training
To use Ridle, you must provide your graph data as a pkl file in the format S-P-O) in the folder dataset. Examples are given in the folder dataset. You can learn the representations on DBLP using Ridle, you can use the following command, specifying the dataset with the argument --dataset. This file loads the umls knowledge graph graph and learns a representation using Ridle, exploiting a target distribution over the usage of relations. The representations are saved in a csv in the same folder as the dataset.
```
python learn_representation.py --dataset umls
```


Afterwards you can evaluate the representations learned by Ridle to predict instance types. The evaluation is based on 10-fold cross-validation. The results are saved in a csv.
```
python evaluate_instance_type.py --dataset umls
```


The file *run.sh* is given to combine both commands in order to immediately evualuate the representations. It applies the method on the knowledge graph umls and stores the embeddings in a csv. Afterwards, it applies the learned representations for instance type prediction. The results are saved in a csv. Due to the size of DBpedia and Wikidata and the limited space for uploading the datasets, we uploaded subsets of the datasets DBp_2016-04, DBp_3.8 and WD_2017-03-13 for which experiments were conducted.

```
sh run.sh
```


## Results
The following image shows the results reported in the [Paper](#link.pdf). Considering the cross-domain KGs (cf. Table 2a), Ridle signifi-cantly outperforms the state-of-the-art methods with respect to the metric F1-macro. Considering the  performance  of  the  approaches  in  the category-specific KGs (cf. Table 2b), we can conclude that Ridle achieves competitive performance in comparison to the best approaches.
The experimental results showed that, on average, Ridle outperforms current state-of-the-art models in several KGs, which sets a new baseline in the tasks of predicting instance type assertions.
![Results of Ridle for predicting instance types](https://github.com/TobiWeller/Ridle/blob/main/results.png?raw=true)



## Citation
If you use this code for evidential learning as part of your project or paper, please cite the following work:  

    @article{weller2021ridle,
      title={Predicting Instance Type Assertions in Knowledge Graphs Using Stochastic Neural Networks},
      author={Weller, Tobias and Acosta, Maribel},
      journal={Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM '21), November 1--5, 2021, Virtual Event, QLD, Australia},
      year={2021}
    }



## Licence
Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
