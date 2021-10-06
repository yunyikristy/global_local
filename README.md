#Contrastive Learning of Global-Local Video Representations

#![arch](asset/teaser.png)

Implementation of ''Contrastive Learning of Global-Local Video Representations. NeurIPS 2021''

### Link:
#[[PDF]] ()
#[[Arxiv]](https://arxiv.org/pdf/2104.05418.pdf)

### Pretrain Instruction

Global-local pretraining on Kinetics


# create conda environment:
```
conda env create -f conda_env.yml

```

# Prepare dataset:

* Kinetics: [[Download]](https://deepmind.com/research/open-source/kinetics)
* UCF101: [[Download]](https://www.crcv.ucf.edu/research/data-sets/ucf101/)
* HMDB51: [[Download]](https://deepai.org/dataset/hmdb-51)
* LRW: [[Download]](https://www.robots.ox.ac.uk/~vgg/data/lip-reading/lrw1.html)


# Pretraining
```
bash run.sh

```

