#Contrastive Learning of Global-Local Video Representations

#![arch](asset/teaser.png)

Implementation of ''Contrastive Learning of Global-Local Video Representations. NeurIPS 2021''

### Link:
#[[PDF]] ()
#[[Arxiv]]()

### Pretrain Instruction

Global-local pretrain on Kinetics


# create conda environment:
```
conda env create -f conda_env.yml

```

# Prepare dataset:

* RGB for UCF101: [[download-from-server]](http://thor.robots.ox.ac.uk/~vgg/data/CoCLR/ucf101_rgb_lmdb.tar) [[download-from-gdrive]](https://drive.google.com/file/d/1jVqBWl6iHYzcnb0IZ5ezpH_uK5jdHtoF/view?usp=sharing) (tar file, 29GB, packed with lmdb)
* TVL1 optical flow for UCF101: [[download-from-server]](http://thor.robots.ox.ac.uk/~vgg/data/CoCLR/ucf101_flow_lmdb.tar) [[download-from-gdrive]](https://drive.google.com/file/d/1NRElvRyVKX8siVu5HFKOETn4uqnzM4GH/view?usp=sharing) (tar file, 20.5GB, packed with lmdb)
* Note: I created these lmdb files with msgpack==0.6.2, when load them with msgpack>=1.0.0, you can do `msgpack.loads(raw_data, raw=True)`([issue#32](https://github.com/TengdaHan/CoCLR/issues/32))


# Pretraining
```
bash run.sh

```

