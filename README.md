## Installation

Clone the repository locally:

```shell
git clone https://github.com/kb256188/pc_seg.git
```

```shell
conda create -n FastSAM python=3.9
conda activate FastSAM
```

Install the packages:

```shell
cd FastSAM
pip install -r requirements.txt
```

## <a name="GettingStarted"></a> Getting Started

First download a [model checkpoint](#model-checkpoints)(FastSAM-x.pt).

Then, Put the checkpoint file in./weights.



## Data Preparation
Put the data in ./guangxi_pc 
You can change the path of the input file on line 282 of the inference.py file

```shell
# Run
python Inference.py 
```

