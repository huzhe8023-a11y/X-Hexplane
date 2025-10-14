# Super-time resolved tomography

If you have any questions, please feel free to email me at zhe.hu@sljus.lu.se
# Environment Setup
```
    # create conda environment
    conda create --name xhexplane python=3.8
    
    # activate env
    conda activate xhexplane
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1  cudatoolkit=11.6 -c pytorch -c conda-forge

    # pip install 
    pip install -r requirements.txt
    python setup.py develop

```
# Data Preparation
The additive manufacturing datasets used for validation of STRT can be found with the doi: 10.5281/zenodo.17349522

# Reconstruction
```
python main.py config=XMPI.yaml
```

# Evaluation
With `render_test=True`, `render_path=True`, results at test viewpoint are automatically evaluated and validation viewpoints are generated after reconstruction.  

Or
```
python main.py config=XMPI_test.yaml systems.ckpt="checkpoint/path" render_only=True
```




