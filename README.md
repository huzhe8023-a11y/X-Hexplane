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


# Reconstruction
```
python main.py config=XMPI.yaml
```


`"HexPlane"` is the complete HexPlane model, whose *Fusion Mechanism*, *Density Field* are controllable. 

*Fusion Mechanism*: XHexPlane computes features from six feature planes, where two complementary planes (like XY and ZT) are paired and there are three pairs in total. 
Consequently, there are two fusion steps to fuse features from six planes. 
`fusion_one` controls the fusion operation between paired feature planes, leading to three fused features, and `fusion_two` controls the operation between three fused features.
Both fusion operation could be chosen from `multiply`, `sum` and `concat`.



# Evaluation
With `render_test=True`, `render_path=True`, results at test viewpoint are automatically evaluated and validation viewpoints are generated after reconstruction.  

Or
```
python main.py config=XMPI_test.yaml systems.ckpt="checkpoint/path" render_only=True
```




