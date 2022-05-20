## Non-rigid Point Cloud Registration with Neural Deformation Pyramid 

### Requirements

The code tested on python=3.8.10, pytorch=1.9.0 with the following packages:
- pytorch3d, open3d, opencv-python, tqdm, mayavi, easydict

 

### Obtain the 4DMatch benchmark
- Download the train/val/4DMatch-F/4DLoMatch-F split, [(google drive, 14G)](https://drive.google.com/file/d/1ySykuxxRyE-OvFY8gDgE_SoacKbexMDz/view?usp=sharing). We filter point cloud pairs with near-rigid motions from the original 4DMatch benchmark.  4DMatch-F & 4DLoMatch-F denote the filtered benchmark.
- Extract it and create a soft link under this repository.
```shell
ln -s /path/to/4Dmatch  ./data
```



### Reproduce the result of NDP (no-learned)
- Run
```eval
python eval_nolearned.py --config config/NDP.yaml  
```
To visualize the registration result, add ```--visualize```.


### Reproduce the result of LNDP (supervised)
- First download pre-trained point cloud matching and outlier rejection models [(google drive, 271M)](https://drive.google.com/file/d/1T8z71iv3dvyAQhZUgct0w5yDtfRFwui9/view?usp=sharing). Move the models to ``correspondence/pretrained``
- Install KPConv
```shell
cd correspondence/cpp_wrappers; sh compile_wrappers.sh; cd ..
```
- Finally run
```
python eval_supervised.py --config config/LNDP.yaml  
```
To visualize the registration result, add ```--visualize```.

 

### Run  shape transfer example
```
python shape_transfer.py -s sim3_demo/AlienSoldier.ply -t sim3_demo/Ortiz.ply
```

## Related projects

### Dataset
The datasets are obtained by agreeing to be bound by their terms and conditions.  
Original 4DMatch: [rabbityl/lepard](https://github.com/rabbityl/lepard) [[License](https://docs.google.com/forms/d/e/1FAIpQLSeQ1hkCmmTiib-oQM9s21y3Tz9ojiI2zB8vZSqTZjT2DiRZ0g/viewform)]  
DeformingThings4D: [rabbityl/DeformingThings4D](https://github.com/rabbityl/DeformingThings4D) [[License](https://docs.google.com/forms/d/e/1FAIpQLSckMLPBO8HB8gJsIXFQHtYVQaTPTdd-rZQzyr9LIIkHA515Sg/viewform)]  
Mixamo: [http://mixamo.com/](http://mixamo.com/) [[License](https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html)]


### Baselines (a few are not official):  
CPD (Myronenko et al., 2010): [siavashk/pycpd](https://github.com/siavashk/pycpd)  
BCPD (Hirose et al., 2020): [ohirose/bcpd](https://github.com/ohirose/bcpd)  
Sinkhorn (Feydy et al., 2019): [jeanfeydy/geomloss](https://github.com/jeanfeydy/geomloss)  
ZoomOut (Melzi et al., 2019): [RobinMagnet/pyFM](https://github.com/RobinMagnet/pyFM)  
NSFP (Li et al., 2021): [Lilac-Lee/Neural_Scene_Flow_Prior](https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior)  
Nerfies (Park et al., 2021): [google/nerfies](https://github.com/google/nerfies)  
GeomFmaps (Donati et al., 2020): [LIX-shape-analysis/GeomFmaps](https://github.com/LIX-shape-analysis/GeomFmaps)  
FLOT (Puy et al., 2020): [valeoai/FLOT](https://github.com/valeoai/FLOT)  
PointPWC (Wu et al., 2020): [DylanWusee/PointPWC](https://github.com/DylanWusee/PointPWC)  
Synorim (Huang et al., 2022): [huangjh-pub/synorim](https://github.com/huangjh-pub/synorim)  
Lepard, (Li et al., 2022): [rabbityl/lepard](https://github.com/rabbityl/lepard)

