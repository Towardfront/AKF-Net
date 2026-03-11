# 📚 Overview
The official implementation of "Adaptive Kernel Selection Module Combined with Feature Enhanced Perception Network for Camouflaged Object Detection"
# 📦 Requirements
- Python >= 3.9 
- PyTorch >= 2.1.1
- Other dependencies can be found in requirements.txt
# 🛠️ Usage
```bash
git clone https://github.com/Towardfront/AKF-Net.git
cd AKF-Net
python train.py
python test.py
```
# 📚 Citation
```bibtex
@ARTICLE{11202305,
  author={Liu, Ruyu and Yang, Guang and Xiao, Feng and Zhang, Jianhua and Chen, Shengyong},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={Adaptive Kernel Selection Module Combined With Feature Enhanced Perception Network for Camouflaged Object Detection}, 
  year={2025},
  volume={},
  number={},
  pages={1-10},
  keywords={Feature extraction;Object detection;Shape;Transformers;Image segmentation;Deep learning;Computer vision;Accuracy;Translation;Transforms;Camouflaged object detection;computer vision;deep learning;salient object detection (SOD);vision transformer},
  doi={10.1109/TII.2025.3609076}}
```

**Note:** We combined the encoders of two large vision models as the backbone, which resulted in significantly improved performance. Specifically, the updated model achieves competitive metrics on the R2C7K dataset, including $M \downarrow = 0.016$, $S_m \uparrow = 0.901$, $\alpha E \uparrow = 0.947$, and $wF \uparrow = 0.839$. Therefore, we have uploaded the latest version of the project.
