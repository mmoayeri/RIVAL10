### Downloading the Data

Download the RIVAL10 dataset from the command line using:
```
curl -L 'https://app.box.com/index.php?rm=box_download_shared_file&shared_name=iflviwl5rbdgtur1rru3t8f7v2vp0gww&file_id=f_944375052992' -o rival10.zip
unzip rival10.zip
```
### Obtaining Models Finetuned on RIVAL10

To finetune all models on RIVAL10, run 'python finetuner.py --all'. 
Alternatively, download weights of final linear layers finetuned for RIVAL10 classification from here. 

### Reproducing Analysis

Most plots are generated in noise_analysis.py. 

Noise analysis: first run compute_noise_robustness.py, then noise_analysis_plots.py
Saliency analysis: first run bg_fg_saliency.py, then saliency_alignment_plots.py
Attribute ablation: run attr_importance.py
