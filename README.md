### Downloading the Data

Download the RIVAL10 dataset from the command line using:
```
curl -L 'https://app.box.com/index.php?rm=box_download_shared_file&shared_name=iflviwl5rbdgtur1rru3t8f7v2vp0gww&file_id=f_944375052992' -o rival10.zip
unzip rival10.zip
```
### Obtaining Models Finetuned on RIVAL10

To finetune all models on RIVAL10, run 'python finetuner.py --all'. 
Alternatively, download weights of final linear layers finetuned for RIVAL10 classification with the following command:
```
curl -L 'https://app.box.com/index.php?rm=box_download_shared_file&shared_name=cvrqihbz6u0y9niyf5f4eazkhzki7lgr&file_id=f_944516981880' -o rival10_ft_model_weights.zip
unzip rival10_ft_model_weights.zip
```

### Reproducing Analysis

Most plots are generated in noise_analysis.py. 

Noise analysis: first run compute_noise_robustness.py, then noise_analysis_plots.py

Saliency analysis: first run bg_fg_saliency.py, then saliency_alignment_plots.py

Attribute ablation: run attr_importance.py

### Citation

If this dataset or code is of use to you, please consider citing our CVPR paper. 
```
@inproceedings{moayeri2022comprehensive,
    title     = {A Comprehensive Study of Image Classification Model Sensitivity to
                 Foregrounds, Backgrounds, and Visual Attributes},
    author    = {Moayeri, Mazda and Pope, Phillip and Balaji, Yogesh and Feizi, Soheil},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    month     = {June},
    year      = {2022},
    }
```
_________________________________

### License

COPYRIGHT AND PERMISSION NOTICE
UMD Software RIVAL10 Dataset Copyright (C) 2022 University of Maryland
All rights reserved.
The University of Maryland (“UMD”) and the developers of RIVAL10 Dataset software (“Software”) give recipient (“Recipient”) permission to download a single copy of the Software in source code form and use by university, non-profit, or research institution users only, provided that the following conditions are met:
1)  Recipient may use the Software for any purpose, EXCEPT for commercial benefit.
2)  Recipient will not copy the Software.
3)  Recipient will not sell the Software.
4)  Recipient will not give the Software to any third party.
5)  Any party desiring a license to use the Software for commercial purposes shall contact:
UM Ventures, College Park at UMD at otc@umd.edu.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS, CONTRIBUTORS, AND THE UNIVERSITY OF MARYLAND "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO  EVENT SHALL THE COPYRIGHT OWNER, CONTRIBUTORS OR THE UNIVERSITY OF MARYLAND BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
