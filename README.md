<p align="center">
 <img src="https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/banner.png" alt="BAGEL" width="480"/>
</p>

<p align="center">
 <!-- Links from the original README -->
 <a href="https://bagel-ai.org/">
  <img
   src="https://img.shields.io/badge/BAGEL-Website-0A66C2?logo=safari&logoColor=white"
   alt="BAGEL Website"
  />
 </a>
 <a href="https://arxiv.org/abs/2505.14683">
  <img
   src="https://img.shields.io/badge/BAGEL-Paper-red?logo=arxiv&logoColor=red"
   alt="BAGEL Paper on arXiv"
  />
 </a>
 <a href="https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT">
  <img
    src="https://img.shields.io/badge/BAGEL-Model-yellow?logo=huggingface&logoColor=yellow"
    alt="BAGEL Model"
  />
 </a>
 <a href="https://demo.bagel-ai.org/">
  <img
   src="https://img.shields.io/badge/BAGEL-Demo-blue?logo=googleplay&logoColor=blue"
   alt="BAGEL Demo"
  />
 </a>
 <a href="https://huggingface.co/spaces/ByteDance-Seed/BAGEL">
  <img
    src="https://img.shields.io/badge/BAGEL-Space-orange?logo=huggingface&logoColor=yellow"
    alt="BAGEL Model"
  />
 </a>
 <a href="https://discord.gg/Z836xxzy">
  <img
   src="https://img.shields.io/badge/BAGEL-Discord-5865F2?logo=discord&logoColor=purple"
   alt="BAGEL Discord"
  />
 </a>
 <a href="mailto:bagel@bytedance.com">
  <img
   src="https://img.shields.io/badge/BAGEL-Email-D14836?logo=gmail&logoColor=red"
   alt="BAGEL Email"
  />
 </a>
</p>

# BAGEL Gradio UI Fork with Extended Features

This is a fork of the official BAGEL project's Gradio WebUI, incorporating several quality-of-life improvements and features.

Based on the original work by [Chaorui Deng* et al.](https://arxiv.org/abs/2505.14683)

## Latest Update (June 19, 2025)

This update adds support for simple TeaCache implementation for Bagel inference pipeline.

### TeaCache Bagel Implementation Test

**Inference Parameters:**

```yaml
seed: 1234
thinking: false
cfg_text_scale: 4
cfg_image_scale: 2
cfg_interval: 0
cfg_renorm_type: text_channel
cfg_renorm_min: 0
timesteps: 50
timestep_shift: 3
```

**Prompt:**
She boards a modern subway, quietly reading a folded newspaper, wearing the same clothes.

| Setting           | Denoising Time | Skipped Steps | Result |
|-------------------|----------------|---------------|--------|
| TeaCache disabled | 00:51<00:00,  1.06it/s         |     0     | <img src="https://github.com/user-attachments/assets/efad65d7-829f-4a40-a1a3-f67068c7f4cf" width="137" height="175"> |
| TeaCache (thrs=0.1, warmup_steps=2)      | 00:51<00:00,  1.06it/s         |     0     | <img src="https://github.com/user-attachments/assets/ec0fa727-88cd-481e-a34f-2783ba264359" width="137" height="175"> |
| TeaCache (thrs=0.2, warmup_steps=2)      | 00:37<00:00,  1.29it/s         |     13     | <img src="https://github.com/user-attachments/assets/6e565f54-6abc-4b3c-81c3-ba517cd5fcca" width="137" height="175"> |
| TeaCache (thrs=0.3, warmup_steps=2)      | 00:26<00:00,  1.86it/s         |     24     | <img src="https://github.com/user-attachments/assets/86f45f04-af6c-465a-a3ea-b0be2bb26f30" width="137" height="175"> |
| TeaCache (thrs=0.4, warmup_steps=2)      | 00:22<00:00,  2.22it/s         |     28     | <img src="https://github.com/user-attachments/assets/7eb18d6e-8704-486f-bf93-7ed672a05ce6" width="137" height="175"> |
| TeaCache (thrs=0.5, warmup_steps=2)      | 00:17<00:00,  2.74it/s         |     32     | <img src="https://github.com/user-attachments/assets/76d44f79-47eb-4285-a628-3039f55a9066" width="137" height="175"> |
| TeaCache (thrs=0.6, warmup_steps=2)      | 00:15<00:00,  3.11it/s         |     34     |  <img src="https://github.com/user-attachments/assets/f03fc774-e8a1-4018-bbbc-e3a550722782" width="137" height="175"> |
| TeaCache (thrs=0.7, warmup_steps=2)      | 00:13<00:00,  3.60it/s         |     36     | <img src="https://github.com/user-attachments/assets/f0104ce7-8dae-4f15-b358-85d900879302" width="137" height="175"> |
| TeaCache (thrs=0.8, warmup_steps=2)      | 00:12<00:00,  3.90it/s         |     37     | <img src="https://github.com/user-attachments/assets/3750595d-906f-4854-bd25-ac3fe7df2a56" width="137" height="175"> |
| TeaCache (thrs=0.9, warmup_steps=2)      | 00:11<00:00,  4.24it/s         |     38     | <img src="https://github.com/user-attachments/assets/7a9de052-6454-44ab-bb5f-93ce3a3e47af" width="137" height="175"> |
| TeaCache (thrs=1.0, warmup_steps=2)      | 00:10<00:00,  4.67it/s         |     39     |  <img src="https://github.com/user-attachments/assets/25dc8c19-6ddd-43e1-aae2-ee3b56433c08" width="137" height="175"> |



## Update (June 03, 2025)

This update adds support for dfloat11 compressed BAGEL models and enhances model management, flexibility and inference speed within the BagelUI:

-   **DFloat11 Compressed Model Support:**
    -   Integrated full support for loading and running DFloat11 compressed version of BAGEL model.
-   **Dynamic Model Loading & Switching:**
    -   Introduced new **⚙️ Models** tab, allowing dynamic loading and switching between different BAGEL model checkpoints and quantizations.
-   **Inference Optimizations:**
    -   Made modifications to reduce memory overhead and speed up operations by disabling gradient tracking.


Special thanks to this repo for the original inference implementation of the DFloat11 model: https://github.com/LeanModels/Bagel-DFloat11/

The BagelUI-Colab.ipynb Jupyter Notebook has also been updated.
 
## ✨ Added Features

This fork builds upon the original BAGEL Gradio UI by adding the following functionalities:

-   **Structured Image Saving:** Automatically saves all generated and edited images to a configurable output directory (`output/` by default) with a clear folder structure based on the tab and mode used (Text-to-Image, Image Edit Standard, Image Edit Task Breakdown projects, X/Y Plot runs).
-   **Batch Image Generation & Editing:** Use the **Batch Size** slider in the Text-to-Image and Image Edit tabs to generate multiple images sequentially with varying seeds (or a fixed seed if specified).
-   **LLM-Powered Task Breakdown for Editing (Experimental):**
    -   An experimental mode in the **Image Edit** tab (`Enable Task Breakdown`) that leverages the built-in Qwen2 LLM to break down a complex editing prompt into sequential sub-steps.
    -   These sub-steps are applied one after another to the image.
-   **X/Y Plotting:**
    -   A dedicated **X/Y Plot** menu in the **Text to Image** and **Image Edit** tabs.
    -   Allows selecting up to two hyperparameters (X and Y axes) and providing comma-separated values for each.
    -   Generates an image for every combination of the selected parameter values.
    -   Includes **Prompt S/R (Search/Replace)** parameter for axes, akin to the same feature in Automatic1111's Stable Diffusion webui. Search for a string in the prompt and replace it with something else (separated by commas).
    -   Assembles the generated images into a single grid with axis labels indicating the parameter values used for each row/column.
-   **Batch Image Understanding/Captioning:**
    -   Adds an **Input Mode**  button to the **Image Understanding** tab for switching between single image and batch processing.
    -   The **Batch (Files/ZIP)** mode accepts multiple image files or a single ZIP file containing images.
    -   Processes each image file in the batch sequentially, saving the understanding result for each into a corresponding `.txt` file.
    -   Provides a downloadable ZIP file containing all processed images and their generated `.txt` files.

## ⚙️ Local Installation
(A Jupyter Notebook 'BagelUI-colab.ipynb' for easy cloud-use is also provided, L4 GPU is enough to run DFloat11)
1.  **Clone this fork:**
    ```bash
    git clone https://github.com/dasjoms/BagelUI.git
    cd BagelUI
    ```

2.  **Set up environment:**
    ```bash
    conda create -n bagel python=3.11.12 -y
    conda activate bagel
    pip install -r requirements.txt
    pip install flash_attn==2.5.8 --no-build-isolation
    ```
    2.5 **Manually install dfloat11 package**
    
     ```bash
       pip install dfloat11
       ```
    
3.  **Download pretrained checkpoint and/or DFloat11 compressed model:**

    ```python
    ### Regular BAGEL-7B Model
    
    from huggingface_hub import snapshot_download

    save_dir = "models/BAGEL-7B-MoT"
    repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
    cache_dir = save_dir + "/cache"

    snapshot_download(cache_dir=cache_dir,
     local_dir=save_dir,
     repo_id=repo_id,
     local_dir_use_symlinks=False,
     resume_download=True,
     allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
    )
    ```
    ```python
    ### DFloat11 Model - Allows for 24GB VRAM Single GPU inference without quality loss
    
    from huggingface_hub import snapshot_download

    save_dir = "models/BAGEL-7B-MoT-DF11"
    repo_id = "DFloat11/BAGEL-7B-MoT-DF11"
    cache_dir = save_dir + "/cache"

    snapshot_download(cache_dir=cache_dir,
     local_dir=save_dir,
     repo_id=repo_id,
     local_dir_use_symlinks=False,
     resume_download=True,
     allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt", "*.model", "vae/*"],
    )
    ```




## ▶️ Usage

Run the modified Gradio WebUI script:

```bash
# For 32GB+ VRAM GPU or multi GPUs. Saves output to ./output/
python app.py

# To specify a different output directory
python app.py --output_dir /path/to/your/output

# For 12~32GB VRAM GPU/NF4 quantization and Chinese UI
python app.py --mode 2 --zh

# Different Requirements apply for using the DFloat11 Model
```

## ❤️ Based on the Original BAGEL Project
This work is based on the amazing BAGEL project. Please refer to the original repository for core model details, training guidelines, and benchmarks.

## ✍️ Citation

```bibtex
@article{deng2025bagel,
  title   = {Emerging Properties in Unified Multimodal Pretraining},
  author  = {Deng, Chaorui and Zhu, Deyao and Li, Kunchang and Gou, Chenhui and Li, Feng and Wang, Zeyu and Zhong, Shu and Yu, Weihao and Nie, Xiaonan and Song, Ziang and Shi, Guang and Fan, Haoqi},
  journal = {arXiv preprint arXiv:2505.14683},
  year    = {2025}
}
```


## 📜 License
BAGEL is licensed under the Apache 2.0.

<p align="center"> <a href="https://bagel-ai.org/">Website</a> | <a href="https://arxiv.org/abs/2505.14683">Paper</a> | <a href="https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT">Model on Hugging Face</a> | <a href="https://demo.bagel-ai.org/">Official Demo</a> | <a href="https://huggingface.co/spaces/ByteDance-Seed/BAGEL">Official Hugging Face Space</a> | <a href="https://discord.gg/Z836xxzy">Discord</a> | <a href="mailto:bagel@bytedance.com">Email</a> </p>
