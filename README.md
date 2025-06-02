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

This is a fork of the official BAGEL project's Gradio WebUI, incorporating several quality-of-life improvements and advanced features for enhanced multimodal interaction.

Based on the original work by [Chaorui Deng* et al.](https://arxiv.org/abs/2505.14683)

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
    -   Includes **Prompt S/R (Search/Replace)** parameter for axes, akin to the same feature in Automatic1111's Stable Diffusion webui. Search for a string in the prompt and replace it with something else (separated by commas)
    -   Assembles the generated images into a single grid with axis labels indicating the parameter values used for each row/column.
-   **Batch Image Understanding/Captioning:**
    -   Adds an **Input Mode**  button to the **Image Understanding** tab for switching between single image and batch processing.
    -   The **Batch (Files/ZIP)** mode accepts multiple image files or a single ZIP file containing images.
    -   Processes each image file in the batch sequentially, saving the understanding result for each into a corresponding `.txt` file.
    -   Provides a downloadable ZIP file containing all processed images and their generated `.txt` files.

## ⚙️ Installation

1.  **Clone this fork:**
    ```bash
    git clone https://github.com/dasjoms/BagelUI.git
    cd BAGEL
    ```

2.  **Set up environment:**
    ```bash
    conda create -n bagel python=3.10 -y
    conda activate bagel
    pip install -r requirements.txt
    pip install flash_attn==2.5.8 --no-build-isolation
    ```
    
3.  **Download pretrained checkpoint:**
    ```python
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

## ▶️ Usage

Run the modified Gradio WebUI script:

```bash
# For 32GB+ VRAM GPU or multi GPUs. Saves output to ./output/
python app_rework.py

# To specify a different output directory
python app_rework.py --output_dir /path/to/your/output

# For 12~32GB VRAM GPU/NF4 quantization and Chinese UI
python app_rework.py --mode 2 --zh


❤️ Based on the Original BAGEL Project
This work is based on the amazing BAGEL project. Please refer to the original repository for core model details, training guidelines, and benchmarks.

<p align="center"> <a href="https://bagel-ai.org/">Website</a> | <a href="https://arxiv.org/abs/2505.14683">Paper</a> | <a href="https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT">Model on Hugging Face</a> | <a href="https://demo.bagel-ai.org/">Official Demo</a> | <a href="https://huggingface.co/spaces/ByteDance-Seed/BAGEL">Official Hugging Face Space</a> | <a href="https://discord.gg/Z836xxzy">Discord</a> | <a href="mailto:bagel@bytedance.com">Email</a> </p>
✍️ Citation
@article{deng2025bagel,
 author = {Deng, Chaorui and Zhu, Deyao and Li, Kunchang and Gou, Chenhui and Li, Feng and Wang, Zeyu and Zhong, Shu and Yu, Weihao and Nie, Xiaonan and Song, Ziang and Shi, Guang and Fan, Haoqi},
 title = {Emerging Properties in Unified Multimodal Pretraining},
 journal = {arXiv preprint arXiv:2505.14683},
 year = {2025}
}
📜 License
This project is licensed under the Apache 2.0 License, inherited from the original BAGEL project.
