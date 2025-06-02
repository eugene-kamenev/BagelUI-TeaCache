import gradio as gr
import numpy as np
import os
import torch
import random
import datetime
import re
from PIL import Image, ImageDraw, ImageFont
import zipfile 
import tempfile 
import shutil 
from PIL import UnidentifiedImageError


from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--server_name", type=str, default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=7860)
parser.add_argument("--share", action="store_true")
parser.add_argument("--model_path", type=str, default="models/BAGEL-7B-MoT")
parser.add_argument("--mode", type=int, default=1)
parser.add_argument("--zh", action="store_true")
parser.add_argument("--output_dir", type=str, default="output", help="Base directory to save generated images.")
args = parser.parse_args()

if args.output_dir:
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generated images will be saved under: {os.path.abspath(args.output_dir)}")

model_path = args.model_path
llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json")); llm_config.qk_norm = True; llm_config.tie_word_embeddings = False; llm_config.layer_module = "Qwen2MoTDecoderLayer"
vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json")); vit_config.rope = False; vit_config.num_hidden_layers -= 1
vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
config = BagelConfig(visual_gen=True, visual_und=True, llm_config=llm_config, vit_config=vit_config, vae_config=vae_config, vit_max_num_patch_per_side=70, connector_act='gelu_pytorch_tanh', latent_patch_size=2, max_latent_size=64)
with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config); vit_model = SiglipVisionModel(vit_config); model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
tokenizer = Qwen2Tokenizer.from_pretrained(model_path); tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
vae_transform = ImageTransform(1024, 512, 16); vit_transform = ImageTransform(980, 224, 14)
device_map = infer_auto_device_map(model, max_memory={i: "80GiB" for i in range(torch.cuda.device_count())}, no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"])
same_device_modules = ['language_model.model.embed_tokens', 'time_embedder', 'latent_pos_embed', 'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed']
if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0");
    for k in same_device_modules: device_map[k] = first_device if k in device_map else "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0]);
    for k in same_device_modules:
        if k in device_map: device_map[k] = first_device
if args.mode == 1: model = load_checkpoint_and_dispatch(model, checkpoint=os.path.join(model_path, "ema.safetensors"), device_map=device_map, offload_buffers=True, offload_folder="offload", dtype=torch.bfloat16, force_hooks=True).eval()
elif args.mode == 2: bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False, bnb_4bit_quant_type="nf4"); model = load_and_quantize_model(model, weights_location=os.path.join(model_path, "ema.safetensors"), bnb_quantization_config=bnb_quantization_config, device_map=device_map, offload_folder="offload").eval()
elif args.mode == 3: bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, torch_dtype=torch.float32); model = load_and_quantize_model(model, weights_location=os.path.join(model_path, "ema.safetensors"), bnb_quantization_config=bnb_quantization_config, device_map=device_map, offload_folder="offload").eval()
else: raise NotImplementedError
inferencer = InterleaveInferencer(model=model, vae_model=vae_model, tokenizer=tokenizer, vae_transform=vae_transform, vit_transform=vit_transform, new_token_ids=new_token_ids)

def set_seed(seed):
    if seed > 0: random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None; torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    elif seed == 0: random.seed(None); np.random.seed(None); torch.manual_seed(random.randint(0, 2**32 -1)); torch.cuda.manual_seed_all(random.randint(0, 2**32 -1)) if torch.cuda.is_available() else None; torch.backends.cudnn.deterministic = False; torch.backends.cudnn.benchmark = True
    return seed

def text_to_image(prompt, show_thinking=False, cfg_text_scale=4.0, cfg_interval=0.4, timestep_shift=3.0, num_timesteps=50, cfg_renorm_min=0.0, cfg_renorm_type="global", max_think_token_n=1024, do_sample=False, text_temperature=0.3, seed=0, image_ratio="1:1", save_to_dir=None, batch_index=0, is_xy_plot_image=False, xy_plot_filename_prefix=""):
    current_seed_for_generation = seed; set_seed(current_seed_for_generation)
    if image_ratio == "1:1": image_shapes = (1024, 1024)
    elif image_ratio == "4:3": image_shapes = (768, 1024)
    elif image_ratio == "3:4": image_shapes = (1024, 768)
    elif image_ratio == "16:9": image_shapes = (576, 1024)
    elif image_ratio == "9:16": image_shapes = (1024, 576)
    else: image_shapes = (1024,1024)
    inference_hyper = dict(max_think_token_n=max_think_token_n if show_thinking else 1024, do_sample=do_sample if show_thinking else False, text_temperature=text_temperature if show_thinking else 0.3, cfg_text_scale=cfg_text_scale, cfg_interval=[cfg_interval, 1.0], timestep_shift=timestep_shift, num_timesteps=num_timesteps, cfg_renorm_min=cfg_renorm_min, cfg_renorm_type=cfg_renorm_type, image_shapes=image_shapes)
    result = inferencer(text=prompt, think=show_thinking, **inference_hyper)
    generated_image = result["image"]
    if save_to_dir and generated_image:
        os.makedirs(save_to_dir, exist_ok=True)
        if is_xy_plot_image: filename = f"{xy_plot_filename_prefix}_seed{current_seed_for_generation}.png"
        else: timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"); filename = f"txt2img_{timestamp}_seed{current_seed_for_generation}_batch{batch_index}.png"
        save_path = os.path.join(save_to_dir, filename)
        try: generated_image.save(save_path); print(f"Saved T2I image: {save_path}")
        except Exception as e: print(f"Error saving T2I image to {save_path}: {e}")
    return generated_image, result.get("text", None)

def edit_image(image: Image.Image, prompt: str, show_thinking=False, cfg_text_scale=4.0, cfg_img_scale=2.0, cfg_interval=0.0, timestep_shift=3.0, num_timesteps=50, cfg_renorm_min=0.0, cfg_renorm_type="text_channel", max_think_token_n=1024, do_sample=False, text_temperature=0.3, seed=0, save_to_dir=None, batch_index=0, sub_step_idx=0, is_xy_plot_image=False, xy_plot_filename_prefix=""):
    current_seed_for_generation = seed; set_seed(current_seed_for_generation)
    if image is None: return "Please upload an image.", ""
    image_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image; image_pil = pil_img2rgb(image_pil)
    inference_hyper = dict(max_think_token_n=max_think_token_n if show_thinking else 1024, do_sample=do_sample if show_thinking else False, text_temperature=text_temperature if show_thinking else 0.3, cfg_text_scale=cfg_text_scale, cfg_img_scale=cfg_img_scale, cfg_interval=[cfg_interval, 1.0], timestep_shift=timestep_shift, num_timesteps=num_timesteps, cfg_renorm_min=cfg_renorm_min, cfg_renorm_type=cfg_renorm_type)
    result = inferencer(image=image_pil, text=prompt, think=show_thinking, **inference_hyper)
    edited_image = result["image"]
    if save_to_dir and edited_image:
        os.makedirs(save_to_dir, exist_ok=True)
        if is_xy_plot_image: filename = f"{xy_plot_filename_prefix}_seed{current_seed_for_generation}.png"
        else: timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"); filename = f"imgedit_{timestamp}_seed{current_seed_for_generation}_batch{batch_index}_substep{sub_step_idx}.png"
        save_path = os.path.join(save_to_dir, filename)
        try: edited_image.save(save_path); print(f"Saved Edit image: {save_path}")
        except Exception as e: print(f"Error saving Edit image to {save_path}: {e}")
    return edited_image, result.get("text", "")

def _perform_image_understanding(pil_image: Image.Image, prompt: str, show_thinking: bool, do_sample: bool, text_temperature: float, max_new_tokens: int):
    if pil_image is None: return "Error: No image provided to understanding core."
    set_seed(0)
    inference_hyper = dict(do_sample=do_sample, text_temperature=text_temperature, max_think_token_n=max_new_tokens)
    result = inferencer(image=pil_image, text=prompt, think=show_thinking, understanding_output=True, **inference_hyper)
    return result["text"]

def load_example_image(image_path):
    try: return Image.open(image_path)
    except Exception as e: print(f"Error loading example image {image_path}: {e}"); return None

def decompose_task_with_llm(user_prompt: str, context_image: Image.Image, show_thinking: bool, do_sample: bool, text_temperature: float, max_new_tokens: int, num_steps_to_generate: int = 3):
    system_prompt = f"""You are an expert at breaking down complex image editing requests into simple, sequential steps.
Given the user's editing instruction: "{user_prompt}"
Your task is to break this instruction down into exactly {num_steps_to_generate} smaller, distinct, and sequential visual editing steps.
Each step should be a clear instruction that can be applied to an image.
Output each step on a new line, starting with 'Step X: ' where X is the step number. Do not add any other commentary or preamble.
Example:
User's instruction: "Make the character look like a cyberpunk detective in a rainy neon city."
Your output (after any internal thinking within <think></think> tags if you use them):
Step 1: Change the character's clothing to a futuristic trench coat and add some cybernetic implants.
Step 2: Alter the background to a dark, rainy city street illuminated by neon signs.
Step 3: Apply a gritty, high-contrast filter to the entire image to enhance the cyberpunk aesthetic.

Now, process the following user instruction:
User's instruction: "{user_prompt}"
Your output:
"""
    print(f"Decomposing prompt: '{user_prompt}' with system message to LLM.")
    raw_llm_output = _perform_image_understanding(pil_image=context_image, prompt=system_prompt, show_thinking=show_thinking, do_sample=do_sample, text_temperature=text_temperature, max_new_tokens=max_new_tokens)
    sub_prompts, text_to_parse_for_steps = [], raw_llm_output
    think_tag_end, last_think_tag_pos = "</think>", raw_llm_output.rfind("</think>")
    if last_think_tag_pos != -1:
        segment_after_think = raw_llm_output[last_think_tag_pos + len(think_tag_end):].strip()
        if segment_after_think: print(f"Found '{think_tag_end}'. Parsing steps from text segment: '{segment_after_think[:200]}...'"); text_to_parse_for_steps = segment_after_think
        else: print(f"Found '{think_tag_end}' but no text followed. Will parse full output if segment parsing fails.")
    else: print(f"No '{think_tag_end}' tag found. Parsing full output for steps.")
    matches = re.findall(r"^[Ss]tep\s*\d+:\s*(.+)", text_to_parse_for_steps, re.MULTILINE)
    if matches: sub_prompts = [match.strip() for match in matches]; print(f"Parsed sub-prompts (regex on target segment): {sub_prompts}")
    if not sub_prompts and text_to_parse_for_steps is not raw_llm_output:
        print("Regex on segmented text failed. Retrying regex on full raw output.")
        matches_full = re.findall(r"^[Ss]tep\s*\d+:\s*(.+)", raw_llm_output, re.MULTILINE)
        if matches_full: sub_prompts = [match.strip() for match in matches_full]; print(f"Parsed sub-prompts (regex on full output - fallback): {sub_prompts}")
    if not sub_prompts:
        print("Regex parsing failed. Using basic newline split fallback on full raw output.")
        potential_steps = [line.strip() for line in raw_llm_output.split('\n') if line.strip()]
        for step_like_line in potential_steps:
            cleaned_step_match = re.match(r"^[Ss]tep\s*\d+:\s*(.+)", step_like_line)
            if cleaned_step_match: cleaned_step = cleaned_step_match.group(1).strip(); sub_prompts.append(cleaned_step) if cleaned_step else None
        print(f"Parsed sub-prompts (newline fallback): {sub_prompts}")
    if not sub_prompts: print("LLM task decomposition critical failure: no usable steps found."); return [user_prompt], f"[CRITICAL FAILURE] Could not parse steps from LLM output:\n{raw_llm_output}"
    if len(sub_prompts) > num_steps_to_generate: print(f"LLM generated {len(sub_prompts)} steps, but requested {num_steps_to_generate}. Truncating."); sub_prompts = sub_prompts[:num_steps_to_generate]
    elif len(sub_prompts) < num_steps_to_generate and sub_prompts != [user_prompt]: print(f"Warning: LLM generated {len(sub_prompts)} steps, but {num_steps_to_generate} were requested.")
    return sub_prompts, raw_llm_output

def get_next_project_folder_path(base_task_breakdown_dir):
    os.makedirs(base_task_breakdown_dir, exist_ok=True)
    existing_projects = [d for d in os.listdir(base_task_breakdown_dir) if os.path.isdir(os.path.join(base_task_breakdown_dir, d))]
    max_num = 0
    for project_name in existing_projects:
        match = re.match(r"(\d{3})_.+", project_name); max_num = max(max_num, int(match.group(1))) if match else max_num
    next_num_str = f"{max_num + 1:03d}"; timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    project_folder_name = f"{next_num_str}_{timestamp_str}"; project_folder_path = os.path.join(base_task_breakdown_dir, project_folder_name)
    os.makedirs(project_folder_path, exist_ok=True); return project_folder_path

def get_xy_plot_run_subdir(base_plot_dir):
    timestamp = datetime.datetime.now().strftime("plot_run_%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(base_plot_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

XY_PLOT_PARAM_MAPPING = {} 
current_tab = "t2i"
NO_SELECTION_STR = "(No Selection)"
PROMPT_SR_ORIGINAL_PLACEHOLDER = "__PROMPT_SR_ORIGINAL__" 

def parse_xy_value_string(value_str, param_type, param_name=""):
    if not value_str.strip(): return []
    if param_name == "Prompt S/R":
        parts = [p.strip() for p in value_str.split(',')]
        if not parts: gr.Warning(f"Prompt S/R value string is empty."); return []
        if len(parts) == 1: return [PROMPT_SR_ORIGINAL_PLACEHOLDER, parts[0]]
        return [PROMPT_SR_ORIGINAL_PLACEHOLDER] + parts[1:]
    values = []
    try:
        for item_str in value_str.split(','):
            item_str = item_str.strip()
            if '-' in item_str and param_type in [int, float]:
                range_parts = item_str.split('-');
                if len(range_parts) == 2:
                    start, end = param_type(range_parts[0]), param_type(range_parts[1])
                    if start > end: start, end = end, start
                    if param_type == int: values.extend(list(range(start, end + 1)))
                    else: values.extend(np.linspace(start, end, num=max(2, int(abs(end-start)/0.5) if abs(end-start)>0.1 else 5 )).tolist())
                    continue
            values.append(param_type(item_str))
        return values
    except ValueError: gr.Warning(f"Invalid value string for X/Y plot: '{value_str}'."); return []
    except Exception as e: gr.Warning(f"Error parsing X/Y plot values '{value_str}': {e}"); return []

def get_label_display_value(val):
    if val == PROMPT_SR_ORIGINAL_PLACEHOLDER: return "Original"
    if isinstance(val, float): return f"{val:.2f}"
    s_val = str(val); return s_val[:20] + '...' if len(s_val) > 23 else s_val

def assemble_xy_plot_image(images_matrix_flat, x_param_name, x_values, y_param_name, y_values, first_img_width, first_img_height):
    if not images_matrix_flat or not first_img_width or not first_img_height: return None
    x_axis_active = x_param_name != NO_SELECTION_STR and x_values
    y_axis_active = y_param_name != NO_SELECTION_STR and y_values
    num_cols = len(x_values) if x_axis_active else 1
    num_rows = len(y_values) if y_axis_active else 1
    label_font_size = 28; label_padding = 20 # Increased font and padding
    try: font = ImageFont.truetype("arial.ttf", label_font_size)
    except IOError: font = ImageFont.load_default(); label_font_size = font.getsize("M")[1] if hasattr(font, "getsize") else 15 
    x_label_height = label_font_size + label_padding if x_axis_active else 0; y_label_width = 0
    if y_axis_active:
        max_y_label_len = 0
        for v in y_values: 
            text = f"{y_param_name}: {get_label_display_value(v)}"
            try: bbox = font.getbbox(text) if hasattr(font, "getbbox") else (0,0, len(text) * label_font_size * 0.6, label_font_size)
            except AttributeError: bbox = (0,0, len(text) * label_font_size * 0.6, label_font_size)
            max_y_label_len = max(max_y_label_len, bbox[2] - bbox[0])
        y_label_width = max_y_label_len + label_padding
    total_width = y_label_width + num_cols * first_img_width; total_height = x_label_height + num_rows * first_img_height
    grid_image = Image.new('RGB', (total_width, total_height), 'white'); draw = ImageDraw.Draw(grid_image)
    if y_axis_active:
        for i, y_val in enumerate(y_values):
            label_text = f"{y_param_name}: {get_label_display_value(y_val)}"
            try: bbox_l = font.getbbox(label_text) if hasattr(font, "getbbox") else (0,0,0,label_font_size)
            except AttributeError: bbox_l = (0,0,0,label_font_size)
            text_height = bbox_l[3] - bbox_l[1] if (bbox_l and len(bbox_l)==4 and bbox_l[3] > bbox_l[1]) else label_font_size
            text_y_pos = x_label_height + i * first_img_height + (first_img_height - text_height) // 2
            draw.text((label_padding//2, text_y_pos), label_text, fill="black", font=font)
    if x_axis_active:
        for j, x_val in enumerate(x_values):
            label_text = f"{x_param_name}: {get_label_display_value(x_val)}"
            try: bbox = font.getbbox(label_text) if hasattr(font, "getbbox") else (0,0, len(label_text) * label_font_size * 0.6, label_font_size)
            except AttributeError: bbox = (0,0, len(label_text) * label_font_size * 0.6, label_font_size) 
            text_width = bbox[2] - bbox[0]; text_x_pos = y_label_width + j * first_img_width + (first_img_width - text_width) // 2
            draw.text((text_x_pos, label_padding//2), label_text, fill="black", font=font)
    for i in range(num_rows):
        for j in range(num_cols):
            img_idx = i * num_cols + j
            if img_idx < len(images_matrix_flat):
                img = images_matrix_flat[img_idx]
                if img and isinstance(img, Image.Image):
                    img_resized = img.resize((first_img_width, first_img_height)); paste_x = y_label_width + j * first_img_width; paste_y = x_label_height + i * first_img_height
                    grid_image.paste(img_resized, (paste_x, paste_y))
                else: 
                    rect_x0, rect_y0 = y_label_width + j * first_img_width, x_label_height + i * first_img_height; rect_x1, rect_y1 = rect_x0 + first_img_width, rect_y0 + first_img_height
                    draw.rectangle([(rect_x0, rect_y0), (rect_x1, rect_y1)], outline="red", fill="lightgray"); draw.text((rect_x0 + 5, rect_y0 + 5), "Error", fill="red", font=font)
    return grid_image

with gr.Blocks() as demo:
    gr.Markdown("""<div> <img src="https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/banner.png" alt="BAGEL" width="380"/> </div>""")
    
    with gr.Tab("📝 Text to Image") as tab_t2i_obj:
        txt_input_t2i = gr.Textbox(label="Prompt", value="A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere.")
        with gr.Row(): show_thinking_t2i = gr.Checkbox(label="Thinking", value=False)
        with gr.Accordion("Inference Hyperparameters", open=False) as t2i_hyperparams_accordion:
            with gr.Group():
                with gr.Row(): 
                    seed_t2i = gr.Slider(minimum=0, maximum=1000000, value=0, step=1, label="Seed", info="0 for random seed, positive for reproducible results")
                    image_ratio_t2i = gr.Dropdown(choices=[("1:1","1:1"), ("4:3","4:3"), ("3:4","3:4"), ("16:9","16:9"), ("9:16","9:16")], value="1:1", label="Image Ratio", info="The longer size is fixed to 1024")
                    batch_size_t2i = gr.Slider(minimum=1, maximum=8, value=1, step=1, interactive=True, label="Batch Size", info="Number of images to generate sequentially.")
                with gr.Row(): 
                    cfg_text_scale_t2i = gr.Slider(minimum=1.0, maximum=8.0, value=4.0, step=0.1, interactive=True, label="CFG Text Scale", info="Controls how strongly the model follows the text prompt (4.0-8.0)")
                    cfg_interval_t2i = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.1, label="CFG Interval", info="Start of CFG application interval (end is fixed at 1.0)")
                with gr.Row(): 
                    cfg_renorm_type_t2i = gr.Dropdown(choices=[("global","global"), ("local","local"), ("text_channel","text_channel")], value="global", label="CFG Renorm Type", info="If the genrated image is blurry, use 'global'")
                    cfg_renorm_min_t2i = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True, label="CFG Renorm Min", info="1.0 disables CFG-Renorm")
                with gr.Row(): 
                    num_timesteps_t2i = gr.Slider(minimum=10, maximum=100, value=50, step=5, interactive=True, label="Timesteps", info="Total denoising steps")
                    timestep_shift_t2i = gr.Slider(minimum=1.0, maximum=5.0, value=3.0, step=0.5, interactive=True, label="Timestep Shift", info="Higher values for layout, lower for details")
                thinking_params_t2i_group = gr.Group(visible=False)
                with thinking_params_t2i_group:
                    with gr.Row(): 
                        do_sample_t2i = gr.Checkbox(label="Sampling", value=False, info="Enable sampling for text generation")
                        max_think_token_n_t2i = gr.Slider(minimum=64, maximum=4006, value=1024, step=64, interactive=True, label="Max Think Tokens", info="Maximum number of tokens for thinking")
                        text_temperature_t2i = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.1, interactive=True, label="Temperature", info="Controls randomness in text generation")
        with gr.Accordion("🔀 X/Y Plot", open=False) as t2i_xy_plot_accordion:
            enable_xy_plot_t2i = gr.Checkbox(label="Enable X/Y Plot", value=False, info="Overrides Batch Size and selected main hyperparameters.")
            with gr.Row(): 
                xy_x_param_t2i = gr.Dropdown(label="X-axis Parameter", choices=[(NO_SELECTION_STR, NO_SELECTION_STR)], value=NO_SELECTION_STR)
                xy_x_values_t2i = gr.Textbox(label="X-values (comma-separated)", placeholder="e.g., 1.0,1.5 or Prompt S/R: find,r1,r2")
            with gr.Row(): 
                xy_y_param_t2i = gr.Dropdown(label="Y-axis Parameter", choices=[(NO_SELECTION_STR, NO_SELECTION_STR)], value=NO_SELECTION_STR)
                xy_y_values_t2i = gr.Textbox(label="Y-values (comma-separated)", placeholder="e.g., 10,20 or Prompt S/R: find,r1,r2")
        thinking_output_t2i = gr.Textbox(label="Thinking Process", visible=False, lines=5)
        img_output_t2i_gallery = gr.Gallery(label="Generated Images", columns=2, object_fit="contain", height="auto", preview=True, visible=True)
        xy_plot_output_t2i_img = gr.Image(label="X/Y Plot Result", type="pil", visible=False)
        gen_btn_t2i = gr.Button("Generate", variant="primary")

    with gr.Tab("🖌️ Image Edit") as tab_edit_obj:
        with gr.Row():
            with gr.Column(scale=1): 
                edit_image_input = gr.Image(label="Input Image", value=load_example_image('test_images/women.jpg'), type="pil")
                edit_prompt = gr.Textbox(label="Prompt", value="She boards a modern subway, quietly reading a folded newspaper, wearing the same clothes.")
            with gr.Column(scale=1): 
                edit_image_output_gallery = gr.Gallery(label="Result", columns=2, object_fit="contain", height="auto", preview=True, visible=True)
                edit_thinking_output = gr.Textbox(label="Thinking Process", visible=False, lines=10)
        with gr.Row(): 
            edit_show_thinking = gr.Checkbox(label="Thinking", value=False)
            enable_task_breakdown = gr.Checkbox(label="Enable Task Breakdown (Experimental)", value=False, info="LLM will try to break the prompt into N steps and apply them sequentially.")
        with gr.Accordion("Inference Hyperparameters", open=False) as edit_hyperparams_accordion:
            with gr.Group():
                with gr.Row(): 
                    edit_seed = gr.Slider(minimum=0, maximum=1000000, value=0, step=1, interactive=True, label="Seed", info="0 for random seed, positive for reproducible results")
                    edit_cfg_text_scale = gr.Slider(minimum=1.0, maximum=8.0, value=4.0, step=0.1, interactive=True, label="CFG Text Scale", info="Controls how strongly the model follows the text prompt")
                    edit_batch_size = gr.Slider(minimum=1, maximum=4, value=1, step=1, interactive=True, label="Batch Size", info="Number of images to generate sequentially.")
                with gr.Row(): 
                    edit_cfg_img_scale = gr.Slider(minimum=1.0, maximum=4.0, value=2.0, step=0.1, interactive=True, label="CFG Image Scale", info="Controls how much the model preserves input image details")
                    edit_cfg_interval = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True, label="CFG Interval", info="Start of CFG application interval (end is fixed at 1.0)")
                with gr.Row(): 
                    edit_cfg_renorm_type = gr.Dropdown(choices=[("global","global"), ("local","local"), ("text_channel","text_channel")], value="text_channel", label="CFG Renorm Type", info="If the genrated image is blurry, use 'global'")
                    edit_cfg_renorm_min = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True, label="CFG Renorm Min", info="1.0 disables CFG-Renorm")
                with gr.Row(): 
                    edit_num_timesteps = gr.Slider(minimum=10, maximum=100, value=50, step=5, interactive=True, label="Timesteps", info="Total denoising steps")
                    edit_timestep_shift = gr.Slider(minimum=1.0, maximum=10.0, value=3.0, step=0.5, interactive=True, label="Timestep Shift", info="Higher values for layout, lower for details")
                with gr.Accordion("LLM Task Breakdown Parameters", open=False, visible=False) as task_breakdown_params_accordion:
                     gr.Markdown("These parameters control the LLM when it breaks down your prompt.")
                     decompose_num_steps = gr.Slider(minimum=1, maximum=3, value=3, step=1, interactive=True, label="Number of Sub-steps", info="How many steps the LLM should try to generate.")
                     decompose_do_sample = gr.Checkbox(label="Sampling (Decomposition)", value=True, info="Enable sampling for LLM step generation.")
                     decompose_max_tokens = gr.Slider(minimum=64, maximum=1024, value=1024, step=32, interactive=True, label="Max Tokens (Decomposition)", info="Max tokens for LLM to generate steps.")
                     decompose_temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.1, step=0.1, interactive=True, label="Temperature (Decomposition)", info="Randomness for LLM step generation.")
                edit_thinking_params_group = gr.Group(visible=False)
                with edit_thinking_params_group:
                    with gr.Row(): 
                        edit_do_sample_img = gr.Checkbox(label="Sampling (Image Edit)", value=False, info="Enable sampling for image edit thinking text.")
                        edit_max_think_token_n_img = gr.Slider(minimum=64, maximum=4006, value=1024, step=64, interactive=True, label="Max Think Tokens (Image Edit)", info="Max tokens for image edit thinking.")
                        edit_text_temperature_img = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.1, interactive=True, label="Temperature (Image Edit)", info="Controls randomness in image edit thinking.")
        with gr.Accordion("🔀 X/Y Plot", open=False) as edit_xy_plot_accordion:
            enable_xy_plot_edit = gr.Checkbox(label="Enable X/Y Plot", value=False, info="Overrides Task Breakdown, and selected main hyperparameters.")
            with gr.Row(): 
                xy_x_param_edit = gr.Dropdown(label="X-axis Parameter", choices=[(NO_SELECTION_STR, NO_SELECTION_STR)], value=NO_SELECTION_STR)
                xy_x_values_edit = gr.Textbox(label="X-values (comma-separated)", placeholder="e.g., 1.0,1.5 or Prompt S/R: find,r1,r2")
            with gr.Row(): 
                xy_y_param_edit = gr.Dropdown(label="Y-axis Parameter", choices=[(NO_SELECTION_STR, NO_SELECTION_STR)], value=NO_SELECTION_STR)
                xy_y_values_edit = gr.Textbox(label="Y-values (comma-separated)", placeholder="e.g., 0.1,0.2 or Prompt S/R: find,r1,r2")
        xy_plot_output_edit_img = gr.Image(label="X/Y Plot Result", type="pil", visible=False)
        edit_btn = gr.Button("Submit", variant="primary")

    XY_PLOT_PARAM_MAPPING.update({
        "Seed": {"var": "seed", "type": int, "t2i_slider": seed_t2i, "edit_slider": edit_seed},
        "CFG Text Scale": {"var": "cfg_text_scale", "type": float, "t2i_slider": cfg_text_scale_t2i, "edit_slider": edit_cfg_text_scale},
        "CFG Interval": {"var": "cfg_interval", "type": float, "t2i_slider": cfg_interval_t2i, "edit_slider": edit_cfg_interval},
        "Timesteps": {"var": "num_timesteps", "type": int, "t2i_slider": num_timesteps_t2i, "edit_slider": edit_num_timesteps},
        "Timestep Shift": {"var": "timestep_shift", "type": float, "t2i_slider": timestep_shift_t2i, "edit_slider": edit_timestep_shift},
        "Prompt S/R": {"var": "prompt_sr", "type": str, "t2i_slider": None, "edit_slider": None},
        "CFG Image Scale": {"var": "cfg_img_scale", "type": float, "t2i_slider": None, "edit_slider": edit_cfg_img_scale},
    })

    xy_plot_choices_t2i_list = [(k, k) for k in XY_PLOT_PARAM_MAPPING.keys() if XY_PLOT_PARAM_MAPPING[k]["t2i_slider"] is not None or k == "Prompt S/R"]
    xy_plot_choices_edit_list = [(k, k) for k in XY_PLOT_PARAM_MAPPING.keys() if XY_PLOT_PARAM_MAPPING[k]["edit_slider"] is not None or k == "Prompt S/R"]

    xy_x_param_t2i.choices = [(NO_SELECTION_STR, NO_SELECTION_STR)] + xy_plot_choices_t2i_list
    xy_y_param_t2i.choices = [(NO_SELECTION_STR, NO_SELECTION_STR)] + xy_plot_choices_t2i_list
    xy_x_param_edit.choices = [(NO_SELECTION_STR, NO_SELECTION_STR)] + xy_plot_choices_edit_list
    xy_y_param_edit.choices = [(NO_SELECTION_STR, NO_SELECTION_STR)] + xy_plot_choices_edit_list
    
    show_thinking_t2i.change(fn=lambda show: (gr.update(visible=show), gr.update(visible=show)), inputs=[show_thinking_t2i], outputs=[thinking_output_t2i, thinking_params_t2i_group])
    enable_xy_plot_t2i.change(fn=lambda enabled: (gr.update(visible=enabled), gr.update(visible=not enabled)), inputs=[enable_xy_plot_t2i], outputs=[xy_plot_output_t2i_img, img_output_t2i_gallery])

    def process_text_to_image_ui(prompt_main, show_thinking_main, cfg_text_scale_main, cfg_interval_main, timestep_shift_main, num_timesteps_main, cfg_renorm_min_main, cfg_renorm_type_main, max_think_token_n_main, do_sample_main, text_temperature_main, seed_main, image_ratio_main, batch_size_main, enable_xy, xy_x_param_name, xy_x_values_str, xy_y_param_name, xy_y_values_str):
        global current_tab; current_tab = "t2i"
        xy_plot_run_save_dir = None 

        if not enable_xy:
            all_images, all_thinking_texts, base_seed = [], [], int(seed_main)
            txt2img_save_dir = os.path.join(args.output_dir, "txt2img"); os.makedirs(txt2img_save_dir, exist_ok=True)
            for i in range(int(batch_size_main)):
                current_iteration_seed = base_seed + i if base_seed > 0 else random.randint(1, 2**32 -1)
                image_result, thinking_result = text_to_image(prompt_main, show_thinking_main, cfg_text_scale_main, cfg_interval_main, timestep_shift_main, num_timesteps_main, cfg_renorm_min_main, cfg_renorm_type_main, max_think_token_n_main, do_sample_main, text_temperature_main, seed=current_iteration_seed, image_ratio=image_ratio_main, save_to_dir=txt2img_save_dir, batch_index=i)
                if image_result: all_images.append(image_result)
                if thinking_result: all_thinking_texts.append(f"Batch {i+1} (Seed: {current_iteration_seed}) Thinking:\n{thinking_result}")
            return all_images, "\n\n".join(all_thinking_texts) if all_thinking_texts else "", gr.update(visible=False)

        gr.Info("X/Y Plot generation started for Text-to-Image...")
        xy_plot_base_dir = os.path.join(args.output_dir, "x_y_plot")
        xy_plot_run_save_dir = get_xy_plot_run_subdir(xy_plot_base_dir)

        plot_images_flat = []
        x_param_info = XY_PLOT_PARAM_MAPPING.get(xy_x_param_name) if xy_x_param_name != NO_SELECTION_STR else None
        y_param_info = XY_PLOT_PARAM_MAPPING.get(xy_y_param_name) if xy_y_param_name != NO_SELECTION_STR else None
        
        x_vals = parse_xy_value_string(xy_x_values_str, x_param_info["type"] if x_param_info else str, xy_x_param_name) if x_param_info else []
        y_vals = parse_xy_value_string(xy_y_values_str, y_param_info["type"] if y_param_info else str, xy_y_param_name) if y_param_info else []

        if not x_vals and not y_vals: return [], "Please select at least one X/Y axis parameter and provide values.", gr.update(value=None, visible=True)

        x_sr_search_term = None
        if x_param_info and xy_x_param_name == "Prompt S/R" and xy_x_values_str:
             parts = [p.strip() for p in xy_x_values_str.split(',')]
             if parts: x_sr_search_term = parts[0]

        y_sr_search_term = None
        if y_param_info and xy_y_param_name == "Prompt S/R" and xy_y_values_str:
            parts = [p.strip() for p in xy_y_values_str.split(',')]
            if parts: y_sr_search_term = parts[0]

        default_params = {"prompt": prompt_main, "show_thinking": False, "cfg_text_scale": cfg_text_scale_main, "cfg_interval": cfg_interval_main, "num_timesteps": num_timesteps_main, "timestep_shift": timestep_shift_main, "cfg_renorm_min": cfg_renorm_min_main, "cfg_renorm_type": cfg_renorm_type_main, "max_think_token_n": max_think_token_n_main, "do_sample": do_sample_main, "text_temperature": text_temperature_main, "seed": int(seed_main), "image_ratio": image_ratio_main, "is_xy_plot_image": True, "save_to_dir": xy_plot_run_save_dir}
        
        x_axis_values = x_vals if x_param_info else [None]; y_axis_values = y_vals if y_param_info else [None]
        first_image_for_size = None

        for i, y_current_val in enumerate(y_axis_values):
            for j, x_current_val in enumerate(x_axis_values):
                current_cell_params = default_params.copy(); param_summary = []
                current_prompt_for_cell = prompt_main
                cell_filename_prefix = f"cell_y{i}_x{j}"

                if y_param_info and y_current_val is not None:
                    if xy_y_param_name == "Prompt S/R":
                        if y_current_val != PROMPT_SR_ORIGINAL_PLACEHOLDER and y_sr_search_term:
                            current_prompt_for_cell = current_prompt_for_cell.replace(y_sr_search_term, str(y_current_val))
                        search_val_display = y_sr_search_term if y_sr_search_term else ""
                        replace_val_display = get_label_display_value(y_current_val)
                        param_summary.append(f"PrmptY:S='{search_val_display}',R='{replace_val_display}'")
                    else:
                        current_cell_params[y_param_info["var"]] = y_current_val; param_summary.append(f"{xy_y_param_name}={y_current_val}")
                
                if x_param_info and x_current_val is not None:
                    if xy_x_param_name == "Prompt S/R":
                        if x_current_val != PROMPT_SR_ORIGINAL_PLACEHOLDER and x_sr_search_term:
                            current_prompt_for_cell = current_prompt_for_cell.replace(x_sr_search_term, str(x_current_val))
                        search_val_display = x_sr_search_term if x_sr_search_term else ""
                        replace_val_display = get_label_display_value(x_current_val)
                        param_summary.append(f"PrmptX:S='{search_val_display}',R='{replace_val_display}'")
                    else:
                        current_cell_params[x_param_info["var"]] = x_current_val; param_summary.append(f"{xy_x_param_name}={x_current_val}")
                
                current_cell_params["prompt"] = current_prompt_for_cell
                
                is_seed_x_axis = x_param_info and x_param_info["var"] == "seed"
                is_seed_y_axis = y_param_info and y_param_info["var"] == "seed"
                if not is_seed_x_axis and not is_seed_y_axis:
                    current_cell_params["seed"] = int(seed_main) 

                current_cell_params["xy_plot_filename_prefix"] = cell_filename_prefix

                print(f"Generating T2I X/Y Plot: {cell_filename_prefix} with {', '.join(param_summary) if param_summary else 'base params'}, Seed: {current_cell_params['seed']}")
                img, _ = text_to_image(**current_cell_params)
                plot_images_flat.append(img)
                if not first_image_for_size and img: first_image_for_size = img
                
        if not plot_images_flat or not first_image_for_size: return [], "Failed to generate any images for the X/Y plot.", gr.update(value=None, visible=True)
        
        actual_x_param_name = xy_x_param_name if x_param_info else NO_SELECTION_STR
        actual_y_param_name = xy_y_param_name if y_param_info else NO_SELECTION_STR
        label_x_vals = x_axis_values if x_param_info else []; label_y_vals = y_axis_values if y_param_info else []

        final_grid_image = assemble_xy_plot_image(plot_images_flat, actual_x_param_name, label_x_vals, actual_y_param_name, label_y_vals, first_image_for_size.width, first_image_for_size.height)
        if final_grid_image:
            plot_grid_filename = f"xy_plot_t2i_grid_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_path = os.path.join(xy_plot_run_save_dir, plot_grid_filename) 
            try: final_grid_image.save(save_path); print(f"Saved T2I X/Y Plot Grid: {save_path}")
            except Exception as e: print(f"Error saving T2I X/Y Plot Grid to {save_path}: {e}")
            return [], "X/Y Plot Generated. Individual cell images saved in subfolder.", gr.update(value=final_grid_image, visible=True)
        else: return [], "Failed to assemble X/Y plot image.", gr.update(value=None, visible=True)

    gen_btn_t2i.click(fn=process_text_to_image_ui, inputs=[txt_input_t2i, show_thinking_t2i, cfg_text_scale_t2i, cfg_interval_t2i, timestep_shift_t2i, num_timesteps_t2i, cfg_renorm_min_t2i, cfg_renorm_type_t2i, max_think_token_n_t2i, do_sample_t2i, text_temperature_t2i, seed_t2i, image_ratio_t2i, batch_size_t2i, enable_xy_plot_t2i, xy_x_param_t2i, xy_x_values_t2i, xy_y_param_t2i, xy_y_values_t2i], outputs=[img_output_t2i_gallery, thinking_output_t2i, xy_plot_output_t2i_img])

    edit_show_thinking.change(fn=lambda show: (gr.update(visible=show), gr.update(visible=show)), inputs=[edit_show_thinking], outputs=[edit_thinking_output, edit_thinking_params_group])
    enable_task_breakdown.change(fn=lambda show: gr.update(visible=show), inputs=[enable_task_breakdown], outputs=[task_breakdown_params_accordion])
    enable_xy_plot_edit.change(fn=lambda enabled: (gr.update(visible=enabled), gr.update(visible=not enabled)), inputs=[enable_xy_plot_edit], outputs=[xy_plot_output_edit_img, edit_image_output_gallery]) 

    def process_edit_image_ui(image_input_pil, user_main_prompt_main, show_thinking_main, enable_breakdown_main, cfg_text_scale_main, cfg_img_scale_main, cfg_interval_main, timestep_shift_main, num_timesteps_main, cfg_renorm_min_main, cfg_renorm_type_main, edit_do_sample_main, edit_max_think_tokens_main, edit_temp_main, seed_main, batch_size_main, decompose_num_s_main, decompose_sample_main, decompose_max_t_main, decompose_temp_main, enable_xy_e, xy_x_param_e_name, xy_x_values_str_e, xy_y_param_e_name, xy_y_values_str_e):
        global current_tab; current_tab = "edit"
        xy_plot_run_save_dir_edit = None

        if image_input_pil is None: gr.Warning("Please upload an image for editing."); return [], "Please upload an image.", gr.update(visible=False)
        if not enable_xy_e:
            all_final_images_for_gallery, all_batch_thinking_texts_accumulated, base_seed = [], [], int(seed_main)
            sub_prompts_from_llm, decomposition_llm_thinking_text = [user_main_prompt_main], ""; current_run_save_dir = ""
            if enable_breakdown_main:
                base_task_dir = os.path.join(args.output_dir, "task_breakdown"); current_run_save_dir = get_next_project_folder_path(base_task_dir)
                sub_prompts_from_llm, decomposition_llm_thinking_text_raw = decompose_task_with_llm(user_main_prompt_main, image_input_pil.copy(), True, decompose_sample_main, decompose_temp_main, decompose_max_t_main, decompose_num_s_main)
                initial_decomposition_summary = f"[INFO] LLM Decomposed into {len(sub_prompts_from_llm)} steps:\n" + "\n".join([f"  - {sp}" for sp in sub_prompts_from_llm])
                if sub_prompts_from_llm == [user_main_prompt_main] and "CRITICAL FAILURE" not in decomposition_llm_thinking_text_raw: decomposition_llm_thinking_text_raw = (decomposition_llm_thinking_text_raw or "") + "\n[INFO] Task decomposition might have failed..."
                decomposition_llm_thinking_text = f"--- LLM Task Decomposition Process (Project: {os.path.basename(current_run_save_dir)}) ---\n{decomposition_llm_thinking_text_raw}\n{initial_decomposition_summary}"
            else: current_run_save_dir = os.path.join(args.output_dir, "img2img"); os.makedirs(current_run_save_dir, exist_ok=True)
            for batch_idx in range(int(batch_size_main)):
                current_image_being_processed = image_input_pil.copy(); current_batch_item_seed = base_seed + batch_idx if base_seed > 0 else random.randint(1, 2**32 -1)
                thinking_for_this_batch_run = [f"--- Starting Batch Item {batch_idx+1}/{batch_size_main} (Seed: {current_batch_item_seed}) ---"]
                if enable_breakdown_main: thinking_for_this_batch_run.append(decomposition_llm_thinking_text)
                active_prompts_for_this_run = sub_prompts_from_llm if enable_breakdown_main else [user_main_prompt_main]
                for step_num, sub_prompt_text in enumerate(active_prompts_for_this_run):
                    thinking_for_this_batch_run.append(f"Processing Sub-step {step_num+1}/{len(active_prompts_for_this_run)}: '{sub_prompt_text}'"); print(f"Batch {batch_idx+1}, Seed {current_batch_item_seed}, Sub-step {step_num+1}: '{sub_prompt_text}'")
                    edited_sub_step_image_pil, thinking_text_from_sub_step = edit_image(current_image_being_processed, sub_prompt_text, show_thinking_main, cfg_text_scale_main, cfg_img_scale_main, cfg_interval_main, timestep_shift_main, num_timesteps_main, cfg_renorm_min_main, cfg_renorm_type_main, edit_max_think_tokens_main, edit_do_sample_main, edit_temp_main, seed=current_batch_item_seed, save_to_dir=current_run_save_dir, batch_index=batch_idx, sub_step_idx=step_num)
                    if edited_sub_step_image_pil and isinstance(edited_sub_step_image_pil, Image.Image): current_image_being_processed = edited_sub_step_image_pil; thinking_for_this_batch_run.append(f"  Sub-step {step_num+1} Thinking: {thinking_text_from_sub_step}") if show_thinking_main and thinking_text_from_sub_step else None
                    else: error_msg = f"  Sub-step {step_num+1} FAILED. {edited_sub_step_image_pil if isinstance(edited_sub_step_image_pil, str ) else 'No valid image.'}"; thinking_for_this_batch_run.append(error_msg); print(error_msg); break
                all_final_images_for_gallery.append(current_image_being_processed); all_batch_thinking_texts_accumulated.append("\n".join(thinking_for_this_batch_run))
            return all_final_images_for_gallery, "\n\n<<< --- END OF BATCH ITEM --- >>>\n\n".join(all_batch_thinking_texts_accumulated), gr.update(visible=False)

        gr.Info("X/Y Plot generation started for Image Edit...")
        xy_plot_base_dir_edit = os.path.join(args.output_dir, "x_y_plot")
        xy_plot_run_save_dir_edit = get_xy_plot_run_subdir(xy_plot_base_dir_edit)

        plot_images_flat_edit = []
        x_param_info_e = XY_PLOT_PARAM_MAPPING.get(xy_x_param_e_name) if xy_x_param_e_name != NO_SELECTION_STR else None
        y_param_info_e = XY_PLOT_PARAM_MAPPING.get(xy_y_param_e_name) if xy_y_param_e_name != NO_SELECTION_STR else None
        x_vals_e = parse_xy_value_string(xy_x_values_str_e, x_param_info_e["type"] if x_param_info_e else str, xy_x_param_e_name) if x_param_info_e else []; y_vals_e = parse_xy_value_string(xy_y_values_str_e, y_param_info_e["type"] if y_param_info_e else str, xy_y_param_e_name) if y_param_info_e else []
        if not x_vals_e and not y_vals_e: return [], "Please select at least one X/Y axis parameter and provide values for Image Edit.", gr.update(visible=True)
        
        x_prompt_sr_search_term_e = None
        if x_param_info_e and xy_x_param_e_name == "Prompt S/R" and xy_x_values_str_e:
            x_prompt_sr_search_term_e = xy_x_values_str_e.split(',')[0].strip()
        y_prompt_sr_search_term_e = None
        if y_param_info_e and xy_y_param_e_name == "Prompt S/R" and xy_y_values_str_e:
            y_prompt_sr_search_term_e = xy_y_values_str_e.split(',')[0].strip()

        default_params_edit = {"image": image_input_pil, "prompt": user_main_prompt_main, "show_thinking": False, "cfg_text_scale": cfg_text_scale_main, "cfg_img_scale": cfg_img_scale_main, "cfg_interval": cfg_interval_main, "num_timesteps": num_timesteps_main, "timestep_shift": timestep_shift_main, "cfg_renorm_min": cfg_renorm_min_main, "cfg_renorm_type": cfg_renorm_type_main, "max_think_token_n": edit_max_think_tokens_main, "do_sample": edit_do_sample_main, "text_temperature": edit_temp_main, "seed": int(seed_main), "is_xy_plot_image": True, "save_to_dir": xy_plot_run_save_dir_edit}
        x_axis_values_e = x_vals_e if x_param_info_e else [None]; y_axis_values_e = y_vals_e if y_param_info_e else [None]
        first_image_for_size_edit = None

        for i, y_curr_e in enumerate(y_axis_values_e):
            for j, x_curr_e in enumerate(x_axis_values_e):
                current_cell_params_e = default_params_edit.copy(); param_summary_e = []
                current_prompt_for_cell_e = user_main_prompt_main
                cell_filename_prefix_e = f"cell_y{i}_x{j}"

                if y_param_info_e and y_curr_e is not None:
                    if xy_y_param_e_name == "Prompt S/R":
                        if y_curr_e != PROMPT_SR_ORIGINAL_PLACEHOLDER and y_prompt_sr_search_term_e:
                            current_prompt_for_cell_e = current_prompt_for_cell_e.replace(y_prompt_sr_search_term_e, str(y_curr_e))
                        search_val_display_e = y_prompt_sr_search_term_e if y_prompt_sr_search_term_e else ""
                        replace_val_display_e = get_label_display_value(y_curr_e)
                        param_summary_e.append(f"PrmptY:S='{search_val_display_e}',R='{replace_val_display_e}'")
                    else:
                        current_cell_params_e[y_param_info_e["var"]] = y_curr_e; param_summary_e.append(f"{xy_y_param_e_name}={y_curr_e}")
                
                if x_param_info_e and x_curr_e is not None:
                    if xy_x_param_e_name == "Prompt S/R":
                        if x_curr_e != PROMPT_SR_ORIGINAL_PLACEHOLDER and x_prompt_sr_search_term_e:
                            current_prompt_for_cell_e = current_prompt_for_cell_e.replace(x_prompt_sr_search_term_e, str(x_curr_e))
                        search_val_display_e = x_prompt_sr_search_term_e if x_prompt_sr_search_term_e else ""
                        replace_val_display_e = get_label_display_value(x_curr_e)
                        param_summary_e.append(f"PrmptX:S='{search_val_display_e}',R='{replace_val_display_e}'")
                    else:
                        current_cell_params_e[x_param_info_e["var"]] = x_curr_e; param_summary_e.append(f"{xy_x_param_e_name}={x_curr_e}")
                
                current_cell_params_e["prompt"] = current_prompt_for_cell_e

                is_seed_x_axis_e = x_param_info_e and x_param_info_e["var"] == "seed"
                is_seed_y_axis_e = y_param_info_e and y_param_info_e["var"] == "seed"
                if not is_seed_x_axis_e and not is_seed_y_axis_e:
                    current_cell_params_e["seed"] = int(seed_main) 
                
                current_cell_params_e["xy_plot_filename_prefix"] = cell_filename_prefix_e

                print(f"Generating Edit X/Y Plot: {cell_filename_prefix_e} with {', '.join(param_summary_e) if param_summary_e else 'base params'}, Seed: {current_cell_params_e['seed']}")
                img_e, _ = edit_image(**current_cell_params_e)
                plot_images_flat_edit.append(img_e)
                if not first_image_for_size_edit and img_e and isinstance(img_e, Image.Image): first_image_for_size_edit = img_e
                
        if not plot_images_flat_edit or not first_image_for_size_edit: return [], "Failed to generate any images for Edit X/Y plot.", gr.update(visible=True)
        
        actual_x_param_name_e = xy_x_param_e_name if x_param_info_e else NO_SELECTION_STR
        actual_y_param_name_e = xy_y_param_e_name if y_param_info_e else NO_SELECTION_STR
        label_x_vals_e = x_axis_values_e if x_param_info_e else []
        label_y_vals_e = y_axis_values_e if y_param_info_e else []

        final_grid_image_edit = assemble_xy_plot_image(plot_images_flat_edit, actual_x_param_name_e, label_x_vals_e, actual_y_param_name_e, label_y_vals_e, first_image_for_size_edit.width, first_image_for_size_edit.height)
        if final_grid_image_edit:
            plot_grid_filename_e = f"xy_plot_edit_grid_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_path_e = os.path.join(xy_plot_run_save_dir_edit, plot_grid_filename_e)
            try: final_grid_image_edit.save(save_path_e); print(f"Saved Edit X/Y Plot Grid: {save_path_e}")
            except Exception as e_save: print(f"Error saving Edit X/Y Plot Grid to {save_path_e}: {e_save}")
            return [final_grid_image_edit], "X/Y Plot Generated for Image Edit. Individual cell images saved in subfolder.", gr.update(value=final_grid_image_edit, visible=True) 
        else: return [], "Failed to assemble Edit X/Y plot image.", gr.update(visible=True)

    edit_btn.click(fn=process_edit_image_ui, inputs=[
        edit_image_input, edit_prompt, edit_show_thinking, enable_task_breakdown, 
        edit_cfg_text_scale, edit_cfg_img_scale, edit_cfg_interval, edit_timestep_shift, 
        edit_num_timesteps, edit_cfg_renorm_min, edit_cfg_renorm_type, 
        edit_do_sample_img, edit_max_think_token_n_img, edit_text_temperature_img, 
        edit_seed, edit_batch_size, decompose_num_steps, decompose_do_sample, decompose_max_tokens, decompose_temperature,
        enable_xy_plot_edit, xy_x_param_edit, xy_x_values_edit, xy_y_param_edit, xy_y_values_edit
        ], outputs=[edit_image_output_gallery, edit_thinking_output, xy_plot_output_edit_img]
    )

    with gr.Tab("🖼️ Image Understanding") as tab_und_obj:
        und_input_type = gr.Radio(
            choices=[("Single Image", "Single Image"), ("Batch (Files/ZIP)", "Batch (Files/ZIP)")], 
            label="Input Mode", 
            value="Single Image"
        )
        with gr.Row(visible=True) as und_single_image_row: # Initially visible
            img_input_und = gr.Image(label="Input Image", value=load_example_image('test_images/meme.jpg'), type="pil")
        with gr.Row(visible=False) as und_batch_files_row: # Initially hidden
            batch_input_und = gr.Files(label="Upload Images or ZIP file", file_types=["image", ".zip"])

        with gr.Column():
            understand_prompt = gr.Textbox(label="Prompt (applies to all images in batch)", value="Can someone explain this meme??")
            with gr.Row():
                understand_show_thinking = gr.Checkbox(label="Thinking", value=False)
            with gr.Accordion("Inference Hyperparameters", open=False):
                with gr.Row(): 
                    understand_do_sample = gr.Checkbox(label="Sampling", value=False)
                    understand_text_temperature = gr.Slider(0.0, 1.0, 0.3, step=0.05, label="Temperature")
                    understand_max_new_tokens = gr.Slider(64, 4096, 512, step=64, label="Max New Tokens")
            img_understand_btn = gr.Button("Submit", variant="primary")
        with gr.Column():
            txt_output_und = gr.Textbox(label="Result (for single image)", lines=10, visible=True)
            batch_log_und = gr.Textbox(label="Batch Process Log", lines=10, visible=False, interactive=False)
            batch_output_und_zip = gr.File(label="Download Batch Results (ZIP)", visible=False)

        def toggle_und_input_mode(mode_selected):
            is_single_mode = (mode_selected == "Single Image")
            return (
                gr.update(visible=is_single_mode),    # und_single_image_row
                gr.update(visible=not is_single_mode), # und_batch_files_row
                gr.update(visible=is_single_mode),    # txt_output_und
                gr.update(visible=not is_single_mode, value=""), # batch_log_und
                gr.update(visible=not is_single_mode, value=None)  # batch_output_und_zip
            )
        und_input_type.change(fn=toggle_und_input_mode, inputs=[und_input_type], outputs=[und_single_image_row, und_batch_files_row, txt_output_und, batch_log_und, batch_output_und_zip])
        
        def process_understanding_ui(input_mode, single_img_pil, batch_file_objs, prompt, show_thinking, do_sample_u, text_temperature_u, max_new_tokens_u):
            if input_mode == "Single Image":
                if single_img_pil is None: gr.Warning("Please upload an image for understanding."); yield "Please upload an image.", "", None; return
                if not isinstance(single_img_pil, Image.Image):
                    try: single_img_pil = Image.fromarray(single_img_pil)
                    except: gr.Warning("Invalid single image format."); yield "Invalid single image format.", "", None; return
                single_img_pil = pil_img2rgb(single_img_pil)
                result_text = _perform_image_understanding(single_img_pil, prompt, show_thinking, do_sample_u, text_temperature_u, max_new_tokens_u)
                yield result_text, "", None # Single text output, empty log, no zip
                return 

            elif input_mode == "Batch (Files/ZIP)":
                if not batch_file_objs: gr.Warning("Please upload files or a ZIP for batch processing."); yield "", "No files uploaded for batch processing.", None; return
                
                log_messages = ["Batch processing started..."]
                yield "", "\n".join(log_messages), None # Initial log update
                processed_files_for_zip_paths = [] 
                temp_batch_dir = tempfile.mkdtemp(); temp_extraction_dir = os.path.join(temp_batch_dir, "extracted_from_zip")
                os.makedirs(temp_extraction_dir, exist_ok=True)
                output_files_build_dir = os.path.join(temp_batch_dir, "outputs_for_zip")
                os.makedirs(output_files_build_dir, exist_ok=True)
                image_paths_to_process = []
                processed_image_counter = 0

                for file_obj in batch_file_objs:
                    original_filename = os.path.basename(file_obj.name) # For user-facing original name
                    temp_file_path = file_obj.name # Actual path to Gradio's temp file
                    
                    log_messages.append(f"Inspecting: {original_filename}"); print(f"Inspecting: {original_filename}")
                    yield "", "\n".join(log_messages), None
                    
                    if original_filename.lower().endswith(".zip"):
                        try:
                            with zipfile.ZipFile(temp_file_path, 'r') as zip_ref: zip_ref.extractall(temp_extraction_dir)
                            log_messages.append(f"  Extracted '{original_filename}'. Scanning for images..."); print(f"  Extracted '{original_filename}'.")
                            yield "", "\n".join(log_messages), None
                            temp_extracted_files = []
                            for root, _, files_in_zip in os.walk(temp_extraction_dir):
                                for f_in_zip in files_in_zip:
                                    if f_in_zip.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                                        extracted_file_path = os.path.join(root, f_in_zip)
                                        temp_extracted_files.append(extracted_file_path)
                            image_paths_to_process.extend(temp_extracted_files) # Add all found images
                        except zipfile.BadZipFile: log_messages.append(f"  Error: '{original_filename}' is invalid ZIP."); print(f"  Error: '{original_filename}' is invalid ZIP.")
                        except Exception as e_zip: log_messages.append(f"  Error processing ZIP '{original_filename}': {e_zip}"); print(f"  Error processing ZIP '{original_filename}': {e_zip}")
                    elif original_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        image_paths_to_process.append(temp_file_path) # Use Gradio's temp path directly
                
                if not image_paths_to_process:
                    shutil.rmtree(temp_batch_dir); log_messages.append("No valid image files found in uploads or ZIPs."); yield "", "\n".join(log_messages), None; return
                
                total_images = len(image_paths_to_process)
                log_messages.append(f"\nFound {total_images} images to process in total."); yield "", "\n".join(log_messages), None

                for idx, img_path in enumerate(image_paths_to_process):

                    
                    try:                        
                        img_pil = Image.open(img_path); img_pil_rgb = pil_img2rgb(img_pil)
                        
                        # Determine a unique base name for output files
                        img_filename_for_output = os.path.basename(img_path)
                        img_filename_base, img_ext = os.path.splitext(img_filename_for_output)

                        log_messages.append(f"\n[{idx+1}/{total_images}] Processing: {img_filename_for_output}"); print(f"Batch Understanding [{idx+1}/{total_images}]: {img_filename_for_output}")
                        yield "", "\n".join(log_messages), None
                        
                        # Copy original image to the build directory for zipping
                        img_copy_for_zip_path = os.path.join(output_files_build_dir, img_filename_for_output)
                        # If img_path is already in temp_extraction_dir (from a zip), copy. If it's a Gradio temp file, copy.
                        shutil.copy2(img_path, img_copy_for_zip_path)

                        understanding_text = _perform_image_understanding(img_pil_rgb, prompt, show_thinking, do_sample_u, text_temperature_u, max_new_tokens_u)
                        
                        txt_filename = f"{img_filename_base}.txt"; txt_filepath = os.path.join(output_files_build_dir, txt_filename)
                        with open(txt_filepath, 'w', encoding='utf-8') as f: f.write(understanding_text)
                        log_messages.append(f"  Generated: {txt_filename}"); processed_files_for_zip_paths.append((img_copy_for_zip_path, txt_filepath))
                        processed_image_counter += 1
                    except UnidentifiedImageError: log_messages.append(f"  Error: Cannot identify image file {os.path.basename(img_path)}. Skipping.")
                    except Exception as e_proc: log_messages.append(f"  Error processing {os.path.basename(img_path)}: {e_proc}")
                    yield "", "\n".join(log_messages), None
                
                if not processed_files_for_zip_paths:
                    shutil.rmtree(temp_batch_dir); log_messages.append("No files were successfully processed."); yield "", "\n".join(log_messages), None; return

                zip_filename_base = f"batch_understanding_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                batch_output_zip_dir = os.path.join(args.output_dir, "batch_understanding_zips"); os.makedirs(batch_output_zip_dir, exist_ok=True)
                final_zip_path_to_serve = os.path.join(batch_output_zip_dir, zip_filename_base)

                log_messages.append(f"\nCreating results ZIP: {zip_filename_base} with {processed_image_counter} processed images."); yield "", "\n".join(log_messages), None
                print(f"Creating results ZIP: {final_zip_path_to_serve}")
                with zipfile.ZipFile(final_zip_path_to_serve, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for img_p, txt_p in processed_files_for_zip_paths:
                        zf.write(img_p, arcname=os.path.basename(img_p))
                        zf.write(txt_p, arcname=os.path.basename(txt_p))
                
                shutil.rmtree(temp_batch_dir)
                log_messages.append("Batch processing complete. ZIP ready for download."); print("Batch processing complete.")
                yield "", "\n".join(log_messages), final_zip_path_to_serve 
                return
            
            yield "", "Invalid input mode.", None

        img_understand_btn.click(fn=process_understanding_ui, inputs=[und_input_type, img_input_und, batch_input_und, understand_prompt, understand_show_thinking, understand_do_sample, understand_text_temperature, understand_max_new_tokens], outputs=[txt_output_und, batch_log_und, batch_output_und_zip])

    # ... (Markdown footer and UI_TRANSLATIONS, apply_localization, __main__ remain the same)
    gr.Markdown("""<div style="display: flex; justify-content: flex-start; flex-wrap: wrap; gap: 10px;"> <a href="https://bagel-ai.org/"><img src="https://img.shields.io/badge/BAGEL-Website-0A66C2?logo=safari&logoColor=white" alt="BAGEL Website"/></a> <a href="https://arxiv.org/abs/2505.14683"><img src="https://img.shields.io/badge/BAGEL-Paper-red?logo=arxiv&logoColor=red" alt="BAGEL Paper on arXiv"/></a> <a href="https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT"><img src="https://img.shields.io/badge/BAGEL-Hugging%20Face-orange?logo=huggingface&logoColor=yellow" alt="BAGEL on Hugging Face"/></a> <a href="https://demo.bagel-ai.org/"><img src="https://img.shields.io/badge/BAGEL-Demo-blue?logo=googleplay&logoColor=blue" alt="BAGEL Demo"/></a> <a href="https://discord.gg/Z836xxzy"><img src="https://img.shields.io/badge/BAGEL-Discord-5865F2?logo=discord&logoColor=purple" alt="BAGEL Discord"/></a> <a href="mailto:bagel@bytedance.com"><img src="https://img.shields.io/badge/BAGEL-Email-D14836?logo=gmail&logoColor=red" alt="BAGEL Email"/></a> </div>""")
UI_TRANSLATIONS = {
    "🔀 X/Y Plot": "🔀 X/Y 图表", "Enable X/Y Plot": "启用 X/Y 图表",
    "Overrides Batch Size and selected main hyperparameters.": "覆盖批量大小和选定的主要超参数。",
    "Overrides Batch Size, Task Breakdown, and selected main hyperparameters.": "覆盖批量大小、任务分解和选定的主要超参数。",
    "X-axis Parameter": "X轴参数", "X-values (comma-separated)": "X值（逗号分隔）",
    "Y-axis Parameter": "Y轴参数", "Y-values (comma-separated)": "Y值（逗号分隔）",
    "X/Y Plot Result": "X/Y 图表结果", "Generated Images / Plot": "生成图像 / 图表",
    "Result / Plot": "结果 / 图表", "(No Selection)": "(无选择)", "Prompt S/R": "提示词 搜索/替换",
    "e.g., 1.0,1.5 or Prompt S/R: find,r1,r2": "例如 1.0,1.5 或 提示词 搜索/替换: 查找内容,替换内容1,替换内容2", 
    "📝 Text to Image":"📝 文生图", "Prompt":"提示词", "Thinking":"思考模式", "Inference Hyperparameters":"推理参数", "Seed":"随机种子",
    "0 for random seed. For batch size > 1, seed will be incremented for each image.":"0为随机种子。当批量大于1时，每个图像的种子会递增。",
    "Image Ratio":"图片比例", "The longer size is fixed to 1024":"长边固定为1024", "Batch Size": "批量大小",
    "Number of images to generate sequentially.": "顺序生成的图像数量。", "CFG Text Scale":"文本CFG强度",
    "Controls how strongly the model follows the text prompt (4.0-8.0)":"控制模型是否遵循文本提示（4.0-8.0）",
    "Controls how strongly the model follows the text prompt":"控制模型是否遵循文本提示的强度", "CFG Interval":"CFG应用间隔",
    "Start of CFG application interval (end is fixed at 1.0)":"CFG应用间隔的开始（结束固定为1.0）", "CFG Renorm Type":"CFG 重归一化类型",
    "If the genrated image is blurry, use 'global'":"如果生成的图像模糊，请使用'global'", "CFG Renorm Min":"CFG 重归一化最小值",
    "1.0 disables CFG-Renorm":"1.0 禁用 CFG 重归一化", "Timesteps":"时间步数", "Total denoising steps":"总去噪步数",
    "Timestep Shift":"时间步偏移", "Higher values for layout, lower for details":"值更大更倾向于布局，值更小更倾向于调整细节",
    "Sampling":"采样", "Enable sampling for text generation":"为文本生成启用采样", "Max Think Tokens":"最大思考token数",
    "Maximum number of tokens for thinking":"思考的最大token数", "Temperature":"温度系数", "Controls randomness in text generation":"控制文本生成的随机性",
    "Thinking Process":"思考过程", "Generated Images":"生成图像", "Generate":"开始生成", "🖌️ Image Edit":"🖌️ 图像编辑",
    "Input Image":"图像输入", "Result":"结果", "CFG Image Scale":"图像CFG强度",
    "Controls how much the model preserves input image details":"控制模型保留输入图像细节的强度", "Submit":"开始生成",
    "🖼️ Image Understanding":"🖼️ 图像理解", "Controls randomness in text generation (0=deterministic, 1=creative)":"控制文本生成的随机性（0=确定，1=creative）",
    "Max New Tokens":"最大新token数", "Maximum length of generated text, including potential thinking":"生成文本的最大长度，包括可能的思考",
    "Enable Task Breakdown (Experimental)": "启用任务分解（实验性）",
    "LLM will try to break the prompt into N steps and apply them sequentially.": "LLM 将尝试把提示分解为N个步骤并按顺序应用。",
    "LLM Task Breakdown Parameters": "LLM 任务分解参数",
    "These parameters control the LLM when it breaks down your prompt.": "这些参数控制 LLM 分解您的提示时的行为。",
    "Number of Sub-steps": "子步骤数量", "How many steps the LLM should try to generate.": "LLM 应尝试生成的步骤数量。",
    "Sampling (Decomposition)": "采样（分解）", "Enable sampling for LLM step generation.": "为 LLM 步骤生成启用采样。",
    "Max Tokens (Decomposition)": "最大Token数（分解）", "Max tokens for LLM to generate steps.": "LLM 生成步骤的最大Token数。",
    "Temperature (Decomposition)": "温度系数（分解）", "Randomness for LLM step generation.": "LLM 步骤生成的随机性。",
    "Sampling (Image Edit)": "采样（图像编辑）", "Enable sampling for image edit thinking text.": "为图像编辑思考文本启用采样。",
    "Max Think Tokens (Image Edit)": "最大思考Token数（图像编辑）", "Max tokens for image edit thinking.": "图像编辑思考的最大Token数。",
    "Temperature (Image Edit)": "温度系数（图像编辑）", "Randomness for image edit thinking.": "图像编辑思考的随机性。",
    "Input Mode": "输入模式", "Single Image": "单个图像", "Batch (Files/ZIP)": "批量 (文件/ZIP)",
    "Upload Images or ZIP file": "上传图像或ZIP文件", "Result (for single image)": "结果 (单个图像)",
    "Batch Process Log": "批量处理日志", "Download Batch Results (ZIP)": "下载批量结果 (ZIP)",
    "Prompt (applies to all images in batch)": "提示 (应用于批量中的所有图像)",
}
def apply_localization(block):
    def process_component(component):
        if not component: return
        attrs_to_check = ['label', 'info', 'placeholder', 'value']
        if isinstance(component, (gr.Dropdown, gr.Radio)) and 'choices' not in attrs_to_check:
             attrs_to_check.append('choices')

        for attr in attrs_to_check:
            if hasattr(component, attr):
                current_value = getattr(component, attr)
                if attr == 'choices' and current_value is not None:
                    new_choices = [];
                    for choice_item in current_value:
                        if isinstance(choice_item, (tuple, list)) and len(choice_item) == 2: 
                            val, lab = choice_item
                            new_choices.append((val, UI_TRANSLATIONS.get(lab, lab) if isinstance(lab, str) else lab))
                        elif isinstance(choice_item, str): 
                            translated_choice = UI_TRANSLATIONS.get(choice_item, choice_item)
                            new_choices.append((translated_choice, translated_choice))
                        else: 
                            new_choices.append(choice_item)
                    setattr(component, attr, new_choices)
                elif isinstance(current_value, str) and current_value in UI_TRANSLATIONS: 
                    setattr(component, attr, UI_TRANSLATIONS[current_value])
        if hasattr(component, 'children'):
            for child in component.children: process_component(child)
    process_component(block); return block

if __name__ == "__main__":
    if args.zh: demo = apply_localization(demo)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share, inbrowser=True)
