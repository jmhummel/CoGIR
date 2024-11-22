import argparse
import time
from argparse import BooleanOptionalAction
from pathlib import Path

import albumentations
import cv2
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
import numpy as np

from data.dataset import get_image_paths
from data.augs import CenterSquareCrop

TEMPLATE = """An image is provided.

Describe this image and its style to generate a detailed and visually rich description suitable for AI-based image generation. Focus on the key elements in the image, including subjects, environment, mood, lighting, colors, textures, composition, and artistic style. Use precise and vivid language to create a mental picture of the scene, avoiding vague or overly generic terms. Avoid describing technical aspects like camera settings, file metadata, or the AI itself. The description should balance conciseness and detail, ideally no more than three sentences. Additionally, exclude undesired elements if necessary to refine the generation process.

### **Tips for Writing High-Quality Descriptions**:
1. **Be Concise**:
   - Avoid overly lengthy sentences or repeating ideas.  
     Example: Instead of:  
     *"The overall mood of the image is one of tranquility and natural beauty, with the plants and flowers providing a sense of serenity and peace,"*  
     use:  
     *"The mood is tranquil, with vibrant plants exuding natural beauty and peace."*

2. **Focus on Unique Details**:
   - Highlight what makes the image distinct without overloading unnecessary details.  
     Example: Instead of:  
     *"The plants are arranged in a way that creates a sense of abundance and diversity, with some plants spilling over the edges of their containers,"*  
     use:  
     *"The plants overflow their containers, creating a sense of abundance."*

3. **Maintain Consistent Tone**:
   - Keep the level of detail and sentence structure consistent throughout. Avoid abrupt shifts in verbosity.  

4. **Avoid Generic Phrases**:
   - Replace vague terms like *"the action of the hand"* with vivid and specific imagery.  
     Example: Instead of:  
     *"The focus is on the action of the hand,"*  
     use:  
     *"The hand carefully places a piece of paper into a soil-filled pot."*

5. **Clarify Artistic Style**:
   - Use specific terms like *"realistic with vibrant lighting"* or *"painterly with diffused tones."* Avoid overly generic descriptors such as *"nice" or "beautiful."*

6. **Balance All Elements**:
   - Ensure you address subjects, environment, mood, lighting, and textures proportionally. Do not overemphasize a single aspect at the expense of others.

### **Examples of Effective Descriptions**:
1. A rugged rock surface with a deep blue hue and patches of moss dominates the foreground, while frothy waves crash against the shore in the blurred background. The contrast between the sharp details of the rock and the soft, dynamic water creates depth and movement. Realistic with a focus on texture and natural tones.

2. A tropical beach with white sands curving around a turquoise lagoon, bordered by dense green palm trees. The aerial perspective highlights the smooth transition from shallow to deeper waters, with coral reefs visible beneath the surface. The mood is tranquil and idyllic, with vibrant colors emphasizing the beach's pristine beauty.

3. A lone figure stands on a windswept hill beneath a dramatic sky filled with churning gray clouds. A single ray of light breaks through to illuminate the figure, creating a stark contrast with the stormy backdrop. The scene is cinematic, focusing on the interplay of light and shadow.

4. A serene lake surrounded by snow-capped mountains, with a hiker gazing at the tranquil water. The sky is a soft gradient of blue with scattered clouds, reflecting on the lake's surface. The scene is naturalistic and peaceful, emphasizing the harmony between the hiker and the majestic landscape.

5. An underwater coral reef teeming with life, showcasing branching and rounded corals in hues of orange, white, and brown. A small fish swims among the corals, while sunlight filters through clear blue water, creating dappled patterns of light. The style is realistic, with enhanced textures emphasizing the reef's vibrant biodiversity."""



class CaptionGenerationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, image_size=(512, 512)):
        image_dir = Path(image_dir)
        self.image_dir = image_dir
        self.image_size = image_size
        self.image_paths = get_image_paths(image_dir)
        self.crop = CenterSquareCrop(always_apply=True, p=1.0)
        self.resize = albumentations.Resize(*image_size, interpolation=cv2.INTER_CUBIC)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.crop(image=image)["image"]
        image = self.resize(image=image)["image"]
        image = image.clip(0, 255).astype(np.uint8)
        return image, image_path

def main(
        image_dir,
        model_id="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        # temperature=0.7,
        quantization=None,
        use_flash_attn=True,
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        timeout=300,
        verbose=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        transformers.logging.set_verbosity_warning()
    else:
        transformers.logging.set_verbosity_error()

    model_kwargs = {}


    if use_flash_attn:
        model_kwargs["use_flash_attention_2"] = True

    if quantization == 4:
        model_kwargs["load_in_4bit"] = True

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        **model_kwargs,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    dataset = CaptionGenerationDataset(image_dir=image_dir, image_size=(384, 384))
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                             pin_memory=pin_memory, persistent_workers=persistent_workers,
                             timeout=timeout, collate_fn=lambda x: list(zip(*x)))

    results = []

    start_time = time.time()

    def create_message():
        return [
            {"role": "user", "content": [
                {"type": "text", "text": TEMPLATE},
                {"type": "image"},

            ]}
        ]

    # batch_idx = 0
    for images, paths in tqdm(data_loader, unit='img', unit_scale=batch_size):
        list_of_messages = [create_message() for _ in images]
        prompts = [processor.apply_chat_template(messages, add_generation_prompt=True) for messages in list_of_messages]
        inputs = processor(
            images=images,
            text=prompts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        ).to(model.device, torch.float16)

        output = model.generate(**inputs, max_new_tokens=256,
                                pad_token_id=processor.tokenizer.pad_token_id,
                                eos_token_id=processor.tokenizer.eos_token_id)
        output = output[:, inputs["input_ids"].shape[1]:]
        refined_texts = processor.batch_decode(output, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=True)

        for path, refined_text in zip(paths, refined_texts):
            results.append({'path': path, 'desc': refined_text})

        # Create dataframe from results
        df = pd.DataFrame(results)

        # Save results to CSV
        df.to_csv('results.csv', index=False)
        # print(f'Saved results to results.csv')



    total_time = time.time() - start_time
    print(f'total_time: {total_time}')


    # # Create dataframe from results
    # results = pd.DataFrame(results)
    #
    # # Save results to CSV
    # results.to_csv('results.csv', index=False)
    # print(f'Saved results to results.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test captioning throughput with random images")
    parser.add_argument("image_dir", type=str, help="Directory containing images to caption")
    parser.add_argument("--model", type=str, default="llava-hf/llava-onevision-qwen2-7b-ov-hf", help="Model to use for captioning")
    # parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling from the model")
    parser.add_argument("--quantization", type=int, default=None, help="Quantization level for the model (choices: 4, 8, default: None)")
    parser.add_argument("--flash-attn", default=True, action=BooleanOptionalAction, help="Use Flash Attention for the model")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for the DataLoader")
    parser.add_argument("--pin-memory", default=True, action=BooleanOptionalAction, help="Use pinned memory for DataLoader")
    parser.add_argument("--persistent-workers", default=True, action=BooleanOptionalAction, help="Use persistent workers for DataLoader")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout for DataLoader workers")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output")

    args = parser.parse_args()

    main(
        image_dir=args.image_dir,
        model_id=args.model,
        # temperature=args.temperature,
        quantization=args.quantization,
        use_flash_attn=args.flash_attn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        timeout=args.timeout,
        verbose=args.verbose)
