import sys
sys.path.append('//home/yandex/MML2024b/shahafl/MyVLM-Video/.')

import json
from dataclasses import dataclass
from pathlib import Path

import pyrallis
import torch
from tqdm import tqdm

from myvlm.common import VALID_IMAGE_EXTENSIONS, VLMType
# from vlms.blip2_wrapper import BLIP2Wrapper
# from vlms.llava_wrapper import LLaVAWrapper
# from vlms.minigpt_wrapper import MiniGPTWrapper
from vlms.videollama2_wrapper import VideoLLaMA2Wrapper 

VLM_TYPE_TO_WRAPPER = {
    # VLMType.BLIP2: BLIP2Wrapper,
    # VLMType.LLAVA: LLaVAWrapper,
    # VLMType.MINIGPT_V2: MiniGPTWrapper,
    VLMType.VIDEOLLAMA2: VideoLLaMA2Wrapper
}

VLM_TYPE_TO_PROMPT = {
    VLMType.BLIP2: '',  # empty prompt for captioning
    VLMType.LLAVA: 'Please caption this image.',
    VLMType.MINIGPT_V2: 'Please caption this image.',
    VLMType.VIDEOLLAMA2: 'What animals are in the video, what are they doing, and how does the video feel?'
}
DEVICE = 'cuda'

@dataclass
class InferenceConfig:
    # Path to the images of the concept. We will save the original captions in this directory.
    images_root: Path
    # The type of VLM to use for captioning
    vlm_type: VLMType
    # Torch dtype
    torch_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        self.image_paths = [str(p) for p in self.images_root.glob('*') if p.suffix.lower() in VALID_IMAGE_EXTENSIONS]


# @pyrallis.wrap()
def run_inference(cfg: InferenceConfig):
    """
    Generate original VLM captions on images in the specified image root. Captions will be saved as a json file in the
    same directory, mapping each image path to its caption.
    """
    print(cfg.image_paths)
    vlm_wrapper = VLM_TYPE_TO_WRAPPER[cfg.vlm_type](device=DEVICE, torch_dtype=cfg.torch_dtype)
    print("Wrapper loaded successfully")
    if (cfg.images_root / f'original_{cfg.vlm_type}_captions.json').exists():
        with open(cfg.images_root / f'original_{cfg.vlm_type}_captions.json', 'r') as f:
            path_to_original_caption = json.load(f)
    else:
        path_to_original_caption = {}
    
    for image_path in tqdm(cfg.image_paths):
        if image_path in path_to_original_caption:
            continue
        print(f"Image path: {image_path}")
        # TODO: Continue from here....
        inputs = vlm_wrapper.preprocess(image_path, prompt=VLM_TYPE_TO_PROMPT[cfg.vlm_type])
        caption = vlm_wrapper.generate(inputs, concept_signals=None)
        print(f"Caption: {caption}")
        path_to_original_caption[image_path] = caption

    with open(cfg.images_root / f'original_{cfg.vlm_type}_captions.json', 'w') as f:
        json.dump(path_to_original_caption, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    inference_config = InferenceConfig(images_root=Path("./video"),
                                       vlm_type=VLMType.VIDEOLLAMA2)
    run_inference(inference_config)
