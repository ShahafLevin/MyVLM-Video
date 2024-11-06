from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Union

from vlms.videollama2 import model_init, mm_infer, mm_infer_image_and_video
from vlms.videollama2.mm_utils import get_model_name_from_path
from vlms.vlm_wrapper import Processor, VLMWrapper
from vlms.videollama2.utils import disable_torch_init


import torch

class VideoLLaMA2Input(NamedTuple):
    pass

class VideoLLaMA2Wrapper(VLMWrapper):
    def __init__(self, device: str = 'cuda', torch_dtype: torch.dtype = torch.bfloat16):
        self.model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B'
        super().__init__(device, torch_dtype)

    def set_model(self):
        disable_torch_init()

        model, video_processor, tokenizer = model_init(self.model_path)
        processor = Processor(tokenizer=tokenizer, image_processor=video_processor)
        
        # model = model.to(self.device, self.torch_dtype)
        
        return model, processor


    def preprocess(self, image_path: Path, prompt: str, target: str = '') -> Dict[str, Any]:
        return {image_path: prompt}

    def generate(self, inputs: Dict, concept_signals: Optional[torch.Tensor] = None) -> Union[str, List[str]]:
        print("Generating...")
        print(inputs)
        # for video_path, prompt in inputs.items():
        video_path = "video/beer.mp4"
        image_path = "video/beer.jpg"
        prompts = [
            "The image contains my special can, named bloby", 
            "What do you see in the video?"
        ]
        return mm_infer_image_and_video(
                        image=self.processor.image_processor["image"](image_path), 
                        video=self.processor.image_processor["video"](video_path), 
                        instruct=prompts, 
                        model=self.model, 
                        tokenizer=self.processor.tokenizer, 
                        do_sample=False, 
                        modal="video"
                    )





