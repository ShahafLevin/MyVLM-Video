from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Union

from vlms.videollama2 import model_init
from vlms.videollama2.mm_utils import get_model_name_from_path
from vlms.vlm_wrapper import Processor, VLMWrapper

import torch

class VideoLLaMA2Input(NamedTuple):
    pass

class VideoLLaMA2Wrapper(VLMWrapper):
    def __init__(self, device: str = 'cuda', torch_dtype: torch.dtype = torch.bfloat16):
        self.model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B'
        super().__init__(device, torch_dtype)

    def set_model(self):
        model_name = get_model_name_from_path(self.model_path)
        model, video_processor, tokenizer = model_init(model_name)
        processor = Processor(tokenizer=tokenizer, image_processor=video_processor)
        
        model = model.to(self.device, self.torch_dtype)
        
        return model, processor


    def preprocess(self, image_path: Path, prompt: str, target: str = '') -> Dict[str, Any]:
        pass

    def generate(self, inputs: Dict, concept_signals: Optional[torch.Tensor] = None) -> Union[str, List[str]]:
        pass




