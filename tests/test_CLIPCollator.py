# In tests/test_CLIPCollator.py
from unml.data import CLIPCollator
from transformers import CLIPImageProcessor

image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_collator_obj = CLIPCollator(image_processor=image_processor)
print(int(image_processor.size.get("shortest_edge")))
print(image_processor.crop_size)
mean = list(image_processor.image_mean)
std = list(image_processor.image_std)
print(mean)        
print(std)        