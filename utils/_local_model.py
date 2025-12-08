# -*- coding:utf-8 -*-
import os
import base64
from io import BytesIO
from copy import deepcopy
import torch
os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True"



# call_qwen2_vl = None

from qwen_vl_utils import process_vision_info
# Model Needed: Qwen/Qwen2-VL-7B-Instruct, Qwen/Qwen2-VL-72B-Instruct
def call_qwen2_vl(client, image, text_prompt, temperature:float=0.0):
    """
    ## 7B
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    qwen2_vl = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto")  # Load the model in half-precision on the available device(s)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    client = [processor, qwen2_vl]
    model = 'qwen2-vl-7b-instruct'

    ## 72B
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    qwen2_vl = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-72B-Instruct", torch_dtype="auto", device_map="auto")  # Load the model in half-precision on the available device(s)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")
    client = [processor, qwen2_vl]
    model = 'qwen2-vl-72b-instruct'
    """
    processor, model = client
    # Only Text
    if image is None:
        # Conversations containing only text query
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
    # Single Image
    elif type(image) != list:
        # PIL to base64
        img_buffer = BytesIO()
        image.save(img_buffer, format='PNG')
        byte_data = img_buffer.getvalue()
        image_base64 = base64.b64encode(byte_data).decode('utf-8')
        # Conversations containing single image and a text query
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f'data:image/png;base64,{image_base64}'},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
    # Multiple Images
    else:
        images = deepcopy(image)
        conversations = [{'role': 'user', 'content': []}]
        for image in images:
            # PIL to base64
            img_buffer = BytesIO()
            image.save(img_buffer, format='PNG')
            byte_data = img_buffer.getvalue()
            image_base64 = base64.b64encode(byte_data).decode('utf-8')
            # Conversations containing multiple images and a text query
            conversations[0]['content'].append({'type': 'image', 'image': f'data:image/png;base64,{image_base64}'})
        conversations[0]['content'].append({'type': 'text', 'text': text_prompt})
    # Preparation for inference
    text = processor.apply_chat_template(conversations , tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversations)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    # Inference
    if temperature > 0:
        generated_ids = model.generate(**inputs, max_new_tokens=10000, do_sample=True, temperature=temperature)
    else:
        generated_ids = model.generate(**inputs, max_new_tokens=10000, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output



# call_intern_vl2 = None

from .intern_vl_module import load_image
# Note: The function load_image has been modified to accept both image path and direct image.
# Model Needed: OpenGVLab/InternVL2-8B, OpenGVLab/InternVL2-26B, OpenGVLab/InternVL2-40B
def call_intern_vl2(client, image, text_prompt, temperature:float=0.0):
    """
    ### InternVL-2
    ## 8B
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL2-8B', trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained('OpenGVLab/InternVL2-8B', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True).eval().cuda()
    client = [tokenizer, model]
    model = 'intern-vl2-8b'

    ## 26B
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL2-26B', trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained('OpenGVLab/InternVL2-26B', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True).eval().cuda()
    client = [tokenizer, model]
    model = 'intern-vl2-26b'

    ## 40B
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL2-40B', trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained('OpenGVLab/InternVL2-40B', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True).eval().cuda()
    client = [tokenizer, model]
    model = 'intern-vl2-40b'


    ### InternVL-2.5
    ## 8B
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL2_5-8B', trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained('OpenGVLab/InternVL2_5-8B', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).eval().cuda()
    client = [tokenizer, model]
    model = 'intern-vl2_5-8b'

    ## 26B
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL2_5-26B', trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained('OpenGVLab/InternVL2_5-26B', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,use_flash_attn=True, trust_remote_code=True).eval().cuda()
    client = [tokenizer, model]
    model = 'intern-vl2_5-26b'

    ## 38B
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL2_5-38B', trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained('OpenGVLab/InternVL2_5-38B', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,use_flash_attn=True, trust_remote_code=True).eval().cuda()
    client = [tokenizer, model]
    model = 'intern-vl2_5-38b'

    ## 78B
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL2_5-78B', trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained('OpenGVLab/InternVL2_5-78B', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,use_flash_attn=True, trust_remote_code=True).eval().cuda()
    client = [tokenizer, model]
    model = 'intern-vl2_5-78b'
    """
    tokenizer, model = client
    if temperature > 0:
        generation_config = dict(max_new_tokens=10000, num_beams=1, do_sample=True, temperature=temperature)
    else:
        generation_config = dict(max_new_tokens=10000, num_beams=1, do_sample=False)
    DEFAULT_IMAGE_TOKEN = '<image>'
    # Only Text
    if image is None:
        question = f"{text_prompt}"
        # Inference
        response = model.chat(tokenizer, None, question, generation_config)
    # Single Image
    elif type(image) != list:
        # Process the image
        pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()
        question = f"{DEFAULT_IMAGE_TOKEN}\n{text_prompt}"
        # Inference
        response = model.chat(tokenizer, pixel_values, question, generation_config)
    # Multiple Images
    else:
        images = deepcopy(image)
        # Process the image
        pixel_values = []
        num_patches_list = []
        for image in images:
            pixels = load_image(image, max_num=12).to(torch.bfloat16).cuda()
            pixel_values.append(pixels)
            num_patches_list.append(pixels.size(0))
        pixel_values = torch.cat(pixel_values, dim=0)
        # Prompt
        prefix_cn = ['第一张图', '第二张图', '第三张图', '第四张图', '第五张图',
                     '第六张图', '第七张图', '第八张图', '第九张图', '第十张图']
        prefix_en = ['The First Image', 'The Second Image', 'The Third Image','The Fourth Image', 'The Fifth Image',
                     'The Sixth Image', 'The Seventh Image', 'The Eighth Image', 'The Ninth Image', 'The Tenth Image']
        prefix = deepcopy(prefix_en) if ('image' in text_prompt.lower() or 'pic' in text_prompt.lower()) else deepcopy(prefix_cn)
        image_tokens = [f'{prefix[n]}: {token}' for n, token in
                        zip(list(range(len(images))),[DEFAULT_IMAGE_TOKEN]*len(images))]
        question = '\n'.join(image_tokens)+f"\n\n{text_prompt}"
        # Inference
        response= model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list)
    output = deepcopy(response)
    return output



# call_llava_onevision = None
'''
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
# Model Needed: lmms-lab/llava-onevision-qwen2-7b-ov, meta-llama/Meta-Llama-3-8B-Instruct, google/siglip-so400m-patch14-384
def call_llava_onevision(client, image, text_prompt, temperature:float=0.0):
    """
    ## 7B
    pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
    from llava.model.builder import load_pretrained_model
    tokenizer, model, image_processor, max_length = load_pretrained_model(model_path="lmms-lab/llava-onevision-qwen2-7b-ov", model_base=None, model_name="llava_qwen", device_map="auto")
    model.eval()
    client = [tokenizer, model, image_processor, max_length]
    model = 'llava-onevision-qwen2-7b-ov'

    ## 72B
    pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
    from llava.model.builder import load_pretrained_model
    tokenizer, model, image_processor, max_length = load_pretrained_model(model_path="lmms-lab/llava-onevision-qwen2-72b-ov-chat", model_base=None, model_name="llava_qwen", device_map="auto")
    model.eval()
    client = [tokenizer, model, image_processor, max_length]
    model = 'llava-onevision-qwen2-72b-ov-chat'
    """
    tokenizer, model, image_processor, max_length = client
    device = "cuda"
    # Only Text
    if image is None:
        # Process the image
        image_tensor = None
        image_sizes = None
        # Chat Prompt
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        question = f"{text_prompt}"
    # Single Image
    elif type(image) != list:
        # Process the image
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
        image_sizes = [image.size]
        # Chat Prompt
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        question = f"{DEFAULT_IMAGE_TOKEN}\n{text_prompt}"
    # Multiple Images
    else:
        images = deepcopy(image)
        # Process the image
        image_tensor = process_images(images, image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
        image_sizes = [image.size for image in images]
        # Chat Prompt
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        prefix_cn = ['第一张图', '第二张图', '第三张图', '第四张图', '第五张图',
                     '第六张图', '第七张图', '第八张图', '第九张图', '第十张图']
        prefix_en = ['The First Image', 'The Second Image', 'The Third Image','The Fourth Image', 'The Fifth Image',
                     'The Sixth Image', 'The Seventh Image', 'The Eighth Image', 'The Ninth Image', 'The Tenth Image']
        prefix = deepcopy(prefix_en) if ('image' in text_prompt or 'pic' in text_prompt) else deepcopy(prefix_cn)
        image_tokens = [f'{prefix[n]}: {token}' for n, token in
                        zip(list(range(len(images))),[DEFAULT_IMAGE_TOKEN]*len(images))]
        question = '\n'.join(image_tokens)+f"\n\n{text_prompt}"
    conv = deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    if temperature > 0:
        generation_config = dict(max_new_tokens=10000, do_sample=True, temperature=temperature)
    else:
        generation_config = dict(max_new_tokens=10000,  do_sample=False)
    # Invoke Llava-OneVision
    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        **generation_config
    )
    output = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    return output
'''


# call_minicpm = None

# Model Needed: openbmb/MiniCPM-V-2_6
def call_minicpm(client, image, text_prompt, temperature:float=0.0):
    """
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)
    model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
        attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()
    client = [tokenizer, model]
    model = 'minicpm-v-2_6'
    """
    tokenizer, model = client
    # Only Text
    if image is None:
        messages = [{'role': 'user', 'content': [text_prompt]}]
    # Single Image
    elif type(image) != list:
        messages = [{'role': 'user', 'content': [image, text_prompt]}]
    # Multiple Image
    else:
        images = deepcopy(image)
        messages = [{'role': 'user', 'content': [*images, text_prompt]}]
    if temperature > 0:
        generation_config = dict(max_new_tokens=10000, do_sample=True, temperature=temperature)
    else:
        generation_config = dict(max_new_tokens=10000,  do_sample=False)
    # Invoke MiniCPM-V-2_6
    answer = model.chat(
        image=None,
        msgs=messages,
        tokenizer=tokenizer,
        **generation_config
    )
    return answer



call_mantis_siglip = None

# from mantis.models.mllava import chat_mllava
# # Model Needed: TIGER-Lab/Mantis-8B-siglip-llama3
# def call_mantis_siglip(client, image, text_prompt, temperature:float=0.0):
#     """
#     !pip install git+https://github.com/TIGER-AI-Lab/Mantis.git
#     from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration
#     processor = MLlavaProcessor.from_pretrained("TIGER-Lab/Mantis-8B-siglip-llama3")
#     model = LlavaForConditionalGeneration.from_pretrained("TIGER-Lab/Mantis-8B-siglip-llama3", device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
#     client = [processor, model]
#     model = processor'mantis-8b-siglip-llama3'
#     """
#     processor, mantis = client
#     DEFAULT_IMAGE_TOKEN = '<image>'
#     # Only Text
#     if image is None:
#         images = image
#         conversation = f"{text_prompt}"
#     # Single Image
#     elif type(image) != list:
#         images = [image]
#         conversation = f"{DEFAULT_IMAGE_TOKEN}\n{text_prompt}"
#     # Multiple Image
#     else:
#         images = deepcopy(image)
#         prefix_cn = ['第一张图', '第二张图', '第三张图', '第四张图', '第五张图',
#                      '第六张图', '第七张图', '第八张图', '第九张图', '第十张图']
#         prefix_en = ['The First Image', 'The Second Image', 'The Third Image','The Fourth Image', 'The Fifth Image',
#                      'The Sixth Image', 'The Seventh Image', 'The Eighth Image', 'The Ninth Image', 'The Tenth Image']
#         prefix = deepcopy(prefix_en) if ('image' in text_prompt or 'pic' in text_prompt) else deepcopy(prefix_cn)
#         image_tokens = [f'{prefix[n]}: {token}' for n, token in
#                         zip(list(range(len(images))),[DEFAULT_IMAGE_TOKEN]*len(images))]
#         conversation = '\n'.join(image_tokens)+f"\n\n{text_prompt}"
#     # Invoke Mantis-8B-siglip-llama3
#     if temperature > 0:
#         generation_kwargs = {"max_new_tokens": 10000, "num_beams": 1, "do_sample": True, "temperature": temperature}
#     else:
#         generation_kwargs = {"max_new_tokens": 10000, "num_beams": 1, "do_sample": False}
#     response, history = chat_mllava(conversation, images, mantis, processor, **generation_kwargs)
#     output = deepcopy(response)
#     return output



call_idefics = None

# # Model Needed: HuggingFaceM4/Idefics3-8B-Llama3
# def call_idefics(client, image, text_prompt, temperature:float=0.0):
#     """
#     from transformers import AutoProcessor, AutoModelForVision2Seq
#     processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3", size= {"longest_edge": 2*364})
#     mantis = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3", torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2").eval().cuda()
#     client = [processor, mantis]
#     model = 'idefics3-8b-llama3'
#     """
#     processor, model = client
#     # Only Text
#     if image is None:
#         images = image
#         conversation = \
#             [
#                 {"role": "user", "content": [
#                     {"type": "text", "text": text_prompt},
#                 ]}
#             ]
#     # Single Image
#     elif type(image) != list:
#         images = [image]
#         conversation = \
#             [
#                 {"role": "user", "content": [
#                     {"type": "image"},
#                     {"type": "text", "text": text_prompt},
#                 ]}
#             ]
#     # Multiple Image
#     else:
#         images = deepcopy(image)
#         conversation = [{'role':'user', 'content':[]}]
#         for _ in images:
#             conversation[0]['content'].append({"type": "image"})
#         conversation[0]['content'].append({"type": "text", "text": text_prompt})
#     # Tokenization
#     prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
#     inputs = processor(text=prompt, images=images, return_tensors="pt")
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
#     # Invoke Idefics3-8B-Llama3
#     if temperature > 0:
#         generation_kwargs = {"max_new_tokens": 10000, "num_beams": 1, "do_sample": True, "temperature": temperature}
#     else:
#         generation_kwargs = {"max_new_tokens": 10000, "num_beams": 1, "do_sample": False}
#     response = model.generate(**inputs, **generation_kwargs)
#     output = processor.batch_decode(response[:, inputs["input_ids"].shape[1]:],
#                                     skip_special_tokens=True)[0]
#     return output

