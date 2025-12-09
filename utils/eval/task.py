# -*- coding:utf-8 -*-
import os
import time
import random
import textwrap
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
import opencc
converter = opencc.OpenCC('t2s')  # Traditional Chinese to Simplified Chinese
import warnings
warnings.filterwarnings("ignore")

from ..image_process import random_4panels, get_4panel_image, get_4image_list
from ..pinyin_augmentation import *
from ..json_process import *
from .._api_model import *
from .._local_model import *


def obtain_client(model):
    if model == 'gpt-4o-2024-08-06':
        # GPT-4o
        from openai import OpenAI
        client = OpenAI()
    elif model == 'gpt-4o-mini-2024-07-18':
        # GPT-4o-mini
        from openai import OpenAI
        client = OpenAI()
    elif model == 'claude-3-5-sonnet-20241022':
        # Claude-3.5-Sonnet
        from anthropic import Anthropic
        client = Anthropic()
    elif model == 'qwen2.5-vl-72b-instruct':
        # Qwen2.5-VL-72B-Instruct
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        qwen25_vl = Qwen2_5_VLForConditionalGeneration.from_pretrained("model/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
        client = [processor, qwen25_vl]
    elif model == 'qwen2.5-vl-7b-instruct':
        # Qwen2.5-VL-7B-Instruct
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        qwen25_vl = Qwen2_5_VLForConditionalGeneration.from_pretrained("/root/autodl-tmp/model/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained("/root/autodl-tmp/model/Qwen2.5-VL-7B-Instruct")
        client = [processor, qwen25_vl]
    elif model == 'intern-vl2_5-78b':
        # InternVL2_5-78B
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL2_5-78B', trust_remote_code=True, use_fast=False)
        internvl = AutoModel.from_pretrained('OpenGVLab/InternVL2_5-78B', torch_dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True, use_flash_attn=True,
                                             trust_remote_code=True).eval().cuda()
        client = [tokenizer, internvl]
    elif model == 'intern-vl2_5-38b':
        # InternVL2_5-38B
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL2_5-38B', trust_remote_code=True, use_fast=False)
        internvl = AutoModel.from_pretrained('OpenGVLab/InternVL2_5-38B', torch_dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True, use_flash_attn=True,
                                             trust_remote_code=True).eval().cuda()
        client = [tokenizer, internvl]
    elif model == 'intern-vl2_5-26b':
        # InternVL2_5-26B
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL2_5-26B', trust_remote_code=True, use_fast=False)
        internvl = AutoModel.from_pretrained('OpenGVLab/InternVL2_5-26B', torch_dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True, use_flash_attn=True,
                                             trust_remote_code=True).eval().cuda()
        client = [tokenizer, internvl]
    elif model == 'intern-vl2_5-8b':
        # InternVL2_5-8B
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL2_5-8B', trust_remote_code=True, use_fast=False)
        internvl = AutoModel.from_pretrained('OpenGVLab/InternVL2_5-8B', torch_dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True, use_flash_attn=True,
                                             trust_remote_code=True).eval().cuda()
        client = [tokenizer, internvl]
    elif model == 'llava-onevision-qwen2-72b-ov-chat':
        # Llava-onevision-qwen2-72b-ov-chat
        # pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
        from llava.model.builder import load_pretrained_model
        tokenizer, llava_onevision, image_processor, max_length = load_pretrained_model(
            model_path="lmms-lab/llava-onevision-qwen2-72b-ov-chat", model_base=None, model_name="llava_qwen",
            device_map="auto")  # if no flash_attn, use attn_implementation="sdpa"
        llava_onevision.eval()
        client = [tokenizer, llava_onevision, image_processor, max_length]
    elif model == 'llava-onevision-qwen2-7b-ov':
        # Llava-onevision-qwen2-7b-ov
        # pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
        from llava.model.builder import load_pretrained_model
        tokenizer, llava_onevision, image_processor, max_length = load_pretrained_model(
            model_path="lmms-lab/llava-onevision-qwen2-7b-ov", model_base=None, model_name="llava_qwen",
            device_map="auto")  # if no flash_attn, use attn_implementation="sdpa"
        llava_onevision.eval()
        client = [tokenizer, llava_onevision, image_processor, max_length]
    elif model == 'minicpm-v-2_6':
        # MiniCPM-V-2_6
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)
        minicpm = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
                                            attn_implementation='flash_attention_2',
                                            torch_dtype=torch.bfloat16)  # sdpa or flash_attention_2, no eager
        minicpm.eval().cuda()
        client = [tokenizer, minicpm]
    elif model == 'mantis-8b-siglip-llama3':
        # Mantis-8B-siglip-llama3
        # pip install git+https://github.com/TIGER-AI-Lab/Mantis.git
        from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration
        processor = MLlavaProcessor.from_pretrained("TIGER-Lab/Mantis-8B-siglip-llama3")
        mantis_8B = LlavaForConditionalGeneration.from_pretrained("TIGER-Lab/Mantis-8B-siglip-llama3",
                                                                  device_map="cuda", torch_dtype=torch.bfloat16,
                                                                  attn_implementation="flash_attention_2")
        client = [processor, mantis_8B]
    else:
        # Idefics3-8B-Llama3
        from transformers import AutoProcessor, AutoModelForVision2Seq
        processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3",
                                                  size={"longest_edge": 2 * 364})
        idefics3 = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3",
                                                          torch_dtype=torch.bfloat16, device_map="auto",
                                                          attn_implementation="flash_attention_2").eval().cuda()
        client = [processor, idefics3]
    return client


def vlm_pun_meme_detection(dataset_path:str, model:str, client, language:str, prompt_mode:int, pilot:bool=True, temperature:float=0.0, save:bool=False):
    """
    Test VLMs' performance on pun meme detection (pun/non-pun)
    Consider seven prompt modes:
    - 1: Only image, zero-shot
    - 2: Image & caption, zero-shot
    - 3: Only image, few-shot
    - 4: Image & caption, few-shot
    - 5: Image & caption (with Chinese Pinyin), few-shot
    - 6: Image & caption, few-shot + CoT
    - 7: Image & caption (with Chinese Pinyin), few-shot + CoT
    """
    call = {'gpt-4o-2024-08-06': call_gpt4o,
            'gpt-4o-mini-2024-07-18': call_gpt4o_mini,
            'claude-3-5-sonnet-20241022': call_claude35sonnet,
            'qwen2.5-vl-72b-instruct': call_qwen2_vl,
            'qwen2.5-vl-7b-instruct': call_qwen2_vl,
            'intern-vl2_5-78b': call_intern_vl2,
            'intern-vl2_5-38b': call_intern_vl2,
            'intern-vl2_5-26b': call_intern_vl2,
            'intern-vl2_5-8b': call_intern_vl2,
#            'llava-onevision-qwen2-72b-ov-chat': call_llava_onevision,
#            'llava-onevision-qwen2-7b-ov': call_llava_onevision,
            'minicpm-v-2_6': call_minicpm}
    mode = [None, 'only_image(zero_shot)', 'image_&_caption(zero_shot)',
            'only_image(few_shot)', 'image_&_caption(few_shot)', 'image_&_caption_&_pinyin(few_shot)',
            'image_&_caption(few_shot+CoT)', 'image_&_caption_&_pinyin(few_shot+CoT)']
    assert language in ['en', 'cn']
    assert model in list(call.keys())
    assert 0 < prompt_mode < len(mode)

    # Dataset Meta
    meta_path = os.path.join(dataset_path, '_Meta.json')
    meta = load_json_file(meta_path)
    example_id = 'CN00601'  # Exclude example from the evaluation
    if pilot:
        dataset = [img for img in meta if example_id not in img and int(img.split('.')[0][2:])<=100]  # Demo
    else:
        dataset = [img for img in meta if example_id not in img]  # Whole dataset
    # Examples
    examples = [[], []]
    for img in ['CN00601.png', 'CN00601.similar_image.png', 'CN00601.similar_text.png']:
        image = Image.open(os.path.join(dataset_path, img))
        caption = meta[img]['caption_1']
        examples[0].append(image); examples[1].append(caption)
    # Save Path
    save_path = './results/pun_meme_eval/pun_meme_detection.json'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    record = extend_dict(load_json_file(save_path), [model, language, mode[prompt_mode]])
    if record[model][language][mode[prompt_mode]] == dict():
        record[model][language][mode[prompt_mode]] = {img:{'model_output':None,'model_judgment':None} for img in dataset}

    # Task Prompt
    py = lambda text: ' '.join(chinese_character_to_pinyin(text, pattern='normal'))
    if language == 'cn':
        pun_meme_definition = textwrap.dedent("""\
        定义：\n对于由图像和字幕组成的表情包，如果表情包字幕利用一词多义或谐音多义的方式有意使表情包具有两层或多层含义，且至少其中一层含义基于表情包的画面元素，则称该表情包为“双关表情包”，否则称为“非双关表情包”。
        """)
        # 1. Only image, zero-shot
        if prompt_mode == 1:
            instruction = textwrap.dedent("""\
            任务描述：\n请你根据以上定义，判断给定的表情包图片是否为双关表情包。你只需回答“双关”或“非双关”，不要输出其他内容。\n\n你的回答：
            """)
        # 2. Image & caption, zero-shot
        elif prompt_mode == 2:
            instruction = lambda caption: textwrap.dedent(f"""\
            任务描述：\n请你根据以上定义，判断给定的表情包图片是否为双关表情包。该表情包内的字幕为“{caption}”。你只需回答“双关”或“非双关”，不要输出其他内容。\n\n你的回答：
            """)
        # 3. Only image, few-shot
        elif prompt_mode == 3:
            instruction = textwrap.dedent("""\
            示例和任务描述：\n现将提供的四张表情包图片按顺序分别记为第一、二、三、四张表情包。根据以上定义可知：
            第一张表情包是双关表情包；
            第二张表情包是非双关表情包；
            第三张表情包也是非双关表情包。
            请你根据定义和前三张表情包的类别，判断第四张表情包是否为双关表情包。你只需回答“双关”或“非双关”，不要输出其他内容。\n\n你的回答：
            """)
        # 4. Image & caption, few-shot
        elif prompt_mode == 4:
            instruction = lambda caption1, caption2, caption3, caption4: textwrap.dedent(f"""\
            示例和任务描述：\n现将提供的四张表情包图片按顺序分别记为第一、二、三、四张表情包。根据以上定义可知：
            第一张表情包内的字幕为“{caption1}”，是双关表情包；
            第二张表情包内的字幕为“{caption2}”，是非双关表情包；
            第三张表情包内的字幕为“{caption3}”，也是非双关表情包。
            请你根据定义和前三张表情包的类别，判断第四张表情包是否为双关表情包。第四张表情包内的字幕为“{caption4}”。你只需回答“双关”或“非双关”，不要输出其他内容。\n\n你的回答：
            """)
        # 5. Image & caption (with Chinese Pinyin), few-shot
        elif prompt_mode == 5:
            instruction = lambda caption1, caption2, caption3, caption4: textwrap.dedent(f"""\
            示例和任务描述：\n现将提供的四张表情包图片按顺序分别记为第一、二、三、四张表情包。根据以上定义可知：
            第一张表情包内的字幕为“{caption1}”，字幕的中文拼音为“{py(caption1)}”，是双关表情包；
            第二张表情包内的字幕为“{caption2}”，字幕的中文拼音为“{py(caption2)}”，是非双关表情包；
            第三张表情包内的字幕为“{caption3}”，字幕的中文拼音为“{py(caption3)}”，也是非双关表情包。
            请你根据定义和前三张表情包的类别，判断第四张表情包是否为双关表情包。第四张表情包内的字幕为“{caption4}”，字幕的中文拼音为“{py(caption4)}”。你只需回答“双关”或“非双关”，不要输出其他内容。\n\n你的回答：
            """)
        # 6. Image & caption, few-shot + CoT
        elif prompt_mode == 6:
            instruction = lambda caption1, caption2, caption3, caption4: textwrap.dedent(f"""\
            示例和任务描述：\n现将提供的四张表情包图片按顺序分别记为第一、二、三、四张表情包。根据以上定义可知：
            表情包分析：第一张表情包的画面中央是一位女性，左右两边各有一位男性，他们的脸被遮住，并标有“男”字，字幕是“{caption1}”，谐音“左右为难”。该表情包表达了一种幽默感，意指画面中女性被两位男性夹在中间不知所措的情景，更是泛指某人在两难的情况下左右为难，无法做出选择。\n表情包判断：双关
            表情包分析：第二张表情包的画面内容展示了一男一女在一起行走的场景，画面下方有字幕“{caption2}”，不谐音其他文字。这个表情包的具体含义是形容一个人偶尔会去查看或关注某个群体或群聊的动态，有点像是负责人或管理员在进行监督或检查的行为，通常带有一些调侃的意味。\n表情包判断：非双关
            表情包分析：第三张表情包中是一只猫被两只手从左右两侧轻轻拉住，看上去表情有些无奈，字幕写着“{caption3}”，不谐音其他文字。这张表情包的含义是形容人在面对选择时感到困惑和不知所措，无法决定站在哪一边。\n表情包判断:非双关
            请你根据定义和前三张表情包的类别，判断第四张表情包是否为双关表情包。第四张表情包内的字幕为“{caption4}”。要求严格按照上面的输出格式，先对第四张表情包进行分析，再判断其“双关”或“非双关”，不要输出其他内容。\n\n你的回答：
            """)
        # 7. Image & caption (with Chinese Pinyin), few-shot + CoT
        else:
            instruction = lambda caption1, caption2, caption3, caption4: textwrap.dedent(f"""\
            示例和任务描述：\n现将提供的四张表情包图片按顺序分别记为第一、二、三、四张表情包。根据以上定义可知：
            表情包分析：第一张表情包的画面中央是一位女性，左右两边各有一位男性，他们的脸被遮住，并标有“男”字，字幕是“{caption1}”。字幕的中文拼音为“{py(caption1)}”，谐音“左右为难”。该表情包表达了一种幽默感，意指画面中女性被两位男性夹在中间不知所措的情景，更是泛指某人在两难的情况下左右为难，无法做出选择。\n表情包判断：双关
            表情包分析：第二张表情包的画面内容展示了一男一女在一起行走的场景，画面下方有字幕“{caption2}”。字幕的中文拼音为“{py(caption2)}”，不谐音其他文字。这个表情包的具体含义是形容一个人偶尔会去查看或关注某个群体或群聊的动态，有点像是负责人或管理员在进行监督或检查的行为，通常带有一些调侃的意味。\n表情包判断：非双关
            表情包分析：第三张表情包中是一只猫被两只手从左右两侧轻轻拉住，看上去表情有些无奈，字幕写着“{caption3}”。字幕的中文拼音为“{py(caption3)}”，不谐音其他文字。这张表情包的含义是形容人在面对选择时感到困惑和不知所措，无法决定站在哪一边。\n表情包判断:非双关
            请你根据定义和前三张表情包的类别，判断第四张表情包是否为双关表情包。第四张表情包内的字幕为“{caption4}”，字幕的中文拼音为“{py(caption4)}”。要求严格按照上面的输出格式，先对第四张表情包进行分析，再判断其“双关”或“非双关”，不要输出其他内容。\n\n你的回答：
            """)
    else:
        pun_meme_definition = textwrap.dedent("""\
        Definition:\nFor memes consisting of images and captions, if the caption intentionally uses polysemy or homophony to create two or more meanings, with at least one meaning based on the visual elements of the image, the meme is called a "pun meme". Otherwise, it is called a "non-pun meme".
        """)
        # 1. Only image, zero-shot
        if prompt_mode == 1:
            instruction = textwrap.dedent("""\
            Task Description:\nBased on the above definition, determine whether the given meme is a pun meme or not. You only need to respond with "pun" or "non-pun". Do not output any other content.\n\nYour Response:
            """)
        # 2. Image & caption, zero-shot
        elif prompt_mode == 2:
            instruction = lambda caption: textwrap.dedent(f"""\
            Task Description:\nBased on the above definition, determine whether the given meme is a pun meme or not. The caption within the meme is "{caption}". You only need to respond with "pun" or "non-pun". Do not output any other content.\n\nYour Response:
            """)
        # 3. Only image, few-shot
        elif prompt_mode == 3:
            instruction = textwrap.dedent("""\
            Example and Task Description:\nHere we label the four provided memes in order as the first, second, third, and fourth meme. Based on the definition above, we can know that:
            The first meme is a pun meme;
            The second meme is a non-pun meme;
            The third meme is also a non-pun meme.
            Please determine whether the fourth meme is a pun meme or not, according to the definition and the categories of the first three memes. You only need to respond with "pun" or "non-pun". Do not output any other content.\n\nYour Response:
            """)
        # 4. Image & caption, few-shot
        elif prompt_mode == 4:
            instruction = lambda caption1, caption2, caption3, caption4: textwrap.dedent(f"""\
            Example and Task Description:\nHere we label the four provided memes in order as the first, second, third, and fourth meme. Based on the definition above, we can know that:
            The caption within the first meme is "{caption1}" and it is a pun meme;
            The caption within the second meme is "{caption2}" and it is a non-pun meme;
            The caption within the third meme is "{caption3}" and it is also a non-pun meme.
            Please determine whether the fourth meme is a pun meme or not, according to the definition and the categories of the first three memes. The caption within the fourth meme is "{caption4}". You only need to respond with "pun" or "non-pun". Do not output any other content.\n\nYour Response:
            """)
        # 5. Image & caption (with Chinese Pinyin), few-shot
        elif prompt_mode == 5:
            instruction = lambda caption1, caption2, caption3, caption4: textwrap.dedent(f"""\
            Example and Task Description:\nHere we label the four provided memes in order as the first, second, third, and fourth meme. Based on the definition above, we can know that:
            The caption within the first meme is "{caption1}", and the Chinese pinyin of the caption is "{py(caption1)}". The first meme is a pun meme;
            The caption within the second meme is "{caption2}", and the Chinese pinyin of the caption is "{py(caption2)}". The second meme is a non-pun meme;
            The caption within the third meme is "{caption3}", and the Chinese pinyin of the caption is "{py(caption3)}". The third meme is also a non-pun meme.
            Please determine whether the fourth meme is a pun meme or not, according to the definition and the categories of the first three memes. The caption within the fourth meme is "{caption4}", and the Chinese pinyin of the caption is "{py(caption4)}". You only need to respond with "pun" or "non-pun". Do not output any other content.\n\nYour Response:
            """)
        # 6. Image & caption, few-shot + CoT
        elif prompt_mode == 6:
            instruction = lambda caption1, caption2, caption3, caption4: textwrap.dedent(f"""\
            Example and Task Description:\nHere we label the four provided memes in order as the first, second, third, and fourth meme. Based on the definition above, we can know that:
            Meme Analysis: The first meme shows a woman in the center of the image, with a man on each side. Their faces are obscured and marked with the Chinese character for "男" and the caption reads "{caption1}", which is a homophone for "左右为难". This meme humorously conveys the situation where the woman is caught between two men and doesn't know what to do, generally referring to someone facing a tough decision and unable to choose.\nMeme Judgment: pun
            Meme Analysis: The second meme depicts a man and a woman walking together, with a caption at the bottom reading "{caption2}", which is not a homophone for any other words. This meme specifically describes someone who occasionally checks or pays attention to the dynamics of a group or chat, similar to how a manager or administrator might supervise or inspect, often with a playful tone.\nMeme Judgment: non-pun
            Meme Analysis: The third meme shows a cat gently pulled from both sides by two hands, looking somewhat helpless. The caption reads "{caption3}", which is not a homophone for any other words. This meme describes a person feeling confused and indecisive when faced with a choice, unable to decide which side to take.\nMeme Judgment: non-pun
            Please determine whether the fourth meme is a pun meme or not, according to the definition and the categories of the first three memes. The caption within the fourth meme is "{caption4}". Strictly follow the output format above: first analyze the fourth meme, then judge if it is a "pun" or "non-pun". Do not output any other content.\n\nYour Response:
            """)
        # 7. Image & caption (with Chinese Pinyin), few-shot + CoT
        else:
            instruction = lambda caption1, caption2, caption3, caption4: textwrap.dedent(f"""\
            Example and Task Description:\nHere we label the four provided memes in order as the first, second, third, and fourth meme. Based on the definition above, we can know that:
            Meme Analysis: The first meme shows a woman in the center of the image, with a man on each side. Their faces are obscured and marked with the Chinese character for "男" and the caption reads "{caption1}". The caption's Chinese pinyin is "{py(caption1)}", which is a homophone for "左右为难". This meme humorously conveys the situation where the woman is caught between two men and doesn't know what to do, generally referring to someone facing a tough decision and unable to choose.\nMeme Judgment: pun
            Meme Analysis: The second meme depicts a man and a woman walking together, with a caption at the bottom reading "{caption2}". The caption's Chinese pinyin is "{py(caption2)}", which is not a homophone for any other words. This meme specifically describes someone who occasionally checks or pays attention to the dynamics of a group or chat, similar to how a manager or administrator might supervise or inspect, often with a playful tone.\nMeme Judgment: non-pun
            Meme Analysis: The third meme shows a cat gently pulled from both sides by two hands, looking somewhat helpless. The caption reads "{caption3}". The caption's Chinese pinyin is "{py(caption3)}", which is not a homophone for any other words. This meme describes a person feeling confused and indecisive when faced with a choice, unable to decide which side to take.\nMeme Judgment: non-pun
            Please determine whether the fourth meme is a pun meme or not, according to the definition and the categories of the first three memes. The caption within the fourth meme is "{caption4}", and the Chinese pinyin of the caption is "{py(caption4)}". Strictly follow the output format above: first analyze the fourth meme, then judge if it is a "pun" or "non-pun". Do not output any other content.\n\nYour Response:
            """)

    # Call VLM to Response
    random.seed(2025)
    random.shuffle(dataset); random.shuffle(dataset)
    for img in tqdm(dataset, desc='1.1 Pun Meme Detection'):
        # Skip the data that has already been judged
        if record[model][language][mode[prompt_mode]][img]['model_output'] is not None:
            continue
        image = Image.open(os.path.join(dataset_path, img))
        caption = meta[img]['caption_1'].replace('\n', '')
        # 1. Only image, zero-shot
        if prompt_mode == 1:
            task_prompt = pun_meme_definition + '\n' + instruction
            images = image
        # 2. Image & caption, zero-shot
        elif prompt_mode == 2:
            task_prompt = pun_meme_definition + '\n' + instruction(caption)
            images = image
        # 3. Only image, few-shot
        elif prompt_mode == 3:
            task_prompt = pun_meme_definition + '\n' + instruction
            images = examples[0] + [image]
        # 4. Image & caption, few-shot
        # 5. Image & caption (with Chinese Pinyin), few-shot
        # 6. Image & caption, few-shot + CoT
        # 7. Image & caption (with Chinese Pinyin), few-shot + CoT
        else:
            task_prompt = pun_meme_definition + '\n' + instruction(*(examples[1] + [caption]))
            images = examples[0] + [image]
        # Remove extra spaces before lines
        task_prompt = '\n'.join([line.strip() for line in task_prompt.split('\n')]).strip('\n')
        # Call VLM
        output = call[model](client=client, image=images, text_prompt=task_prompt, temperature=temperature)
        # Print image, prompt and model output
        if not save:
            if type(images) != list:
                plt.figure(); plt.imshow(images); plt.show()
            else:
                for image in images:
                    plt.figure(); plt.imshow(image); plt.show()
            print(task_prompt); print('*'*60); print(output)
            break
        # Record
        if 'CoT' not in mode[prompt_mode]:
            temp = output.strip().lower()
            if temp[0:3] == '非双关' or temp == 'non-pun':
                judgment = 'non-pun'
            elif temp[0:2] == '双关' or temp == 'pun':
                judgment = 'pun'
            else:
                judgment = None  # Rely on subsequent processing
        else:
            if '判断：' in output or 'judgment:' in output.lower():
                temp = output.lower().split('判断：')[-1].split('judgment:')[-1].strip()
                if temp[0:3] == '非双关' or temp == 'non-pun':
                    judgment = 'non-pun'
                elif temp[0:2] == '双关' or temp == 'pun':
                    judgment = 'pun'
                else:
                    judgment = None  # Rely on subsequent processing
            else:
                judgment = None
        record[model][language][mode[prompt_mode]][img] = {'model_output': output,
                                                           'model_judgment': judgment}
        # Save
        if save:
            save_json_file(record, save_path)


def vlm_pun_meme_sentiment_analysis(dataset_path:str, model:str, client, language:str, prompt_mode:int, pilot:bool=True, temperature:float=0.7, save:bool=False):
    """
    Test VLMs' performance on pun meme sentiment analysis (target:self/both/other  type:positive/neutral/negative)
    Consider seven prompt modes:
    - 1: Only image, zero-shot
    - 2: Image & caption, zero-shot
    - 3: Only image, few-shot
    - 4: Image & caption, few-shot
    - 5: Image & caption (with Chinese Pinyin), few-shot
    - 6: Image & caption, few-shot + CoT
    - 7: Image & caption (with Chinese Pinyin), few-shot + CoT
    """
    def upsampling(dataset, meta, num_per_target:int=400):
        # Upsampling based on sentiment target
        random.seed(2025)
        new_dataset = []
        for target in ['self', 'both', 'other']:
            subset= [img for img in dataset if meta[img]['meme_sentiment'].split('-')[0]==target]
            repeat_time, remainder = num_per_target // len(subset), num_per_target % len(subset)
            new_subset, i = [], 0
            while i < repeat_time:
                new_subset.extend([f'{img}@{i+1}' for img in subset])
                i += 1
            if remainder > 0:
                new_subset.extend([f'{img}@{i+1}' for img in random.sample(subset, remainder)])
            new_dataset.extend(new_subset)
        new_dataset = sorted(new_dataset)
        return new_dataset
    def sentiment_en_to_cn(sentiment):
        f1 = {'self':'自己', 'both':'双方', 'other':'他人'}
        f2 = {'positive':'积极', 'neutral':'中性', 'negative':'消极'}
        target, sent_type = sentiment.split('-')
        return f'{f1[target]}-{f2[sent_type]}'
    def sentiment_cn_to_en(sentiment):
        f1 = {'自己':'self', '双方':'both', '他人':'other'}
        f2 = {'积极':'positive', '中性':'neutral', '消极':'negative'}
        try:
            target, sent_type = sentiment.split('-')
            return f'{f1[target]}-{f2[sent_type]}'
        except:
            return None
    def json_sentiment(sentiment, language):
        if language == 'cn':
            sentiment = sentiment_en_to_cn(sentiment)
            target, sent_type = sentiment.split('-')
            return f'{{"情感指向对象": "{target}", "情感类别": "{sent_type}"}}'
        else:
            target, sent_type = sentiment.split('-')
            return f'{{"Sentiment Target": "{target}", "Sentiment Type": "{sent_type}"}}'
    def parse_json_sentiment(sentiment, language):
        try:
            sentiment = eval(sentiment.lower().strip())
            if language == 'cn':
                target, sent_type = sentiment['情感指向对象'], sentiment['情感类别']
                return sentiment_cn_to_en(f'{target}-{sent_type}')
            else:
                target, sent_type = sentiment['sentiment target'], sentiment['sentiment type']
                return f'{target}-{sent_type}'
        except:
            return None

    call = {'gpt-4o-2024-08-06': call_gpt4o,
            'gpt-4o-mini-2024-07-18': call_gpt4o_mini,
            'claude-3-5-sonnet-20241022': call_claude35sonnet,
            'qwen2.5-vl-72b-instruct': call_qwen2_vl,
            'qwen2.5-vl-7b-instruct': call_qwen2_vl,
            'intern-vl2_5-78b': call_intern_vl2,
            'intern-vl2_5-38b': call_intern_vl2,
            'intern-vl2_5-26b': call_intern_vl2,
            'intern-vl2_5-8b': call_intern_vl2,
#            'llava-onevision-qwen2-72b-ov-chat': call_llava_onevision,
#            'llava-onevision-qwen2-7b-ov': call_llava_onevision,
            'minicpm-v-2_6': call_minicpm}
    mode = [None, 'only_image(zero_shot)', 'image_&_caption(zero_shot)',
            'only_image(few_shot)', 'image_&_caption(few_shot)', 'image_&_caption_&_pinyin(few_shot)',
            'image_&_caption(few_shot+CoT)', 'image_&_caption_&_pinyin(few_shot+CoT)']
    assert language in ['en', 'cn']
    assert model in list(call.keys())
    assert 0 < prompt_mode < len(mode)

    # Dataset Meta
    meta_path = os.path.join(dataset_path, '_Meta.json')
    meta = load_json_file(meta_path)
    example_ids = ['CN00601.png', 'CN00603.png', 'CN00604.png']  # Exclude example from the evaluation
    if pilot:
        dataset = [img for img in meta if meta[img]['meme_type']=='pun' and img not in example_ids and int(img.split('.')[0][2:])<=100]  # Demo
    else:
        dataset = [img for img in meta if meta[img]['meme_type']=='pun' and img not in example_ids]  # Whole dataset
        dataset = upsampling(dataset=sorted(dataset), meta=meta)  # Upsampling
    # Examples
    examples = [[], [], []]
    for img in example_ids:
        image = Image.open(os.path.join(dataset_path, img))
        caption = meta[img]['caption_1']
        sentiment = meta[img]['meme_sentiment']
        examples[0].append(image); examples[1].append(caption); examples[2].append(sentiment)
    # Save Path
    save_path = './results/pun_meme_eval/meme_sentiment_analysis.json'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    record = extend_dict(load_json_file(save_path), [model, language, mode[prompt_mode]])
    if record[model][language][mode[prompt_mode]] == dict():
        record[model][language][mode[prompt_mode]] = {img:{'model_output':None,'model_judgment':None} for img in dataset}

    # Task Prompt
    py = lambda text: ' '.join(chinese_character_to_pinyin(text, pattern='normal'))
    if language == 'cn':
        pun_meme_definition = textwrap.dedent("""\
        定义：\n对于由图像和字幕组成的表情包，如果表情包字幕利用一词多义或谐音多义的方式有意使表情包具有两层或多层含义，且至少其中一层含义基于表情包的画面元素，则称该表情包为“双关表情包”，否则称为“非双关表情包”。
        """)
        # 1. Only image, zero-shot
        if prompt_mode == 1:
            instruction = textwrap.dedent("""\
            任务描述：\n给定的表情包图片是满足以上定义的双关表情包，请你判断该表情包用于网络聊天时表达的情感，包括情感指向对象（自己、双方、他人）和情感类别（积极、中性、消极）。你只需直接回答表情包的情感，回答格式为JSON格式：{"情感指向对象": "XXX", "情感类别": "XXX"}，不要输出其他内容。\n\n你的回答：
            """)
        # 2. Image & caption, zero-shot
        elif prompt_mode == 2:
            instruction = lambda caption: textwrap.dedent(f"""\
            任务描述：\n给定的表情包图片是满足以上定义的双关表情包。请你判断该表情包用于网络聊天时表达的情感，包括情感指向对象（自己、双方、他人）和情感类别（积极、中性、消极）。该表情包内的字幕为“{caption}”。你只需直接回答表情包的情感，回答格式为JSON格式：{{"情感指向对象": "XXX", "情感类别": "XXX"}}，不要输出其他内容。\n\n你的回答：
            """)
        # 3. Only image, few-shot
        elif prompt_mode == 3:
            instruction = lambda sent1, sent2, sent3: textwrap.dedent(f"""\
            示例和任务描述：\n给定的四张表情包图片都是满足以上定义的双关表情包，现将这四张表情包按顺序分别记为第一、二、三、四张表情包。表情包在网络聊天时表达的情感包括情感指向对象（自己、双方、他人）和情感类别（积极、中性、消极），由此可知：
            第一张表情包的情感是{json_sentiment(sent1, language=language)}；
            第二张表情包的情感是{json_sentiment(sent2, language=language)}；
            第三张表情包的情感是{json_sentiment(sent3, language=language)}。
            请你根据前三张表情包的情感，判断第四张表情包用于网络聊天时表达的情感。你只需直接回答表情包的情感，回答格式为JSON格式：{{"情感指向对象": "XXX", "情感类别": "XXX"}}，不要输出其他内容。\n\n你的回答：
            """)
        # 4. Image & caption, few-shot
        elif prompt_mode == 4:
            instruction = lambda caption1, caption2, caption3, caption4, sent1, sent2, sent3: textwrap.dedent(f"""\
            示例和任务描述：\n给定的四张表情包图片都是满足以上定义的双关表情包，现将这四张表情包按顺序分别记为第一、二、三、四张表情包。表情包在网络聊天时表达的情感包括情感指向对象（自己、双方、他人）和情感类别（积极、中性、消极），由此可知：
            第一张表情包内的字幕为“{caption1}”，情感是{json_sentiment(sent1, language=language)}；
            第二张表情包内的字幕为“{caption2}”，情感是{json_sentiment(sent2, language=language)}；
            第三张表情包内的字幕为“{caption3}”，情感是{json_sentiment(sent3, language=language)}。
            请你根据前三张表情包的情感，判断第四张表情包用于网络聊天时表达的情感。第四张表情包内的字幕为“{caption4}”。你只需直接回答表情包的情感，回答格式为JSON格式：{{"情感指向对象": "XXX", "情感类别": "XXX"}}，不要输出其他内容。\n\n你的回答：
            """)
        # 5. Image & caption (with Chinese Pinyin), few-shot
        elif prompt_mode == 5:
            instruction = lambda caption1, caption2, caption3, caption4, sent1, sent2, sent3: textwrap.dedent(f"""\
            示例和任务描述：\n给定的四张表情包图片都是满足以上定义的双关表情包，现将这四张表情包按顺序分别记为第一、二、三、四张表情包。表情包在网络聊天时表达的情感包括情感指向对象（自己、双方、他人）和情感类别（积极、中性、消极），由此可知：
            第一张表情包内的字幕为“{caption1}”，字幕的中文拼音为“{py(caption1)}”，表情包情感是{json_sentiment(sent1, language=language)}；
            第二张表情包内的字幕为“{caption2}”，字幕的中文拼音为“{py(caption2)}”，表情包情感是{json_sentiment(sent2, language=language)}；
            第三张表情包内的字幕为“{caption3}”，字幕的中文拼音为“{py(caption3)}”，表情包情感是{json_sentiment(sent3, language=language)}。
            请你根据前三张表情包的情感，判断第四张表情包用于网络聊天时表达的情感。第四张表情包内的字幕为“{caption4}”，字幕的中文拼音为“{py(caption4)}”。你只需直接回答表情包的情感，回答格式为JSON格式：{{"情感指向对象": "XXX", "情感类别": "XXX"}}，不要输出其他内容。\n\n你的回答：
            """)
        # 6. Image & caption, few-shot + CoT
        elif prompt_mode == 6:
            instruction = lambda caption1, caption2, caption3, caption4, sent1, sent2, sent3: textwrap.dedent(f"""\
            示例和任务描述：\n给定的四张表情包图片都是满足以上定义的双关表情包，现将这四张表情包按顺序分别记为第一、二、三、四张表情包。表情包在网络聊天时表达的情感包括情感指向对象（自己、双方、他人）和情感类别（积极、中性、消极），由此可知：
            表情包分析：第一张表情包的画面中央是一位女性，左右两边各有一位男性，他们的脸被遮住，并标有“男”字，字幕是“{caption1}”，谐音“左右为难”。该表情包表达了一种幽默感，意指画面中女性被两位男性夹在中间不知所措的情景，更是泛指某人在两难的情况下左右为难，无法做出选择。\n表情包情感：{json_sentiment(sent1, language=language)}
            表情包分析：第二张表情包中展示了一只玩具熊被夹在晾衣架上，头和身体分开，字幕写着“{caption2}”，不谐音其他文字。这个表情包利用了“分头”一词的双关表达，即指画面中的头与身体分开，更是幽默地表达了彼此分开行动或分工合作的含义。\n表情包情感：{json_sentiment(sent2, language=language)}
            表情包分析：第三张表情包中一个可爱的小方块人物骑在白马背上，面带微笑，字幕写着“{caption3}”，不谐音其他文字。“{caption3}”是一种祝福语，寓意事情顺利，很快取得成功。表情包利用马的画面形象和成语“{caption3}”幽默地表达了迅速达成目标的美好愿望，多用于祝福他人。\n表情包情感：{json_sentiment(sent3, language=language)}
            请你根据前三张表情包的情感，判断第四张表情包用于网络聊天时表达的情感。第四张表情包内的字幕为“{caption4}”。要求严格按照上面的输出格式，先对第四张表情包进行分析，再给出其JSON格式的情感：{{"情感指向对象": "XXX", "情感类别": "XXX"}}，不要输出其他内容。\n\n你的回答：
            """)
        # 7. Image & caption (with Chinese Pinyin), few-shot + CoT
        else:
            instruction = lambda caption1, caption2, caption3, caption4, sent1, sent2, sent3: textwrap.dedent(f"""\
            示例和任务描述：\n给定的四张表情包图片都是满足以上定义的双关表情包，现将这四张表情包按顺序分别记为第一、二、三、四张表情包。表情包在网络聊天时表达的情感包括情感指向对象（自己、双方、他人）和情感类别（积极、中性、消极），由此可知：
            表情包分析：第一张表情包的画面中央是一位女性，左右两边各有一位男性，他们的脸被遮住，并标有“男”字，字幕是“{caption1}”。字幕的中文拼音为“{py(caption1)}”，谐音“左右为难”。该表情包表达了一种幽默感，意指画面中女性被两位男性夹在中间不知所措的情景，更是泛指某人在两难的情况下左右为难，无法做出选择。\n表情包情感：{json_sentiment(sent1, language=language)}
            表情包分析：第二张表情包中展示了一只玩具熊被夹在晾衣架上，头和身体分开，字幕写着“{caption2}”。字幕的中文拼音为“{py(caption2)}”，不谐音其他文字。这个表情包利用了“分头”一词的双关表达，即指画面中的头与身体分开，更是幽默地表达了彼此分开行动或分工合作的含义。\n表情包情感：{json_sentiment(sent2, language=language)}
            表情包分析：第三张表情包中一个可爱的小方块人物骑在白马背上，面带微笑，字幕写着“{caption3}”。字幕的中文拼音为“{py(caption3)}”，不谐音其他文字。“{caption3}”是一种祝福语，寓意事情顺利，很快取得成功。表情包利用马的画面形象和成语“{caption3}”幽默地表达了迅速达成目标的美好愿望，多用于祝福他人。\n表情包情感：{json_sentiment(sent3, language=language)}
            请你根据前三张表情包的情感，判断第四张表情包用于网络聊天时表达的情感。第四张表情包内的字幕为“{caption4}”，字幕的中文拼音为“{py(caption4)}”。要求严格按照上面的输出格式，先对第四张表情包进行分析，再给出其JSON格式的情感：{{"情感指向对象": "XXX", "情感类别": "XXX"}}，不要输出其他内容。\n\n你的回答：
            """)
    else:
        pun_meme_definition = textwrap.dedent("""\
        Definition:\nFor memes consisting of images and captions, if the caption intentionally uses polysemy or homophony to create two or more meanings, with at least one meaning based on the visual elements of the image, the meme is called a "pun meme". Otherwise, it is called a "non-pun meme".
        """)
        # 1. Only image, zero-shot
        if prompt_mode == 1:
            instruction = textwrap.dedent("""\
            Task Description:\nThe given meme image is a pun meme that meets the above definition. Please determine the sentiment conveyed by the meme when used in online chat, including the sentiment target (self, both, other) and the sentiment type (positive, neutral, negative). You only need to respond with the meme sentiment in the JSON format {"Sentiment Target": "XXX", "Sentiment Type": "XXX"}. Do not output any other content.\n\nYour Response:
            """)
        # 2. Image & caption, zero-shot
        elif prompt_mode == 2:
            instruction = lambda caption: textwrap.dedent(f"""\
            Task Description:\nThe given meme image is a pun meme that meets the above definition. Please determine the sentiment conveyed by the meme when used in online chat, including the sentiment target (self, both, other) and the sentiment type (positive, neutral, negative). The caption within the meme is "{caption}". You only need to respond with the meme sentiment in the JSON format {{"Sentiment Target": "XXX", "Sentiment Type": "XXX"}}. Do not output any other content.\n\nYour Response:
            """)
        # 3. Only image, few-shot
        elif prompt_mode == 3:
            instruction = lambda sent1, sent2, sent3: textwrap.dedent(f"""\
            Example and Task Description:\nThe four given meme images are pun memes that meet the above definition. Here we label these four memes in order as the first, second, third, and fourth meme. The sentiment conveyed by memes in online chat includes the sentiment target (self, both, other) and the sentiment type (positive, neutral, negative). From this, we can know that:
            The sentiment of the first meme is {json_sentiment(sent1, language=language)};
            The sentiment of the second meme is {json_sentiment(sent2, language=language)};
            The sentiment of the third meme is {json_sentiment(sent3, language=language)}.
            Please determine the sentiment conveyed by the fourth meme when used in online chat, based on the sentiments of the first three memes. You only need to respond with the meme sentiment in the JSON format {{"Sentiment Target": "XXX", "Sentiment Type": "XXX"}}. Do not output any other content.\n\nYour Response:
            """)
        # 4. Image & caption, few-shot
        elif prompt_mode == 4:
            instruction = lambda caption1, caption2, caption3, caption4, sent1, sent2, sent3: textwrap.dedent(f"""\
            Example and Task Description:\nThe four given meme images are pun memes that meet the above definition. Here we label these four memes in order as the first, second, third, and fourth meme. The sentiment conveyed by memes in online chat includes the sentiment target (self, both, other) and the sentiment type (positive, neutral, negative). From this, we can know that:
            The caption within the first meme is "{caption1}" and the sentiment of the meme is {json_sentiment(sent1, language=language)};
            The caption within the second meme is "{caption2}" and the sentiment of the meme is {json_sentiment(sent2, language=language)};
            The caption within the third meme is "{caption3}" and the sentiment of the meme is {json_sentiment(sent3, language=language)}.
            Please determine the sentiment conveyed by the fourth meme when used in online chat, based on the sentiments of the first three memes. The caption within the fourth meme is "{caption4}". You only need to respond with the meme sentiment in the JSON format {{"Sentiment Target": "XXX", "Sentiment Type": "XXX"}}. Do not output any other content.\n\nYour Response:
            """)
        # 5. Image & caption (with Chinese Pinyin), few-shot
        elif prompt_mode == 5:
            instruction = lambda caption1, caption2, caption3, caption4, sent1, sent2, sent3: textwrap.dedent(f"""\
            Example and Task Description:\nThe four given meme images are pun memes that meet the above definition. Here we label these four memes in order as the first, second, third, and fourth meme. The sentiment conveyed by memes in online chat includes the sentiment target (self, both, other) and the sentiment type (positive, neutral, negative). From this, we can know that:
            The caption within the first meme is "{caption1}", and the Chinese pinyin of the caption is "{py(caption1)}". The sentiment of the first meme is {json_sentiment(sent1, language=language)};
            The caption within the second meme is "{caption2}", and the Chinese pinyin of the caption is "{py(caption2)}". The sentiment of the second meme is {json_sentiment(sent2, language=language)};
            The caption within the third meme is "{caption3}", and the Chinese pinyin of the caption is "{py(caption3)}". The sentiment of the third meme is {json_sentiment(sent3, language=language)}.
            Please determine the sentiment conveyed by the fourth meme when used in online chat, based on the sentiments of the first three memes. The caption within the fourth meme is "{caption4}", and the Chinese pinyin of the caption is "{py(caption4)}". You only need to respond with the meme sentiment in the JSON format {{"Sentiment Target": "XXX", "Sentiment Type": "XXX"}}. Do not output any other content.\n\nYour Response:
            """)
        # 6. Image & caption, few-shot + CoT
        elif prompt_mode == 6:
            instruction = lambda caption1, caption2, caption3, caption4, sent1, sent2, sent3: textwrap.dedent(f"""\
            Example and Task Description:\nThe four given meme images are pun memes that meet the above definition. Here we label these four memes in order as the first, second, third, and fourth meme. The sentiment conveyed by memes in online chat includes the sentiment target (self, both, other) and the sentiment type (positive, neutral, negative). From this, we can know that:
            Meme Analysis: The first meme shows a woman in the center of the image, with a man on each side. Their faces are obscured and marked with the Chinese character for "男" and the caption reads "{caption1}", which is a homophone for "左右为难". This meme humorously conveys the situation where the woman is caught between two men and doesn't know what to do, generally referring to someone facing a tough decision and unable to choose.\nMeme Sentiment: {json_sentiment(sent1, language=language)}
            Meme Analysis: The second meme shows a teddy bear caught in a clothes hanger, with its head and body separated, and the caption reads "{caption2}", which is not a homophone for any other words. This meme uses the pun of "分头", indicating the head and body are separated in the image, humorously implying acting separately or working in collaboration.\nMeme Sentiment: {json_sentiment(sent2, language=language)}
            Meme Analysis: The third meme depicts a cute cube character riding a white horse and smiling, with the caption "{caption3}", which does not sound like other words. "{caption3}" is a blessing, meaning things will go smoothly and success will come quickly. The meme humorously uses the image of a horse and the idiom "{caption3}" to express a wish for quickly achieving goals, often used to bless others.\nMeme Sentiment: {json_sentiment(sent3, language=language)}
            Please determine the sentiment conveyed by the fourth meme when used in online chat, based on the sentiments of the first three memes. The caption within the fourth meme is "{caption4}". Strictly follow the output format above: first analyze the fourth meme, then give its sentiment in the JSON format {{"Sentiment Target": "XXX", "Sentiment Type": "XXX"}}. Do not output any other content.\n\nYour Response:
            """)
        # 7. Image & caption (with Chinese Pinyin), few-shot + CoT
        else:
            instruction = lambda caption1, caption2, caption3, caption4, sent1, sent2, sent3: textwrap.dedent(f"""\
            Example and Task Description:\nThe four given meme images are pun memes that meet the above definition. Here we label these four memes in order as the first, second, third, and fourth meme. The sentiment conveyed by memes in online chat includes the sentiment target (self, both, other) and the sentiment type (positive, neutral, negative). From this, we can know that:
            Meme Analysis: The first meme shows a woman in the center of the image, with a man on each side. Their faces are obscured and marked with the Chinese character for "男" and the caption reads "{caption1}". The caption's Chinese pinyin is "{py(caption1)}", which is a homophone for "左右为难". This meme humorously conveys the situation where the woman is caught between two men and doesn't know what to do, generally referring to someone facing a tough decision and unable to choose.\nMeme Sentiment: {json_sentiment(sent1, language=language)}
            Meme Analysis: The second meme shows a teddy bear caught in a clothes hanger, with its head and body separated, and the caption reads "{caption2}". The caption's Chinese pinyin is "{py(caption2)}", which is not a homophone for any other words. This meme uses the pun of "分头", indicating the head and body are separated in the image, humorously implying acting separately or working in collaboration.\nMeme Sentiment: {json_sentiment(sent2, language=language)}
            Meme Analysis: The third meme depicts a cute cube character riding a white horse and smiling, with the caption "{caption3}". The Chinese pinyin of the caption is "{py(caption3)}", which does not sound like other words. "{caption3}" is a blessing, meaning things will go smoothly and success will come quickly. The meme humorously uses the image of a horse and the idiom "{caption3}" to express a wish for quickly achieving goals, often used to bless others.\nMeme Sentiment: {json_sentiment(sent3, language=language)}
            Please determine the sentiment conveyed by the fourth meme when used in online chat, based on the sentiments of the first three memes. The caption within the fourth meme is "{caption4}", and the Chinese pinyin of the caption is "{py(caption4)}". Strictly follow the output format above: first analyze the fourth meme, then give its sentiment in the JSON format {{"Sentiment Target": "XXX", "Sentiment Type": "XXX"}}. Do not output any other content.\n\nYour Response:
            """)

    # Call VLM to Response
    random.seed(2025)
    random.shuffle(dataset); random.shuffle(dataset)
    for img in tqdm(dataset, desc='1.2 Meme Sentiment Analysis'):
        # Skip the data that has already been judged
        if record[model][language][mode[prompt_mode]][img]['model_output'] is not None:
            continue
        img0 = img.split('@')[0]
        image = Image.open(os.path.join(dataset_path, img0))
        caption = meta[img0]['caption_1'].replace('\n', '')
        # 1. Only image, zero-shot
        if prompt_mode == 1:
            task_prompt = pun_meme_definition + '\n' + instruction
            images = image
        # 2. Image & caption, zero-shot
        elif prompt_mode == 2:
            task_prompt = pun_meme_definition + '\n' + instruction(caption)
            images = image
        # 3. Only image, few-shot
        elif prompt_mode == 3:
            task_prompt = pun_meme_definition + '\n' + instruction(*examples[2])
            images = examples[0] + [image]
        # 4. Image & caption, few-shot
        # 5. Image & caption (with Chinese Pinyin), few-shot
        # 6. Image & caption, few-shot + CoT
        # 7. Image & caption (with Chinese Pinyin), few-shot + CoT
        else:
            task_prompt = pun_meme_definition + '\n' + instruction(*(examples[1] + [caption] + examples[2]))
            images = examples[0] + [image]
        # Remove extra spaces before lines
        task_prompt = '\n'.join([line.strip() for line in task_prompt.split('\n')]).strip('\n')
        # Call VLM
        output = call[model](client=client, image=images, text_prompt=task_prompt, temperature=temperature)
        # Print image, prompt and model output
        if not save:
            if type(images) != list:
                plt.figure(); plt.imshow(images); plt.show()
            else:
                for image in images:
                    plt.figure(); plt.imshow(image); plt.show()
            print(task_prompt); print('*'*60); print(output)
            break
        # Record
        if '{' in output and '}' in output:
            temp = output[output.index('{'): output.index('}')+1]
            judgment = parse_json_sentiment(temp, language=language)
        else:
            judgment = None  # Rely on subsequent processing
        record[model][language][mode[prompt_mode]][img] = {'model_output': output,
                                                           'model_judgment': judgment}
        # Save
        if save:
            save_json_file(record, save_path)


def vlm_chat_driven_meme_response(dataset_path:str, difficulty:str, model:str, client, language:str, prompt_mode:int, pilot:bool=True, temperature:float=0.0, save:bool=False):
    """
    Test VLMs' performance on chat-driven meme response (choose one correct meme out of four)
    Consider seven prompt modes:
    - 1: Only image, zero-shot
    - 2: Image & caption, zero-shot
    - 3: Only image, few-shot
    - 4: Image & caption, few-shot
    - 5: Image & caption (with Chinese Pinyin), few-shot
    - 6: Image & caption, few-shot + CoT
    - 7: Image & caption (with Chinese Pinyin), few-shot + CoT
    """
    def get_example_analysis(example_captions, prompt_mode, language):
        if language == 'cn':
            if 'pinyin' in mode[prompt_mode]:
                example_analysis = textwrap.dedent(f"""\
                第一张表情包中有一个可爱的蛋造型的卡通形象，脸上带有一滴汗水，表情略显无奈。下方配有字幕“{example_captions[1]}”，字幕的中文拼音为“{py(example_captions[1])}”。这个表情包通过图文双关表达自嘲或调侃，表示自己在经历不幸或倒霉的事情，相当于在说“是我，倒霉的那个人”。
                第二张表情包中，一只卡通鸭子背后有一个气阀正在排气，旁边有字幕“{example_captions[2]}”，字幕的中文拼音为“{py(example_captions[2])}”。这是一种双关的幽默表达，把“放气了”与“放弃了”结合在一起（二者谐音），表现出一种无奈放弃的情绪。
                第三张表情包展示了一只表情滑稽的汉堡，上面有一滴汗珠，表情看起来有些无奈或尴尬，底部的字幕是“{example_captions[3]}”，字幕的中文拼音为“{py(example_captions[3])}”。该表情包利用“吃堡了撑的”与“吃饱了撑的”的谐音表达了对某人做多余的事的无奈，带有调侃或讽刺的意味。
                第四张表情包的画面中，有一个卷心菜戴着耳机，面前有一台电脑和一个鼠标，卷心菜上贴着一张粉色便签纸，上面画了一个简单的表情。画面底部的字幕是“{example_captions[4]}”，字幕的中文拼音为“{py(example_captions[4])}”。这里的“菜”一语双关，表面上指画面里正看着电脑的卷心菜，实际上是形容在游戏中表现不佳的人，意思是“难道表现不佳就不能打游戏吗？”。这是一个自嘲或调侃自己游戏水平不高但仍然喜欢玩游戏的幽默表达。
                综上，第三张表情包最适合用来回复，因为甲乙两人主要在吐槽游戏策划对角色的改动很多余，故选C。
                """).strip('\n')
            else:
                example_analysis = textwrap.dedent(f"""\
                第一张表情包中有一个可爱的蛋造型的卡通形象，脸上带有一滴汗水，表情略显无奈。下方配有字幕“{example_captions[1]}”。这个表情包通过图文双关表达自嘲或调侃，表示自己在经历不幸或倒霉的事情，相当于在说“是我，倒霉的那个人”。
                第二张表情包中，一只卡通鸭子背后有一个气阀正在排气，旁边有字幕“{example_captions[2]}”。这是一种双关的幽默表达，把“放气了”与“放弃了”结合在一起（二者谐音），表现出一种无奈放弃的情绪。
                第三张表情包展示了一只表情滑稽的汉堡，上面有一滴汗珠，表情看起来有些无奈或尴尬，底部的字幕是“{example_captions[3]}”。该表情包利用“吃堡了撑的”与“吃饱了撑的”的谐音表达了对某人做多余的事的无奈，带有调侃或讽刺的意味。
                第四张表情包的画面中，有一个卷心菜戴着耳机，面前有一台电脑和一个鼠标，卷心菜上贴着一张粉色便签纸，上面画了一个简单的表情。画面底部的字幕是“{example_captions[4]}”。这里的“菜”一语双关，表面上指画面里正看着电脑的卷心菜，实际上是形容在游戏中表现不佳的人，意思是“难道表现不佳就不能打游戏吗？”。这是一个自嘲或调侃自己游戏水平不高但仍然喜欢玩游戏的幽默表达。
                综上，第三张表情包最适合用来回复，因为甲乙两人主要在吐槽游戏策划对角色的改动很多余，故选C。
                """).strip('\n')
        else:
            if 'pinyin' in mode[prompt_mode]:
                example_analysis = textwrap.dedent(f"""\
                The first meme features a cute cartoon egg character with a drop of sweat on its face, displaying a slightly helpless expression. The caption below reads "{example_captions[1]}" and the Chinese pinyin of the caption is "{py(example_captions[1])}". This meme uses a pun in both image and text to express self-deprecation or teasing, indicating that the person is experiencing something unfortunate or unlucky, akin to saying "It's me, the unlucky one."
                The second meme shows a cartoon duck with an exhaust valve releasing air behind it, accompanied by the caption "{example_captions[2]}". The caption's Chinese pinyin is "{py(example_captions[2])}", which is a homophone for "放弃了". This is a pun that combines "放气了 (letting out air)" with "放弃了 (giving up)" because they sound similar in Chinese, conveying a sense of resigned surrender.
                The third meme depicts a comical-looking hamburger with a drop of sweat, appearing somewhat helpless or embarrassed, with the caption "{example_captions[3]}". The caption's Chinese pinyin is "{py(example_captions[3])}", which is a homophone for "吃饱了撑的". This meme uses a pun on "吃堡了撑的 (stuffed from eating a burger)" and "吃饱了撑的 (looking for trouble when there's nothing else to do)", expressing frustration at someone doing unnecessary things, with a hint of mockery or sarcasm.
                The fourth meme shows a cabbage wearing headphones, with a computer and a mouse in front of it, and a pink sticky note with a simple face drawn on it. The caption at the bottom reads "{example_captions[4]}" and the Chinese pinyin of the caption is "{py(example_captions[4])}". The word "菜" is a pun, referring both to the cabbage in the picture and to someone performing poorly in a game. It means "Can't someone with poor performance still play games?" This is a humorous way of teasing oneself about having low gaming skills but still enjoying playing.
                Overall, the third meme is the most suitable for a reply because Person '甲' and Person '乙' are mainly complaining about the unnecessary changes made by the game planners to the characters, hence option C is chosen.
                """).strip('\n')
            else:
                example_analysis = textwrap.dedent(f"""\
                The first meme features a cute cartoon egg character with a drop of sweat on its face, displaying a slightly helpless expression. The caption below reads "{example_captions[1]}". This meme uses a pun in both image and text to express self-deprecation or teasing, indicating that the person is experiencing something unfortunate or unlucky, akin to saying "It's me, the unlucky one."
                The second meme shows a cartoon duck with an exhaust valve releasing air behind it, accompanied by the caption "{example_captions[2]}". This is a pun that combines "放气了 (letting out air)" with "放弃了 (giving up)" because they sound similar in Chinese, conveying a sense of resigned surrender.
                The third meme depicts a comical-looking hamburger with a drop of sweat, appearing somewhat helpless or embarrassed, with the caption "{example_captions[3]}". This meme uses a pun on "吃堡了撑的 (stuffed from eating a burger)" and "吃饱了撑的 (looking for trouble when there's nothing else to do)", expressing frustration at someone doing unnecessary things, with a hint of mockery or sarcasm.
                The fourth meme shows a cabbage wearing headphones, with a computer and a mouse in front of it, and a pink sticky note with a simple face drawn on it. The caption at the bottom reads "{example_captions[4]}". The word "菜" is a pun, referring both to the cabbage in the picture and to someone performing poorly in a game. It means "Can't someone with poor performance still play games?" This is a humorous way of teasing oneself about having low gaming skills but still enjoying playing.
                Overall, the third meme is the most suitable for a reply because Person '甲' and Person '乙' are mainly complaining about the unnecessary changes made by the game planners to the characters, hence option C is chosen.
                """).strip('\n')
        return example_analysis
    def get_chat_data(chat_id, first_half:bool, language:str):
        # Chat history
        chat_history = '\n'.join(chat[chat_id]['chat_history']) + '（表情包回复）'
        # Choices
        choices_img = [v for k,v in chat[chat_id]['meme_response'].items() if 'candidates' not in k]
        random.shuffle(choices_img); random.shuffle(choices_img)
        if language == 'cn':
            choices_text = ['A: 第一张表情包', 'B: 第二张表情包', 'C: 第三张表情包', 'D: 第四张表情包'] if first_half else \
                           ['A: 第五张表情包', 'B: 第六张表情包', 'C: 第七张表情包', 'D: 第八张表情包']
        else:
            choices_text = ['A: the first meme', 'B: the second meme', 'C: the third meme', 'D: the fourth meme'] if first_half else \
                           ['A: the fifth meme', 'B: the sixth meme', 'C: the seventh meme', 'D: the eighth meme']
        choices = dict(zip(choices_img, choices_text))
        # Meme images & captions
        meme_images, meme_captions = [], [None]
        for img in choices:
            meme_images.append(Image.open(os.path.join(dataset_path, img)))
            meme_captions.append(meta[img]['caption_1'])
        # Correct Answer
        correct_img = chat[chat_id]['meme_response']['correct']
        answer = choices[correct_img].split(':')[0]
        return meme_images, meme_captions, chat_history, choices, answer

    call = {'gpt-4o-2024-08-06': call_gpt4o,
            'gpt-4o-mini-2024-07-18': call_gpt4o_mini,
            'claude-3-5-sonnet-20241022': call_claude35sonnet,
            'qwen2.5-vl-72b-instruct': call_qwen2_vl,
            'qwen2.5-vl-7b-instruct': call_qwen2_vl,
            'intern-vl2_5-78b': call_intern_vl2,
            'intern-vl2_5-38b': call_intern_vl2,
            'intern-vl2_5-26b': call_intern_vl2,
            'intern-vl2_5-8b': call_intern_vl2,
#            'llava-onevision-qwen2-72b-ov-chat': call_llava_onevision,
#            'llava-onevision-qwen2-7b-ov': call_llava_onevision,
            'minicpm-v-2_6': call_minicpm}
    mode = [None, 'only_image(zero_shot)', 'image_&_caption(zero_shot)',
            'only_image(few_shot)', 'image_&_caption(few_shot)', 'image_&_caption_&_pinyin(few_shot)',
            'image_&_caption(few_shot+CoT)', 'image_&_caption_&_pinyin(few_shot+CoT)']
    assert language in ['en', 'cn']
    assert model in list(call.keys())
    assert difficulty in ['hard', 'easy']
    assert 0 < prompt_mode < len(mode)

    # Meta & Chat
    meta_path = os.path.join(dataset_path, f'_Meta.json')
    meta = load_json_file(meta_path)
    chat_path = os.path.join(dataset_path, f'_Chat({difficulty}).json')
    chat = load_json_file(chat_path)
    example_id = 'chat_00100'  # Exclude example from the evaluation
    if pilot:
        dataset = [chat_id for chat_id in chat if chat_id != example_id and int(chat_id.split('_')[-1])<=101]  # Demo
    else:
        dataset = [chat_id for chat_id in chat if chat_id != example_id]  # Whole dataset
    # Examples
    random.seed(2024)
    py = lambda text: ' '.join(chinese_character_to_pinyin(text, pattern='normal'))
    example_images, example_captions, example_chat_history, example_choices, example_answer = get_chat_data(chat_id=example_id, first_half=True, language=language)
    example_choices_string = '\n'.join(list(example_choices.values()))  # {'CN00135.png':'A: 第一张表情包', 'CN00595.png':'B: 第二张表情包', 'CN00031.png':'C: 第三张表情包', 'CN00507.png':'D: 第四张表情包'}
    example_analysis = get_example_analysis(example_captions=example_captions, prompt_mode=prompt_mode, language=language)
    # Save Path
    save_path = f'./results/pun_meme_eval/text_driven_meme_response.json'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    record = extend_dict(load_json_file(save_path), [difficulty, model, language, mode[prompt_mode]])
    if record[difficulty][model][language][mode[prompt_mode]] == dict():
        record[difficulty][model][language][mode[prompt_mode]] = {chat_id:{'choices':None,'model_output':None,'model_judgment':None} for chat_id in dataset}

    # Task Prompt
    if language == 'cn':
        pun_meme_definition = textwrap.dedent("""\
        定义：\n对于由图像和字幕组成的表情包，如果表情包字幕利用一词多义或谐音多义的方式有意使表情包具有两层或多层含义，且至少其中一层含义基于表情包的画面元素，则称该表情包为“双关表情包”，否则称为“非双关表情包”。
        """)
        # 1. Only image, zero-shot
        if prompt_mode == 1:
            instruction = lambda chat_history, choices: textwrap.dedent(f"""\
            任务描述：\n给定的四张表情包图片都是满足以上定义的双关表情包，现将这四张表情包按顺序分别记为第一、二、三、四张表情包。以下是甲、乙两人的一次网络聊天记录，缺少了最后的表情包回复。请你根据聊天内容，人物关系和情绪氛围，从四张表情包中选出最合适的一个作为回复。仅需回答代表选项的字母（A-D），不要输出其他内容。
            【聊天记录】：\n{chat_history}
            【选项】：\n{choices}\n\n你的回答：
            """)
        # 2. Image & caption, zero-shot
        elif prompt_mode == 2:
            instruction = lambda captions, chat_history, choices: textwrap.dedent(f"""\
            任务描述：\n给定的四张表情包图片都是满足以上定义的双关表情包，现将这四张表情包按顺序分别记为第一、二、三、四张表情包，其中：
            第一张表情包内的字幕为“{captions[1]}”；
            第二张表情包内的字幕为“{captions[2]}”；
            第三张表情包内的字幕为“{captions[3]}”；
            第四张表情包内的字幕为“{captions[4]}”。
            以下是甲、乙两人的一次网络聊天记录，缺少了最后的表情包回复。请你根据聊天内容，人物关系和情绪氛围，从四张表情包中选出最合适的一个作为回复。仅需回答代表选项的字母（A-D），不要输出其他内容。
            【聊天记录】：\n{chat_history}
            【选项】：\n{choices}\n\n你的回答：
            """)
        # 3. Only image, few-shot
        elif prompt_mode == 3:
            instruction = lambda chat_history1, choices1, answer1, chat_history2, choices2: textwrap.dedent(f"""\
            示例和任务描述：\n给定的八张表情包图片都是满足以上定义的双关表情包，现将这八张表情包按顺序分别记为第一、二、三、四、五、六、七、八张表情包。以下是甲、乙两人的两次网络聊天记录，都缺少了最后的表情包回复。要求第一次聊天只能从前四张表情包中进行选择，第二次聊天只能从后四张表情包中进行选择。请你根据聊天内容，人物关系和情绪氛围，从限定的四张表情包中选出最合适的一个作为回复。
            【聊天记录一】：\n{chat_history1}
            【选项】：\n{choices1}\n【正确答案】：{answer1}\n
            请你参考聊天记录一的例子，选出你认为的聊天记录二的正确答案。仅需回答代表选项的字母（A-D），不要输出其他内容。
            【聊天记录二】：\n{chat_history2}
            【选项】：\n{choices2}\n\n你的回答：
            """)
        # 4. Image & caption, few-shot
        elif prompt_mode == 4:
            instruction = lambda captions, chat_history1, choices1, answer1, chat_history2, choices2: textwrap.dedent(f"""\
            示例和任务描述：\n给定的八张表情包图片都是满足以上定义的双关表情包，现将这八张表情包按顺序分别记为第一、二、三、四、五、六、七、八张表情包，其中：
            第一张表情包内的字幕为“{captions[1]}”；\n第二张表情包内的字幕为“{captions[2]}”；
            第三张表情包内的字幕为“{captions[3]}”；\n第四张表情包内的字幕为“{captions[4]}”；
            第五张表情包内的字幕为“{captions[5]}”；\n第六张表情包内的字幕为“{captions[6]}”；
            第七张表情包内的字幕为“{captions[7]}”；\n第八张表情包内的字幕为“{captions[8]}”。
            以下是甲、乙两人的两次网络聊天记录，都缺少了最后的表情包回复。要求第一次聊天只能从前四张表情包中进行选择，第二次聊天只能从后四张表情包中进行选择。请你根据聊天内容，人物关系和情绪氛围，从限定的四张表情包中选出最合适的一个作为回复。
            【聊天记录一】：\n{chat_history1}
            【选项】：\n{choices1}\n【正确答案】：{answer1}\n
            请你参考聊天记录一的例子，选出你认为的聊天记录二的正确答案。仅需回答代表选项的字母（A-D），不要输出其他内容。
            【聊天记录二】：\n{chat_history2}
            【选项】：\n{choices2}\n\n你的回答：
            """)
        # 5. Image & caption (with Chinese Pinyin), few-shot
        elif prompt_mode == 5:
            instruction = lambda captions, chat_history1, choices1, answer1, chat_history2, choices2: textwrap.dedent(f"""\
            示例和任务描述：\n给定的八张表情包图片都是满足以上定义的双关表情包，现将这八张表情包按顺序分别记为第一、二、三、四、五、六、七、八张表情包，其中：
            第一张表情包内的字幕为“{captions[1]}”，字幕的中文拼音为“{py(captions[1])}”；
            第二张表情包内的字幕为“{captions[2]}”，字幕的中文拼音为“{py(captions[2])}”；
            第三张表情包内的字幕为“{captions[3]}”，字幕的中文拼音为“{py(captions[3])}”；
            第四张表情包内的字幕为“{captions[4]}”，字幕的中文拼音为“{py(captions[4])}”；
            第五张表情包内的字幕为“{captions[5]}”，字幕的中文拼音为“{py(captions[5])}”；
            第六张表情包内的字幕为“{captions[6]}”，字幕的中文拼音为“{py(captions[6])}”；
            第七张表情包内的字幕为“{captions[7]}”，字幕的中文拼音为“{py(captions[7])}”；
            第八张表情包内的字幕为“{captions[8]}”，字幕的中文拼音为“{py(captions[8])}”。
            以下是甲、乙两人的两次网络聊天记录，都缺少了最后的表情包回复。要求第一次聊天只能从前四张表情包中进行选择，第二次聊天只能从后四张表情包中进行选择。请你根据聊天内容，人物关系和情绪氛围，从限定的四张表情包中选出最合适的一个作为回复。
            【聊天记录一】：\n{chat_history1}
            【选项】：\n{choices1}\n【正确答案】：{answer1}\n
            请你参考聊天记录一的例子，选出你认为的聊天记录二的正确答案。仅需回答代表选项的字母（A-D），不要输出其他内容。
            【聊天记录二】：\n{chat_history2}
            【选项】：\n{choices2}\n\n你的回答：
            """)
        # 6. Image & caption, few-shot + CoT
        elif prompt_mode == 6:
            instruction = lambda captions, chat_history1, choices1, analysis1, answer1, chat_history2, choices2: textwrap.dedent(f"""\
            示例和任务描述：\n给定的八张表情包图片都是满足以上定义的双关表情包，现将这八张表情包按顺序分别记为第一、二、三、四、五、六、七、八张表情包，其中：
            第一张表情包内的字幕为“{captions[1]}”；\n第二张表情包内的字幕为“{captions[2]}”；
            第三张表情包内的字幕为“{captions[3]}”；\n第四张表情包内的字幕为“{captions[4]}”；
            第五张表情包内的字幕为“{captions[5]}”；\n第六张表情包内的字幕为“{captions[6]}”；
            第七张表情包内的字幕为“{captions[7]}”；\n第八张表情包内的字幕为“{captions[8]}”。
            以下是甲、乙两人的两次网络聊天记录，都缺少了最后的表情包回复。要求第一次聊天只能从前四张表情包中进行选择，第二次聊天只能从后四张表情包中进行选择。请你根据聊天内容，人物关系和情绪氛围，从限定的四张表情包中选出最合适的一个作为回复。
            【聊天记录一】：\n{chat_history1}
            【选项】：\n{choices1}\n【表情包分析及正确答案】：\n{analysis1}\n正确答案：{answer1}\n
            请你参考聊天记录一的例子，选出你认为的聊天记录二的正确答案。要求严格按照上面的输出格式，先对选项中的表情包进行分析，再以“正确答案：X”的格式给出代表选项的字母（A-D），不要输出其他内容。
            【聊天记录二】：\n{chat_history2}
            【选项】：\n{choices2}\n\n你的回答：
            """)
        # 7. Image & caption (with Chinese Pinyin), few-shot + CoT
        else:
            instruction = lambda captions, chat_history1, choices1, analysis1, answer1, chat_history2, choices2: textwrap.dedent(f"""\
            示例和任务描述：\n给定的八张表情包图片都是满足以上定义的双关表情包，现将这八张表情包按顺序分别记为第一、二、三、四、五、六、七、八张表情包，其中：
            第一张表情包内的字幕为“{captions[1]}”，字幕的中文拼音为“{py(captions[1])}”；
            第二张表情包内的字幕为“{captions[2]}”，字幕的中文拼音为“{py(captions[2])}”；
            第三张表情包内的字幕为“{captions[3]}”，字幕的中文拼音为“{py(captions[3])}”；
            第四张表情包内的字幕为“{captions[4]}”，字幕的中文拼音为“{py(captions[4])}”；
            第五张表情包内的字幕为“{captions[5]}”，字幕的中文拼音为“{py(captions[5])}”；
            第六张表情包内的字幕为“{captions[6]}”，字幕的中文拼音为“{py(captions[6])}”；
            第七张表情包内的字幕为“{captions[7]}”，字幕的中文拼音为“{py(captions[7])}”；
            第八张表情包内的字幕为“{captions[8]}”，字幕的中文拼音为“{py(captions[8])}”。
            以下是甲、乙两人的两次网络聊天记录，都缺少了最后的表情包回复。要求第一次聊天只能从前四张表情包中进行选择，第二次聊天只能从后四张表情包中进行选择。请你根据聊天内容，人物关系和情绪氛围，从限定的四张表情包中选出最合适的一个作为回复。
            【聊天记录一】：\n{chat_history1}
            【选项】：\n{choices1}\n【表情包分析及正确答案】：\n{analysis1}\n正确答案：{answer1}\n
            请你参考聊天记录一的例子，选出你认为的聊天记录二的正确答案。要求严格按照上面的输出格式，先对选项中的表情包进行分析，再以“正确答案：X”的格式给出代表选项的字母（A-D），不要输出其他内容。
            【聊天记录二】：\n{chat_history2}
            【选项】：\n{choices2}\n\n你的回答：
            """)
    else:
        pun_meme_definition = textwrap.dedent("""\
        Definition:\nFor memes consisting of images and captions, if the caption intentionally uses polysemy or homophony to create two or more meanings, with at least one meaning based on the visual elements of the image, the meme is called a "pun meme". Otherwise, it is called a "non-pun meme".
        """)
        # 1. Only image, zero-shot
        if prompt_mode == 1:
            instruction = lambda chat_history, choices: textwrap.dedent(f"""\
            Task Description:\nThe four given meme images are pun memes that meet the above definition. Here we label these four memes in order as the first, second, third, and fourth meme. Below is an online chat record between two people, '甲' and '乙', which lacks a final meme reply. Please choose the most suitable meme from the four as a reply, based on the chat content, character relationship, and emotional tone. Simply respond with the letter (A-D) representing your choice. Do not output any other content.
            [Chat Record]:\n{chat_history}
            [Options]:\n{choices}\n\nYour Response:
            """)
        # 2. Image & caption, zero-shot
        elif prompt_mode == 2:
            instruction = lambda captions, chat_history, choices: textwrap.dedent(f"""\
            Task Description:\nThe four given meme images are pun memes that meet the above definition. Here we label these four memes in order as the first, second, third, and fourth meme. From these memes, we can know that:
            The caption within the first meme is "{captions[1]}";
            The caption within the second meme is "{captions[2]}";
            The caption within the third meme is "{captions[3]}";
            The caption within the fourth meme is "{captions[4]}".
            Below is an online chat record between two people, '甲' and '乙', which lacks a final meme reply. Please choose the most suitable meme from the four as a reply, based on the chat content, character relationship, and emotional tone. Simply respond with the letter (A-D) representing your choice. Do not output any other content.
            [Chat Record]:\n{chat_history}
            [Options]:\n{choices}\n\nYour Response:
            """)
        # 3. Only image, few-shot
        elif prompt_mode == 3:
            instruction = lambda chat_history1, choices1, answer1, chat_history2, choices2: textwrap.dedent(f"""\
            Example and Task Description:\nThe eight given meme images are pun memes that meet the above definition. Here we label these eight memes in order as the first, second, third, fourth, fifth, sixth, seventh, and eighth meme. Below are two online chat records between two people, '甲' and '乙', both lacking a final meme reply. For the first chat, the meme reply can only be chosen from the first four memes, and for the second chat, the meme reply can only be chosen from the last four memes. Please choose the most suitable meme from the specified four as a reply, based on the chat content, character relationship, and emotional tone.
            [Chat Record 1]:\n{chat_history1}
            [Options]:\n{choices1}\n[Correct Answer]: {answer1}\n
            Please refer to the example of Chat Record 1 and choose the correct answer you think is for Chat Record 2. Simply respond with the letter (A-D) representing your choice. Do not output any other content.
            [Chat Record 2]:\n{chat_history2}
            [Options]:\n{choices2}\n\nYour Response:
            """)
        # 4. Image & caption, few-shot
        elif prompt_mode == 4:
            instruction = lambda captions, chat_history1, choices1, answer1, chat_history2, choices2: textwrap.dedent(f"""\
            Example and Task Description:\nThe eight given meme images are pun memes that meet the above definition. Here we label these eight memes in order as the first, second, third, fourth, fifth, sixth, seventh, and eighth meme. From these memes, we can know that:
            The caption within the first meme is "{captions[1]}";\nThe caption within the second meme is "{captions[2]}";
            The caption within the third meme is "{captions[3]}";\nThe caption within the fourth meme is "{captions[4]}";
            The caption within the fifth meme is "{captions[5]}";\nThe caption within the sixth meme is "{captions[6]}";
            The caption within the seventh meme is "{captions[7]}";\nThe caption within the eighth meme is "{captions[8]}".
            Below are two online chat records between two people, '甲' and '乙', both lacking a final meme reply. For the first chat, the meme reply can only be chosen from the first four memes, and for the second chat, the meme reply can only be chosen from the last four memes. Please choose the most suitable meme from the specified four as a reply, based on the chat content, character relationship, and emotional tone.
            [Chat Record 1]:\n{chat_history1}
            [Options]:\n{choices1}\n[Correct Answer]: {answer1}\n
            Please refer to the example of Chat Record 1 and choose the correct answer you think is for Chat Record 2. Simply respond with the letter (A-D) representing your choice. Do not output any other content.
            [Chat Record 2]:\n{chat_history2}
            [Options]:\n{choices2}\n\nYour Response:
            """)
        # 5. Image & caption (with Chinese Pinyin), few-shot
        elif prompt_mode == 5:
            instruction = lambda captions, chat_history1, choices1, answer1, chat_history2, choices2: textwrap.dedent(f"""\
            Example and Task Description:\nThe eight given meme images are pun memes that meet the above definition. Here we label these eight memes in order as the first, second, third, fourth, fifth, sixth, seventh, and eighth meme. From these memes, we can know that:
            The caption within the first meme is "{captions[1]}", and the Chinese pinyin of the caption is "{py(captions[1])}";
            The caption within the second meme is "{captions[2]}", and the Chinese pinyin of the caption is "{py(captions[2])};
            The caption within the third meme is "{captions[3]}", and the Chinese pinyin of the caption is "{py(captions[3])};
            The caption within the fourth meme is "{captions[4]}", and the Chinese pinyin of the caption is "{py(captions[4])};
            The caption within the fifth meme is "{captions[5]}", and the Chinese pinyin of the caption is "{py(captions[5])};
            The caption within the sixth meme is "{captions[6]}", and the Chinese pinyin of the caption is "{py(captions[6])};
            The caption within the seventh meme is "{captions[7]}", and the Chinese pinyin of the caption is "{py(captions[7])};
            The caption within the eighth meme is "{captions[8]}", and the Chinese pinyin of the caption is "{py(captions[8])}.
            Below are two online chat records between two people, '甲' and '乙', both lacking a final meme reply. For the first chat, the meme reply can only be chosen from the first four memes, and for the second chat, the meme reply can only be chosen from the last four memes. Please choose the most suitable meme from the specified four as a reply, based on the chat content, character relationship, and emotional tone.
            [Chat Record 1]:\n{chat_history1}
            [Options]:\n{choices1}\n[Correct Answer]: {answer1}\n
            Please refer to the example of Chat Record 1 and choose the correct answer you think is for Chat Record 2. Simply respond with the letter (A-D) representing your choice. Do not output any other content.
            [Chat Record 2]:\n{chat_history2}
            [Options]:\n{choices2}\n\nYour Response:
            """)
        # 6. Image & caption, few-shot + CoT
        elif prompt_mode == 6:
            instruction = lambda captions, chat_history1, choices1, analysis1, answer1, chat_history2, choices2: textwrap.dedent(f"""\
            Example and Task Description:\nThe eight given meme images are pun memes that meet the above definition. Here we label these eight memes in order as the first, second, third, fourth, fifth, sixth, seventh, and eighth meme. From these memes, we can know that:
            The caption within the first meme is "{captions[1]}";\nThe caption within the second meme is "{captions[2]}";
            The caption within the third meme is "{captions[3]}";\nThe caption within the fourth meme is "{captions[4]}";
            The caption within the fifth meme is "{captions[5]}";\nThe caption within the sixth meme is "{captions[6]}";
            The caption within the seventh meme is "{captions[7]}";\nThe caption within the eighth meme is "{captions[8]}".
            Below are two online chat records between two people, '甲' and '乙', both lacking a final meme reply. For the first chat, the meme reply can only be chosen from the first four memes, and for the second chat, the meme reply can only be chosen from the last four memes. Please choose the most suitable meme from the specified four as a reply, based on the chat content, character relationship, and emotional tone.
            [Chat Record 1]:\n{chat_history1}
            [Options]:\n{choices1}\n[Meme Analysis and Correct Answer]:\n{analysis1}\nCorrect Answer: {answer1}\n
            Please refer to the example of Chat Record 1 and choose the correct answer you think is for Chat Record 2. Strictly follow the output format above: first analyze the memes in the options, then provide the correct answer in the format "Correct Answer: X" using the corresponding letter (A-D). Do not output any other content.
            [Chat Record 2]:\n{chat_history2}
            [Options]:\n{choices2}\n\nYour Response:
            """)
        # 7. Image & caption (with Chinese Pinyin), few-shot + CoT
        else:
            instruction = lambda captions, chat_history1, choices1, analysis1, answer1, chat_history2, choices2: textwrap.dedent(f"""\
            Example and Task Description:\nThe eight given meme images are pun memes that meet the above definition. Here we label these eight memes in order as the first, second, third, fourth, fifth, sixth, seventh, and eighth meme. From these memes, we can know that:
            The caption within the first meme is "{captions[1]}", and the Chinese pinyin of the caption is "{py(captions[1])}";
            The caption within the second meme is "{captions[2]}", and the Chinese pinyin of the caption is "{py(captions[2])};
            The caption within the third meme is "{captions[3]}", and the Chinese pinyin of the caption is "{py(captions[3])};
            The caption within the fourth meme is "{captions[4]}", and the Chinese pinyin of the caption is "{py(captions[4])};
            The caption within the fifth meme is "{captions[5]}", and the Chinese pinyin of the caption is "{py(captions[5])};
            The caption within the sixth meme is "{captions[6]}", and the Chinese pinyin of the caption is "{py(captions[6])};
            The caption within the seventh meme is "{captions[7]}", and the Chinese pinyin of the caption is "{py(captions[7])};
            The caption within the eighth meme is "{captions[8]}", and the Chinese pinyin of the caption is "{py(captions[8])}.
            Below are two online chat records between two people, '甲' and '乙', both lacking a final meme reply. For the first chat, the meme reply can only be chosen from the first four memes, and for the second chat, the meme reply can only be chosen from the last four memes. Please choose the most suitable meme from the specified four as a reply, based on the chat content, character relationship, and emotional tone.
            [Chat Record 1]:\n{chat_history1}
            [Options]:\n{choices1}\n[Meme Analysis and Correct Answer]:\n{analysis1}\nCorrect Answer: {answer1}\n
            Please refer to the example of Chat Record 1 and choose the correct answer you think is for Chat Record 2. Strictly follow the output format above: first analyze the memes in the options, then provide the correct answer in the format "Correct Answer: X" using the corresponding letter (A-D). Do not output any other content.
            [Chat Record 2]:\n{chat_history2}
            [Options]:\n{choices2}\n\nYour Response:
            """)

    # Call VLM to Response
    random.seed(2025)
    random.shuffle(dataset); random.shuffle(dataset)
    for chat_id in tqdm(dataset, desc='1.3 Text-Driven Meme Response'):
        # Skip the data that has already been judged
        if record[difficulty][model][language][mode[prompt_mode]][chat_id]['model_output'] is not None:
            continue
        if 'zero_shot' in mode[prompt_mode]:
            images, captions, chat_history, choices, answer = get_chat_data(chat_id=chat_id, first_half=True, language=language)
        else:
            images, captions, chat_history, choices, answer = get_chat_data(chat_id=chat_id, first_half=False, language=language)
        choices_string = '\n'.join(list(choices.values()))
        # 1. Only image, zero-shot
        if prompt_mode == 1:
            task_prompt = pun_meme_definition + '\n' + instruction(chat_history, choices_string)
            memes = images
        # 2. Image & caption, zero-shot
        elif prompt_mode == 2:
            task_prompt = pun_meme_definition + '\n' + instruction(captions, chat_history, choices_string)
            memes = images
        # 3. Only image, few-shot
        elif prompt_mode == 3:
            task_prompt = pun_meme_definition + '\n' + instruction(example_chat_history, example_choices_string, example_answer, chat_history, choices_string)
            memes = example_images + images
        # 4. Image & caption, few-shot
        # 5. Image & caption (with Chinese Pinyin), few-shot
        elif prompt_mode == 4 or prompt_mode == 5:
            task_prompt = pun_meme_definition + '\n' + instruction(example_captions+captions[1:], example_chat_history, example_choices_string, example_answer, chat_history, choices_string)
            memes = example_images + images
        # 6. Image & caption, few-shot + CoT
        # 7. Image & caption (with Chinese Pinyin), few-shot + CoT
        else:
            task_prompt = pun_meme_definition + '\n' + instruction(example_captions+captions[1:], example_chat_history, example_choices_string, example_analysis, example_answer, chat_history, choices_string)
            memes = example_images + images
        # Remove extra spaces before lines
        task_prompt = '\n'.join([line.strip() for line in task_prompt.split('\n')]).strip('\n')
        # Call VLM
        output = call[model](client=client, image=memes, text_prompt=task_prompt, temperature=temperature)
        # Print image, prompt and model output
        if not save:
            for image in memes:
                plt.figure(); plt.imshow(image); plt.show()
            print(task_prompt); print('*'*60); print('ModelOutput:', output); print('*'*60); print('CorrectAnswer:', answer)
            break
        # Record
        tags = [choice.split(':')[0].strip() for choice in choices.values()]
        if 'CoT' in mode[prompt_mode]:
            temp = output.split('正确答案：')[-1].strip() if language=='cn' else output.split('Correct Answer:')[-1].strip()
        else:
            temp = output.strip()
        judgment = temp.split(':')[0].strip() if temp.split(':')[0].strip() in tags else None  # Rely on subsequent processing
        record[difficulty][model][language][mode[prompt_mode]][chat_id] = {
            'choices': choices,
            'model_output': output,
            'model_judgment': judgment}
        # Save
        if save:
            save_json_file(record, save_path)


def build_OCR_dataset(meta_path:str, nums:int=200):
    """
    Select pun meme data for caption OCR
    """
    meta = load_json_file(meta_path)
    indices = list(range(1, 654))
    random.seed(2025)
    random.shuffle(indices); random.shuffle(indices)
    imgs = [f'CN{ind:05d}.png' for ind in indices[:nums]]
    OCR_dataset = {'img_folder': os.path.dirname(meta_path),
                   'instances': dict()}
    for img in imgs:
        OCR_dataset['instances'][img] = {'true_caption': meta[img]['caption_1']}
    return OCR_dataset


def vlm_caption_pinyin(dataset, model:str, client, language:str, one_shot:bool, temperature:float=0.0, debug:bool=False, save:bool=True):
    """
    Testing VLMs' performance on converting Chinese captions into pinyin
    """
    assert language in ['cn', 'en']
    call = {'gpt-4o-2024-08-06': call_gpt4o,
            'gpt-4o-mini-2024-07-18': call_gpt4o_mini,
            'claude-3-5-sonnet-20241022': call_claude35sonnet,
#            'claude-3-5-haiku-20241022': call_claude35haiku,
            'qwen2.5-vl-72b-instruct': call_qwen2_vl,
            'qwen2.5-vl-7b-instruct': call_qwen2_vl,
            'intern-vl2_5-78b': call_intern_vl2,
            'intern-vl2_5-38b': call_intern_vl2,
            'intern-vl2_5-26b': call_intern_vl2,
            'intern-vl2_5-8b': call_intern_vl2,
#            'llava-onevision-qwen2-72b-ov-chat': call_llava_onevision,
#            'llava-onevision-qwen2-7b-ov': call_llava_onevision,
            'minicpm-v-2_6': call_minicpm,
            'mantis-8b-siglip-llama3': call_mantis_siglip,
            'idefics3-8b-llama3': call_idefics}
    assert model in list(call.keys())
    img_folder = dataset['img_folder']
    instances = dataset['instances']

    keep_chinese = lambda text: converter.convert(''.join([char for char in text if '\u4e00' <= char <= '\u9fff']))
    py = lambda text: ' '.join(chinese_character_to_pinyin(keep_chinese(text)))
    # Examples
    # example_image = Image.open('./data/other/OCR_example.png')
    example_caption = '堪察加帝王蟹，我们走'
    # Save Path
    path = f'./results/other_eval/caption_pinyin.json'
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    prompt_type = 'zero-shot' if not one_shot else 'one-shot'
    record = extend_dict(load_json_file(path), [model, language, prompt_type])
    # Task Prompt
    if not one_shot:  # zero-shot
        if language == 'cn':
            task_prompt = lambda caption: textwrap.dedent(f"""\
            拼音是一种辅助汉字读音的工具。请将以下中文文本转化成对应的带有声调的拼音。不要输出其他内容。
            中文文本：{caption} -> 拼音：
            """).strip('\n')
        else:
            task_prompt = lambda caption: textwrap.dedent(f"""\
            Pinyin is a tool that aids in the pronunciation of Chinese characters. Please convert the following Chinese text into corresponding pinyin with tones. Do not output any other content.
            Chinese text: {caption} -> Pinyin:
            """).strip('\n')
    else:  # one-shot
        if language == 'cn':
            task_prompt = lambda caption: textwrap.dedent(f"""\
            拼音是一种辅助汉字读音的工具。请将以下中文文本转化成对应的带有声调的拼音。不要输出其他内容。
            中文文本：{keep_chinese(example_caption)} -> 拼音：{py(example_caption)}
            中文文本：{caption} -> 拼音：
            """).strip('\n')
        else:
            task_prompt = lambda caption: textwrap.dedent(f"""\
            Pinyin is a tool that aids in the pronunciation of Chinese characters. Please convert the following Chinese text into corresponding pinyin with tones. Do not output any other content.
            Chinese text: {keep_chinese(example_caption)} -> Pinyin: {py(example_caption)}
            Chinese text: {caption} -> Pinyin:
            """).strip('\n')

    for img in tqdm(instances, desc='2.1 Caption OCR & Pinyin'):
        # Skip data that has already been tested
        if img in record[model][language][prompt_type]:
            continue
        # Data to be tested
        true_caption = instances[img]['true_caption']
        # Call VLM
        text_prompt = task_prompt(caption=keep_chinese(true_caption))
        output = call[model](client=client, image=None, text_prompt=text_prompt, temperature=temperature)
        if debug:
            print(task_prompt); print('*'*60)
            print(output); print('*'*60)
            break
        # Record
        record[model][language][prompt_type][img] = {'pred_pinyin': output.split('->')[-1].split('拼音：')[-1].split('Pinyin:')[-1],
                                                     'true_pinyin': py(true_caption)}
        # Save
        if save:
            save_json_file(record, path)


def vlm_image_OCR(dataset, model:str, client, language:str, one_shot:bool, temperature:float=0.0, debug:bool=False, save:bool=True):
    """
    Testing VLMs' performance on image caption OCR
    """
    assert language in ['cn', 'en']
    call = {'gpt-4o-2024-08-06': call_gpt4o,
            'gpt-4o-mini-2024-07-18': call_gpt4o_mini,
            'claude-3-5-sonnet-20241022': call_claude35sonnet,
            'qwen2.5-vl-72b-instruct': call_qwen2_vl,
            'qwen2.5-vl-7b-instruct': call_qwen2_vl,
            'intern-vl2_5-78b': call_intern_vl2,
            'intern-vl2_5-38b': call_intern_vl2,
            'intern-vl2_5-26b': call_intern_vl2,
            'intern-vl2_5-8b': call_intern_vl2,
#            'llava-onevision-qwen2-72b-ov-chat': call_llava_onevision,
#            'llava-onevision-qwen2-7b-ov': call_llava_onevision,
            'minicpm-v-2_6': call_minicpm,
            'mantis-8b-siglip-llama3': call_mantis_siglip,
            'idefics3-8b-llama3': call_idefics}
    assert model in list(call.keys())
    img_folder = dataset['img_folder']
    instances = dataset['instances']

    # Examples
    example_image = Image.open('./data/other/OCR_example.png')
    example_caption = '堪察加帝王蟹，我们走'
    # Save Path
    path = f'./results/other_eval/image_OCR.json'
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    prompt_type = 'zero-shot' if not one_shot else 'one-shot'
    record = extend_dict(load_json_file(path), [model, language, prompt_type])
    # Task Prompt
    if not one_shot:  # zero-shot
        if language == 'cn':
            task_prompt = textwrap.dedent("""\
            以上是一张网络聊天用的中文表情包，请提取出该表情包中的主要字幕。不要输出其他内容。
            表情包中的字幕：
            """).strip('\n')
        else:
            task_prompt = textwrap.dedent("""\
            Here is a Chinese meme used in online chat. Please extract the main caption/subtitle from the meme. Do not output any other content.
            Meme caption/subtitle:
            """).strip('\n')
    else:  # one-shot
        if language == 'cn':
            task_prompt = textwrap.dedent(f"""\
            以上是两张网络聊天用的中文表情包。根据图像可知第一张表情包中的字幕为“{example_caption}”。请提取出第二张表情包中的主要字幕，不要输出其他内容。
            第二张表情包中的字幕：
            """).strip('\n')
        else:
            task_prompt = textwrap.dedent(f"""\
            Here are two Chinese memes used in online chat. From the meme images, we can know that the caption/subtitle on the first meme is "{example_caption}". Please extract the main caption/subtitle from the second meme, without outputting any other content.
            The caption/subtitle on the second meme:
            """).strip('\n')

    for img in tqdm(instances, desc='2.1 Caption OCR & Pinyin'):
        # Skip data that has already been tested
        if img in record[model][language][prompt_type]:
            continue
        # Data to be tested
        if not one_shot:  # zero-shot
            images = Image.open(os.path.join(img_folder, img))
        else:  # one-shot
            images = [example_image, Image.open(os.path.join(img_folder, img))]
        true_caption = instances[img]['true_caption']
        # Call VLM
        output = call[model](client=client, image=images, text_prompt=task_prompt, temperature=temperature)
        if debug:
            if type(images) == list:
                for image in images:
                    plt.figure(); plt.imshow(image); plt.show()
            else:
                plt.figure(); plt.imshow(images); plt.show()
            print(task_prompt); print('*'*60)
            print(output); print('*'*60)
            break
        # Record
        record[model][language][prompt_type][img] = {'OCR_caption': output,
                                                     'true_caption': true_caption}
        # Save
        if save:
            save_json_file(record, path)



def load_dataset(data_type:str):
    """
    Load different types of harmful Chinese memes
    - ToxiCN_MM_harmful_meme
    - nonpun_harmful_meme
    - pun_harmful_meme
    """
    assert data_type in ['ToxiCN_MM_train', 'ToxiCN_MM_harmful_meme', 'nonpun_harmful_meme', 'pun_harmful_meme']
    dataset = dict()
    if data_type == 'ToxiCN_MM_train':
        img_folder = './data/other/harmful_meme/ToxiCN_MM/train'
        dataset['img_folder'] = img_folder
        dataset['instances'] = dict()
        descriptions = load_json_file(os.path.join(img_folder, '_description.json'))
    elif data_type == 'ToxiCN_MM_harmful_meme':
        img_folder = './data/other/harmful_meme/ToxiCN_MM/test_448png'
        dataset['img_folder'] = img_folder
        dataset['instances'] = dict()
        descriptions = load_json_file(os.path.join(img_folder, '_description.json'))
    elif data_type == 'nonpun_harmful_meme':
        img_folder = './data/other/harmful_meme/our_meme'
        dataset['img_folder'] = img_folder
        dataset['instances'] = dict()
        descriptions = load_json_file(os.path.join(img_folder, '_description.json'))
        descriptions = [desc for desc in descriptions if desc['meme_type'] == 'non-pun']
    else:
        img_folder = './data/other/harmful_meme/our_meme'
        dataset['img_folder'] = img_folder
        dataset['instances'] = dict()
        descriptions = load_json_file(os.path.join(img_folder, '_description.json'))
        descriptions = [desc for desc in descriptions if desc['meme_type'] == 'pun']
    dataset['instances'] = descriptions
    print(f'Dataset: {data_type}, {len(descriptions)}')
    return dataset


def vlm_harmful_meme_detection(dataset_type:str, model:str, client, language:str, temperature:float=0.0, debug:bool=False, save:bool=True):
    """
    Testing VLMs' performance on harmful meme detection. There are three datasets:
    - ToxiCN_MM_harmful_meme
    - nonpun_harmful_meme
    - pun_harmful_meme
    """
    assert language in ['cn', 'en']
    call = {'gpt-4o-2024-08-06': call_gpt4o,
            'claude-3-5-sonnet-20241022': call_claude35sonnet}
    assert model in list(call.keys())
    assert dataset_type in ['ToxiCN_MM_harmful_meme', 'nonpun_harmful_meme', 'pun_harmful_meme']

    # Load Dataset
    dataset = load_dataset(data_type=dataset_type)
    img_folder = dataset['img_folder']
    instances = dataset['instances']
    # Save Path
    path = f'./results/other_eval/harmful_meme_detection.json'
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    record = extend_dict(load_json_file(path), [model, dataset_type, language])
    # Task Prompt
    if language == 'cn':  # Chinese Prompt
        task_prompt = lambda caption: textwrap.dedent(f"""\
        中文有害表情包的定义:
        中文有害表情包是由图像和图像内字幕组成的多模态单元，具有对个人、组织、社区、社会群体等造成伤害的潜力。它们往往反映和强化了中国互联网上潜在的负面价值观和文化态度。

        中文有害表情包检测：
        以上是一张网络聊天用的中文表情包，其中包含字幕“{caption}”。请根据中文有害表情包的定义，将其分类为有害或无害。符合以下任何一个标准的表情包都将被判定为有害。
        1. 表情包表达了对特定个人或社会群体的厌恶、偏见或刻板印象。
        2. 表情包包含了讽刺或粗鲁的内容，没有特定的目标。
        3. 表情包暗示了性意图，以引起性冲动。
        4. 表情包包含了颓废和绝望的情绪，或传达了一种自我否定的态度。

        评估步骤:
        1. 结合表情包图像和字幕分析表情包的含义；
        2. 与给定的评估标准进行比较，检查是否有任何匹配；
        3. 返回“有害”或“无害”的结论。

        要求必须以JSON格式输出三个评估步骤，形式为{{"表情包含义": "XXXX", "匹配评估标准": "XXXX", "结论": "有害/无害"}}
        """).strip('\n')
    else:  # English Prompt
        task_prompt = lambda caption: textwrap.dedent(f"""\
        Definition of Chinese Harmful Memes:
        Chinese harmful memes are multimodal units consisting of an image and inline caption that have the potential to cause harm to individuals, organizations, communities, or social groups. They often reflect and reinforce underlying negative values and cultural attitudes on the Chinese Internet.

        Detection of Chinese Harmful Memes:
        The image above is a Chinese meme used in online chat. The caption in the meme is "{caption}". Based on the definition of Chinese harmful memes, please classify it as harmful or harmless. A meme will be deemed harmful if it meets any of the following criteria:
        1. The meme expresses disgust, prejudice, or stereotypes towards specific individuals or social groups.
        2. The meme contains sarcastic or rude content without a specific target.
        3. The meme implies sexual intent to arouse sexual impulses.
        4. The meme integrates decadent and desperate emotions or conveys a self-negative attitude.

        Evaluation Steps:
        1. Analyze the meaning of the meme by considering both the image and the caption.
        2. Compare the meme against the criteria to check for any matches;
        3. Return the conclusion as "harmful" or "harmless".

        The output must be in JSON format for the three evaluation steps, structured as {{"Meme_Meaning": "XXXX", "Criteria_Matching": "XXXX", "Conclusion": "harmful/harmless"}}
        """).strip('\n')

    for instance in tqdm(instances, desc='2.3 Harmful Meme Detection'):
        img = instance['new_path']
        # Skip data that has already been tested
        if img in record[model][dataset_type][language]:
            continue
        # Data to be tested
        image = Image.open(os.path.join(img_folder, img))
        caption = instance['text']
        caption = caption.replace('\n', '')
        # Call VLM
        text_prompt = task_prompt(caption=caption)
        output = call[model](client=client, image=image, text_prompt=text_prompt, temperature=temperature)
        if debug:
            plt.figure(); plt.imshow(image); plt.show()
            print(text_prompt); print('*'*60)
            print(output); print('*'*60)
            break
        # Record
        try:
            output = eval(output[output.index('{'): output.index('}')+1].replace('\n', ' '))
            meme_meaning = output['Meme_Meaning'] if language == 'en' else output['表情包含义']
            criteria_matching = output['Criteria_Matching'] if language == 'en' else output['匹配评估标准']
            conclusion = output['Conclusion'] if language == 'en' else output['结论']
            model_pred = 1 if (conclusion.lower() == '有害' or conclusion.lower() == 'harmful') else 0
            true_label = instance['label']
        except:
            meme_meaning = output
            criteria_matching = ''
            conclusion = ''
            if meme_meaning[:12] == 'I apologize,':
                model_pred = 1  # Claude refuses to answer because the input is harmful
            else:
                model_pred = None
            true_label = instance['label']
        record[model][dataset_type][language][img] = \
            {'meme_meaning': meme_meaning, 'criteria_matching': criteria_matching, 'conclusion': conclusion,
             'model_pred': model_pred, 'true_label': true_label}
        # Save
        if save:
            save_json_file(record, path)
