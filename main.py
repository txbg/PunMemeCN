# -*- coding:utf-8 -*-
import argparse
from utils.eval.task import *


if __name__ == "__main__":
    main_models = ['gpt-4o-2024-08-06', 'gpt-4o-mini-2024-07-18',
                   'claude-3-5-sonnet-20241022',
                   'qwen2.5-vl-72b-instruct', 'qwen2.5-vl-7b-instruct',
                   'intern-vl2_5-78b', 'intern-vl2_5-38b', 'intern-vl2_5-26b', 'intern-vl2_5-8b',
                   'llava-onevision-qwen2-72b-ov-chat', 'llava-onevision-qwen2-7b-ov',
                   'minicpm-v-2_6']
    all_models = main_models + ['mantis-8b-siglip-llama3', 'idefics3-8b-llama3']
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, help='model',
                        choices=all_models)
    # Receive parameters
    args = parser.parse_args()
    model = args.model
    assert model in all_models
    client = obtain_client(model=model)

    # 1. Main Experiments
    if model in main_models:
        if 'intern-vl' in model:
            languages, prompt_types = ['cn'], [1, 2, 3, 4, 5, 6, 7]
        else:
            languages, prompt_types = ['cn'], [1, 2]
        dataset_path = './data/pun_meme/cn'
        '''
        # 1.1. Pun Meme Detection
        for language in languages:
            for prompt_type in prompt_types:
                vlm_pun_meme_detection(dataset_path, model=model, client=client, language=language,
                                       prompt_mode=prompt_type, pilot=False, save=True)
        '''
        # 1.2. Pun Meme Sentiment Analysis
        for language in languages:
            for prompt_type in prompt_types:
                vlm_pun_meme_sentiment_analysis(dataset_path, model=model, client=client, language=language,
                                                prompt_mode=prompt_type, pilot=False, save=True)
        # 1.3. Text-Driven Meme Response
        for language in languages:  # Hard
            for prompt_type in prompt_types:
                vlm_chat_driven_meme_response(dataset_path, difficulty='hard', model=model, client=client,
                                              language=language, prompt_mode=prompt_type, pilot=False, save=True)
        for language in ['cn']:  # Easy
            for prompt_type in [1, 2]:
                vlm_chat_driven_meme_response(dataset_path, difficulty='easy', model=model, client=client,
                                              language=language, prompt_mode=prompt_type, pilot=False, save=True)

    # 2. Additional experiments
    if model in all_models:
        # 2.1. Caption OCR & Pinyin
        meta_path, nums = './data/pun_meme/cn/_Meta.json', 200
        OCR_dataset = build_OCR_dataset(meta_path=meta_path, nums=nums)
        for language in ['cn']:
            for one_shot in [False]:
                vlm_caption_pinyin(dataset=OCR_dataset, model=model, client=client, language=language, one_shot=one_shot)
                vlm_image_OCR(dataset=OCR_dataset, model=model, client=client, language=language, one_shot=one_shot)
        # 2.2. Harmful Meme Detection
        if model in ['gpt-4o-2024-08-06', 'claude-3-5-sonnet-20241022']:
            for dataset_type in ['ToxiCN_MM_harmful_meme', 'nonpun_harmful_meme', 'pun_harmful_meme']:
                for language in ['cn']:
                    vlm_harmful_meme_detection(dataset_type=dataset_type, model=model, client=client, language=language)
    # End of Experiments
    print('[Info]: Evaluation of current VLM is Over!!!')

