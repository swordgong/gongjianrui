import os, gc
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from omegaconf import OmegaConf
import argparse
import requests
import random
import hashlib
import json
#大模型过滤
import re
import spacy
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models
#clip相似度计算依赖库
from PIL import Image
from PIL import ImageFilter
from transformers import CLIPModel, CLIPProcessor


#过滤类
class DescriptionCompressor:
    def __init__(self, keyword_map_file="keyword_rules.txt"):
        # 加载关键词映射规则（格式：关键词|同义词1,同义词2|描述特征1,特征2）
        self.keyword_rules = self._load_keyword_rules(keyword_map_file)
        self.client = self._init_hunyuan_client()
        self.model = "hunyuan-lite"
        self.nlp = spacy.load("zh_core_web_sm")  # 加载spaCy模型
        print("中文模型加载成功")
        
    def _init_hunyuan_client(self):
        """初始化混元客户端"""
        cred = credential.Credential("AKID76tSoxsZ3LTO6", "iHpETdcQI0Wr0xhDW")
        http_profile = HttpProfile(endpoint="hunyuan.tencentcloudapi.com")
        client_profile = ClientProfile(httpProfile=http_profile)
        return hunyuan_client.HunyuanClient(cred, "ap-guangzhou", client_profile)

    def _load_keyword_rules(self, filepath):
        """加载关键词映射规则"""
        rules = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('|')
                    keyword = parts[0]
                    synonyms = parts[1].split(',') if len(parts) > 1 else []
                    features = parts[2].split(',') if len(parts) > 2 else []
                    rules[keyword] = {
                        'synonyms': synonyms,
                        'features': features
                    }
        return rules

    def _is_description(self, text):
        """判断是否为描述性文本"""
        req = models.ChatCompletionsRequest()
        req.Model = self.model
        req.Messages = [{
            "Role": "user",
            "Content": f"判断以下文本是否包含物体描述（只需回答是/否）:\n{text}"
        }]
        try:
            resp = self.client.ChatCompletions(req)
            return resp.Choices[0].Message.Content.strip() == '是'
        except Exception as e:
            print(f"API调用失败: {e}")
            return False

    def _extract_keyword(self, description):
        """从描述中提取核心关键词"""
        # 优先检查规则库
        for keyword, rule in self.keyword_rules.items():
            for feature in rule['features']:
                if feature in description:
                    return keyword

        # 使用NLP工具分析句子结构
        doc = self.nlp(description)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "OBJECT"]:
                return ent.text

        # 使用大模型提取
        req = models.ChatCompletionsRequest()
        req.Model = self.model
        req.Messages = [{
            "Role": "user",
            "Content": f"结合下列所有文本，判断或者推测以下外貌描述最有可能是什么物体（只需返回单个名词）:\n{description}"
        }]
        try:
            resp = self.client.ChatCompletions(req)
            return resp.Choices[0].Message.Content.strip()
        except Exception as e:
            print(f"关键词提取失败: {e}")
            return description.split()[0]  # 失败时返回第一个词

    def compress_text(self, text):
        """
        压缩文本中的描述为关键词
        :return: (压缩后的文本, 替换记录)
        """
        # 分句处理
        sentences = [s.strip() for s in re.split(r'(?<=[。！？；\n])', text) if s.strip()]
        result = []
        replacements = {}

        for sent in sentences:
            if self._is_description(sent):
                keyword = self._extract_keyword(sent)
                replacements[sent] = keyword
                result.append(keyword)
            else:
                result.append(sent)

        return ''.join(result), replacements




#翻译函数
def translate_text(text, app_id, app_key, from_lang='auto', to_lang='en'):

    url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    salt = str(random.randint(32768, 65536))
    sign = hashlib.md5((app_id + text + salt + app_key).encode()).hexdigest()
    params = {
        'q': text,
        'from': from_lang,
        'to': to_lang,
        'appid': app_id,
        'salt': salt,
        'sign': sign
    }
    try:
        response = requests.get(url, params=params)
        result = json.loads(response.text)
        return result['trans_result'][0]['dst']
    except Exception as e:
        print(f"翻译失败: {e}")
        return None

def calculate_clip_similarity(image, prompt, clip_model, clip_processor, device):
    """计算图像和提示词之间的 CLIP 相似度"""
    inputs = clip_processor(images=image, text=prompt, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    similarity = logits_per_image[0][0].item()
    return similarity

def load_banned_concepts(filepath="banned_concepts.txt"):
    #加载禁止生成概念列表
    banned_concepts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                concept = line.strip()
                if concept:
                    banned_concepts.append(concept)
    except FileNotFoundError:
        print(f"文件未找到: {filepath}")
    return banned_concepts


def main(args):
    app_id = "20250415002333986" 
    app_key = "WF1OihmcE21d7yfQI7VI"#调用百度翻译api

    banned_concepts = load_banned_concepts()  # 从文件加载禁止生成的概念列表
    similarity_threshold = 20  # 相似度阈值，超过此值则进行模糊处理

    model_id = args.pretrained_model_name_or_path
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(args.device)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    torch.Generator(device=args.device).manual_seed(42)
    
    # 加载 CLIP 模型和处理器
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(args.device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if args.generate_training_data:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        num_images = 8
        count = 0
        for single_concept in args.multi_concept:
            for c, t in single_concept:
                count += 1
                print(f"Generating training data for concept {count}: {c}...")
                c = c.replace('-', ' ')
                output_folder = f"{args.output_dir}/{c}"
                os.makedirs(output_folder, exist_ok=True)
                if t == "object":
                    prompt = f"a photo of the {c}"
                    print(f'Inferencing: {prompt}')
                    images = pipe(prompt, num_inference_steps=args.steps, guidance_scale=7.5, num_images_per_prompt=num_images).images
                    for i, im in enumerate(images):
                        im.save(f"{output_folder}/{prompt.replace(' ', '-')}_{i}.jpg")
                elif t == "style":
                    prompt = f"a photo in the style of {c}"
                    print(f'Inferencing: {prompt}')
                    images = pipe(prompt, num_inference_steps=args.steps, guidance_scale=7.5, num_images_per_prompt=num_images).images
                    for i, im in enumerate(images):
                        im.save(f"{output_folder}/{prompt.replace(' ', '-')}_{i}.jpg")
                else:
                    raise ValueError("unknown concept type.")
                del images
                torch.cuda.empty_cache()
                gc.collect()
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        num_images = args.num_images
        output_folder = f"{args.output_dir}/generated_images"
        os.makedirs(output_folder, exist_ok=True)
        print(f"Inference using {args.pretrained_model_name_or_path}...")
        prompt = args.prompt
 
        # 判断是否开启大模型处理
        if args.use_large_model:
            compressor = DescriptionCompressor()
            prompts, replacements = compressor.compress_text(prompt)
            print("原始文本:")
            print(prompt)
            print("\n压缩结果:")
            print(prompts)
            print("\n替换记录:")
            for orig, keyword in replacements.items():
                print(f"{keyword} ← {orig}")
        else:
            prompts = prompt
            print("未使用大模型处理")

        #翻译提示词
        translated_prompt = translate_text(prompts, app_id, app_key)
        print(f"翻译后结果: {translated_prompt}")
       
        images = pipe(translated_prompt, num_inference_steps=args.steps, guidance_scale=7.5, num_images_per_prompt=num_images).images
        
        # 使用原始 SD1.4 模型生成图像
        if args.compare_with_original:
            original_pipe = StableDiffusionPipeline.from_pretrained(args.original_model_path).to(args.device)
            original_pipe.scheduler = DPMSolverMultistepScheduler.from_config(original_pipe.scheduler.config)
            try:
                original_images = original_pipe(translated_prompt, num_inference_steps=args.steps, guidance_scale=7.5, num_images_per_prompt=num_images).images
                print(f"生成原始图像成功！")
            except Exception as e:
                print(f"生成原始图像时出错: {e}")
                original_images = None
            del original_pipe  # 释放显存
            torch.cuda.empty_cache()
            gc.collect()
        else:
            original_images = None

        for i, im in enumerate(images):
            blurred = False  # 标记是否已模糊处理

            # 判断是否开启 CLIP 相似度判断
            if args.use_clip_similarity:
                for banned_concept in banned_concepts:
                    # 计算 CLIP 相似度
                    similarity = calculate_clip_similarity(im, banned_concept, clip_model, clip_processor,
                                                             args.device)
                    print(f"与 '{banned_concept}' 的 CLIP 相似度: {similarity}")
                    if similarity > similarity_threshold:
                        im.save(f"{output_folder}/test.jpg")#存储原图
                        print(f"与 '{banned_concept}' 相似度过高，进行模糊处理...")
                        blurred_image = im.filter(ImageFilter.GaussianBlur(radius=15))  #调整模糊半径
                        blurred_image.save(f"{output_folder}/test_blurred_{banned_concept}.jpg")#存储模糊后的图
                        blurred = True
                        break
            else:
                print("未使用 CLIP 相似度判断")

            if not blurred:
                im.save(f"{output_folder}/test.jpg")
            # 保存原始 SD1.4 模型生成的图像
            if original_images and i < len(original_images):  # 确保 original_images 不为 None 且 i 在范围内
                original_im = original_images[i]
                original_im.save(f"{output_folder}/test_original.jpg")
        torch.cuda.empty_cache()
        gc.collect()

    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_images', type=int, default=3)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    #大模型使用开关
    parser.add_argument('--use_large_model', action='store_true', help='是否使用大模型处理提示词')
    # CLIP 相似度判断开关
    parser.add_argument('--use_clip_similarity', action='store_true', help='是否使用 CLIP 相似度判断')
    # 添加一个参数来指定原始 SD1.4 模型的路径
    parser.add_argument('--compare_with_original', action='store_true', help='是否与原始 SD1.4 模型进行比较')
    parser.add_argument('--original_model_path', type=str, default="CompVis/stable-diffusion-v1-4", help='原始 SD1.4 模型的路径')
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    steps = 30
    model_id = args.model_path
    output_dir = args.save_path
    num_images = args.num_images
    prompt = args.prompt
    
    main(OmegaConf.create({
        "pretrained_model_name_or_path": model_id,
        "generate_training_data": False,
        "device": device,
        "steps": steps,
        "output_dir": output_dir,
        "num_images": num_images,
        "prompt": prompt,
        "use_large_model": args.use_large_model,  # 传递开关参数
        "use_clip_similarity": args.use_clip_similarity, # 传递 CLIP 相似度判断开关参数
        "compare_with_original": args.compare_with_original,
        "original_model_path": args.original_model_path
    }))
