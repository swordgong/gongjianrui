import os
import re
import gc
import json
import torch
import argparse
import hashlib
import random
from omegaconf import OmegaConf
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models

# 从环境变量获取敏感信息
TENCENT_SECRET_ID = os.getenv('TENCENT_SECRET_ID')
TENCENT_SECRET_KEY = os.getenv('TENCENT_SECRET_KEY')
BAIDU_APP_ID = os.getenv('BAIDU_APP_ID')
BAIDU_APP_KEY = os.getenv('BAIDU_APP_KEY')

class DescriptionCompressor:
    def __init__(self, keyword_map_file="keyword_rules.txt"):
        """初始化关键词提取器"""
        if not all([TENCENT_SECRET_ID, TENCENT_SECRET_KEY]):
            raise ValueError("腾讯云API凭证未配置")
            
        self.keyword_rules = self._load_keyword_rules(keyword_map_file)
        self.client = self._init_hunyuan_client()
        
    def _init_hunyuan_client(self):
        """安全初始化混元客户端"""
        cred = credential.Credential(TENCENT_SECRET_ID, TENCENT_SECRET_KEY)
        http_profile = HttpProfile(endpoint="hunyuan.tencentcloudapi.com")
        client_profile = ClientProfile(httpProfile=http_profile)
        return hunyuan_client.HunyuanClient(cred, "ap-guangzhou", client_profile)

    def _load_keyword_rules(self, filepath):
        """加载关键词映射规则"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"关键词规则文件不存在: {filepath}")
            
        rules = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('|')
                    if len(parts) < 3:
                        continue
                    keyword = parts[0]
                    synonyms = [s.strip() for s in parts[1].split(',') if s.strip()]
                    features = [f.strip() for f in parts[2].split(',') if f.strip()]
                    rules[keyword] = {'synonyms': synonyms, 'features': features}
        return rules

    def _safe_api_call(self, messages):
        """安全的API调用封装"""
        try:
            req = models.ChatCompletionsRequest()
            req.Messages = messages
            resp = self.client.ChatCompletions(req)
            return resp.Choices[0].Message.Content.strip()
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return None

    def compress_text(self, text):
        """
        压缩文本中的描述为关键词
        返回: (压缩后的文本, 替换记录)
        """
        if not text.strip():
            return text, {}
            
        sentences = [s.strip() for s in re.split(r'(?<=[。！？；\n])', text) if s.strip()]
        result = []
        replacements = {}

        for sent in sentences:
            # 判断是否为描述性文本
            is_desc = self._safe_api_call([{
                "Role": "user",
                "Content": f"判断以下文本是否包含物体外貌描述（只需回答是/否）:\n{sent}"
            }])
            
            if is_desc == '是':
                # 提取关键词
                keyword = self._safe_api_call([{
                    "Role": "user",
                    "Content": f"从以下描述中提取最核心的物体名称（只需返回单个名词）:\n{sent}"
                }])
                
                if keyword:
                    replacements[sent] = keyword
                    result.append(keyword)
                    continue
                    
            result.append(sent)

        return ''.join(result), replacements

def translate_text(text, app_id, app_key, from_lang='auto', to_lang='en'):
    """安全的百度翻译API封装"""
    if not text or not all([app_id, app_key]):
        return None
        
    url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    salt = str(random.randint(32768, 65536))
    sign = hashlib.md5((app_id + text + salt + app_key).encode()).hexdigest()
    
    try:
        response = requests.get(url, params={
            'q': text,
            'from': from_lang,
            'to': to_lang,
            'appid': app_id,
            'salt': salt,
            'sign': sign
        }, timeout=10)
        response.raise_for_status()
        return response.json()['trans_result'][0]['dst']
    except Exception as e:
        print(f"翻译失败: {str(e)}")
        return None

def init_pipeline(model_path, device):
    """初始化Stable Diffusion管道"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if 'cuda' in device else torch.float32
    ).to(device)
    
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe

def main(args):
    # 验证关键参数
    if not args.prompt:
        raise ValueError("必须提供提示词")
    
    # 初始化组件
    try:
        compressor = DescriptionCompressor()
        pipe = init_pipeline(args.pretrained_model_name_or_path, args.device)
        torch.Generator(device=args.device).manual_seed(42)
        
        # 处理提示词
        compressed_prompt, replacements = compressor.compress_text(args.prompt)
        print(f"原始提示: {args.prompt}")
        print(f"压缩结果: {compressed_prompt}")
        
        # 翻译提示词
        translated_prompt = translate_text(
            compressed_prompt, 
            BAIDU_APP_ID, 
            BAIDU_APP_KEY
        )
        print(f"翻译结果: {translated_prompt}")
        
        # 生成图像
        os.makedirs(args.output_dir, exist_ok=True)
        images = pipe(
            translated_prompt,
            num_inference_steps=args.steps,
            guidance_scale=7.5,
            num_images_per_prompt=args.num_images
        ).images
        
        # 保存结果
        for i, img in enumerate(images):
            img.save(os.path.join(args.output_dir, f"result_{i}.jpg"))
            
    finally:
        # 清理资源
        if 'pipe' in locals():
            del pipe
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', type=int, default=1)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default="output")
    parser.add_argument('--steps', type=int, default=30)
    args = parser.parse_args()

    main(OmegaConf.create({
        "pretrained_model_name_or_path": args.model_path,
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        "steps": args.steps,
        "output_dir": args.save_path,
        "num_images": args.num_images,
        "prompt": args.prompt,
    }))