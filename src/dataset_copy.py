from torch.utils.data import Dataset
from src.cfr_utils import *
from torchvision import transforms
from PIL import Image
import random
from pathlib import Path
import os
from openai import OpenAI
import regex as re

import spacy
import torch
import networkx as nx
import pandas as pd
import json
from transformers import AutoModel, AutoTokenizer
from srcc.cfr_utils import prompt_augmentation #new
nlp = spacy.load("zh_core_web_sm")  # 采用spacy中文工具
import pickle

BASE_URL = ''
API_KEY = ''

#构建新的知识图谱
def load_chinese_conceptnet(file_path="chineseconceptnet.csv"):
    conceptnet_data = pd.read_csv(file_path, sep='\t', header=None, quotechar='"', encoding='utf-8')
    knowledge_graph = nx.MultiDiGraph()
    for _, row in conceptnet_data.iterrows():
        relation = row[1]
        start = row[2]
        end = row[3]
        edge_data = json.loads(row[4])
        weight = edge_data.get('weight', 1.0)
        knowledge_graph.add_edge(start, end, relation=relation, weight=weight)
    print(f"知识图谱构建成功！节点数: {len(knowledge_graph.nodes)}, 边数: {len(knowledge_graph.edges)}")
    
    return knowledge_graph

def load_saved_knowledge_graph(file_path="knowledge_graph.pkl"):
    #加载预构建的知识图谱
    try:
        with open(file_path, 'rb') as f:
            knowledge_graph = pickle.load(f)
        print(f"已加载知识图谱，节点数: {len(knowledge_graph.nodes)}, 边数: {len(knowledge_graph.edges)}")
        return knowledge_graph
    except Exception as e:
        print(f"加载知识图谱失败: {e}")
        return None


#BERT嵌入，使用多语言模型
class BERTEmbedding:
    def __init__(self, model_name="bert-base-multilingual-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        #如果可用，将模型移到GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def __call__(self, entity_names):
        if isinstance(entity_names, str):
            entity_names = [entity_names]
            
        try:
            # 确保输入的实体名称是小写(因为使用了uncased模型)
            entity_names = [name.lower() for name in entity_names]
            
            inputs = self.tokenizer(
                entity_names, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # 获取[CLS]标记的输出作为实体嵌入
            embeddings = outputs.last_hidden_state[:, 0, :]
            print(f"成功为实体生成嵌入: {entity_names}")
            return embeddings
            
        except Exception as e:
            print(f"实体嵌入生成失败: {e}")
            return None

#清理提示词
def clean_prompt(class_prompt_collection):
    class_prompt_collection = [re.sub(
        r"[0-9]+", lambda num: '' * len(num.group(0)), prompt) for prompt in class_prompt_collection]
    class_prompt_collection = [re.sub(
        r"^\.+", lambda dots: '' * len(dots.group(0)), prompt) for prompt in class_prompt_collection]
    class_prompt_collection = [x.strip() for x in class_prompt_collection]
    class_prompt_collection = [x.replace('"', '') for x in class_prompt_collection]
    return class_prompt_collection

#使用大模型生成相关概念，一并擦除
def text_augmentation(erased_concept, mapping_concept, concept_type, num_text_augmentations=100):
    
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    
    class_prompt_collection = []

    if concept_type == 'object':
        messages = [
            {"role": "system", "content": "You can describe any image via text and provide captions for wide variety of images that is possible to generate."},
            {"role": "user", "content": f"Generate {num_text_augmentations} captions for images containing {erased_concept}. The caption should also contain the word '{erased_concept}'. Please do not use any emojis in the captions."},
        ]
        
        while True:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
            )
            class_prompt_collection += [x for x in completion.choices[0].message.content.lower(
            ).split('\n') if erased_concept in x]
            messages.append(
                {"role": "assistant", "content": completion.choices[0].message.content})
            messages.append(
                {"role": "user", "content": f"Generate {num_text_augmentations-len(class_prompt_collection)} more captions"})
            if len(class_prompt_collection) >= num_text_augmentations:
                break
            
        class_prompt_collection = clean_prompt(class_prompt_collection)[:num_text_augmentations]
        class_prompt_formated = []
        mapping_prompt_formated = []
        
        for prompt in class_prompt_collection:
            class_prompt_formated.append((prompt, erased_concept))
            mapping_prompt_formated.append((prompt.replace(erased_concept, mapping_concept), mapping_concept))
    

        print("相关概念已生成")
        return class_prompt_formated, mapping_prompt_formated
        
#预处理准备数据        
class MACEDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        tokenizer,
        size=512,
        center_crop=False,
        use_pooler=False,
        multi_concept=None,
        mapping=None,
        augment=True,
        batch_size=None,
        with_prior_preservation=False,
        preserve_info=None,
        num_class_images=None,
        train_seperate=False,
        aug_length=50,
        prompt_len=250,
        input_data_path=None,
        use_gpt=False,

        knowledge_graph=None,                               # 知识图谱对象
        entity_embedding=None,                              # 实体嵌入模型（BERT）
        kg_processor=spacy.load("zh_core_web_sm"),          #实体提取和链接工具（spaCy）
        knowledge_graph_path="knowledge_graph.pkl",         # 知识图谱文件路径
        memory_efficient=False                             # 是否对Bert嵌入使用半精度内存优化
    ):  
        #新增
        if knowledge_graph is None:
            #self.knowledge_graph = load_chinese_conceptnet()  # 加载知识图谱
            self.knowledge_graph = load_saved_knowledge_graph(knowledge_graph_path)
        # else:
        #     self.knowledge_graph = knowledge_graph

        # if entity_embedding is None:
        #     self.entity_embedding = BERTEmbedding(
        #         model_name="bert-base-multilingual-uncased"
        #     )
        #     if memory_efficient:
        #         self.entity_embedding.model = self.entity_embedding.model.half()  # 使用半精度
        #         torch.cuda.empty_cache()    # 清理GPU缓存
        #         print("已启用内存优化,模型使用半精度,清理GPU缓存。")
        # else:
        #     self.entity_embedding = entity_embedding

        try:
            self.entity_embedding = BERTEmbedding(model_name="bert-base-multilingual-uncased")
            if memory_efficient:
                self.entity_embedding.model = self.entity_embedding.model.half()
                torch.cuda.empty_cache()
            print("实体嵌入模型加载成功!")
            self.has_entity_embedding = True
        except Exception as e:
            print(f"实体嵌入模型加载失败: {e}")
            self.entity_embedding = None  # 确保后续代码能继续运行
            self.has_entity_embedding = False
    

        self.with_prior_preservation = with_prior_preservation
        self.use_pooler = use_pooler
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.batch_counter = 0
        self.batch_size = batch_size
        self.concept_number = 0
        self.train_seperate = train_seperate
        self.aug_length = aug_length
        self.entity_embedding = entity_embedding  # 保存 entity_embedding 模型
        self.kg_processor = kg_processor  # 用于从文本中提取实体

        self.all_concept_image_path  = []
        self.all_concept_mask_path  = []
        single_concept_images_path = []
        self.instance_prompt  = []
        self.target_prompt  = []
        
        self.num_instance_images = 0
        self.dict_for_close_form = []
        self.class_images_path = []
        
        for concept_idx, (data, mapping_concept) in enumerate(zip(multi_concept, mapping)):
            c, t = data
            
            if input_data_path is not None:
                p = Path(os.path.join(input_data_path, c.replace("-", " ")))
                if not p.exists():
                    raise ValueError(f"实例 {p} 图片不存在。")
                
                if t == "object":
                    p_mask = Path(os.path.join(input_data_path, c.replace("-", " ")).replace(f'{c.replace("-", " ")}', f'{c.replace("-", " ")} mask'))
                    if not p_mask.exists():
                        raise ValueError(f"实例 {p_mask} 图片不存在。")
            else:
                raise ValueError(f"未提供输入数据路径。")    
            
            image_paths = sorted(list(p.iterdir()))
            single_concept_images_path = []
            single_concept_images_path += image_paths
            self.all_concept_image_path.append(single_concept_images_path)
            
            if t == "object":
                mask_paths = sorted(list(p_mask.iterdir()))
                single_concept_masks_path = []
                single_concept_masks_path += mask_paths
                self.all_concept_mask_path.append(single_concept_masks_path)
                     
            erased_concept = c.replace("-", " ")
            
            if use_gpt:
                class_prompt_collection, mapping_prompt_collection = text_augmentation(erased_concept, mapping_concept, t, num_text_augmentations=self.aug_length)
                self.instance_prompt.append(class_prompt_collection)
                self.target_prompt.append(mapping_prompt_collection)
            else: 
                sampled_indices = random.sample(range(0, prompt_len), self.aug_length)
                self.instance_prompt.append(prompt_augmentation(erased_concept, augment=augment, sampled_indices=sampled_indices, concept_type=t))
                self.target_prompt.append(prompt_augmentation(mapping_concept, augment=augment, sampled_indices=sampled_indices, concept_type=t))
                
            self.num_instance_images += len(single_concept_images_path)
            
            entry = {"old": self.instance_prompt[concept_idx], "new": self.target_prompt[concept_idx]}
            self.dict_for_close_form.append(entry)
        #先验知识保留    
        if with_prior_preservation:
            class_data_root = Path(preserve_info['preserve_data_dir'])
            if os.path.isdir(class_data_root):
                class_images_path = list(class_data_root.iterdir())
                class_prompt = [preserve_info["preserve_prompt"] for _ in range(len(class_images_path))]
            else:
                with open(class_data_root, "r") as f:
                    class_images_path = f.read().splitlines()
                with open(preserve_info["preserve_prompt"], "r") as f:
                    class_prompt = f.read().splitlines()

            class_img_path = [(x, y) for (x, y) in zip(class_images_path, class_prompt)]
            self.class_images_path.extend(class_img_path[:num_class_images])
                     
        self.image_transforms = transforms.Compose(
            [
                # transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),#裁剪
                transforms.ToTensor(),#转化为pytorch张量
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),#标准化
            ]
        )
        
        self._concept_num = len(self.instance_prompt)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_instance_images // self._concept_num, self.num_class_images)
        print("知识图谱是否加载:", self.knowledge_graph is not None)
        print("实体嵌入模型是否加载:", self.entity_embedding is not None)

        
    def __len__(self):
        return self._length

    #提取实体并关联信息
    def extract_entities(self, prompt):
        if self.kg_processor is None or self.knowledge_graph is None:
            print(f"警告: kg_processor 或 knowledge_graph 为空")
            return []
        print(f"正在处理提示词: {prompt}")
        doc = self.kg_processor(prompt)
        entities = []
        for ent in doc.ents:
            print(f"发现实体: {ent.text}")
            kg_info = None
            if ent.text in self.knowledge_graph.nodes:
                # 获取与实体相关的边的信息
                edges = self.knowledge_graph.edges(ent.text, data=True)
                kg_info = list(edges)  # 保存边的信息
                print(f"实体 '{ent.text}' 在知识图谱中找到 {len(kg_info)} 条关系")
            else:
                print(f"实体 '{ent.text}' 在知识图谱中未找到")
            entities.append((ent.text, kg_info))
        if not entities:
            print(f"警告: 在提示词中未找到任何实体")
        return entities


#按索引加载单条训练数据
    def __getitem__(self, index):
        example = {}
        
        if not self.train_seperate:
            if self.batch_counter % self.batch_size == 0:
                self.concept_number = random.randint(0, self._concept_num - 1)
            self.batch_counter += 1
        
        instance_image = Image.open(self.all_concept_image_path[self.concept_number][index % self._length])
        
        if len(self.all_concept_mask_path) == 0:
            # artistic style erasure
            binary_tensor = None
        else:
            # object/celebrity erasure
            instance_mask = Image.open(self.all_concept_mask_path[self.concept_number][index % self._length])
            instance_mask = instance_mask.convert('L')
            trans = transforms.ToTensor()
            binary_tensor = trans(instance_mask)
        
        prompt_number = random.randint(0, len(self.instance_prompt[self.concept_number]) - 1)
        instance_prompt, target_tokens = self.instance_prompt[self.concept_number][prompt_number]
        
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_prompt"] = instance_prompt
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_masks"] = binary_tensor

        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        prompt_ids = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length
        ).input_ids

        concept_ids = self.tokenizer(
            target_tokens,
            add_special_tokens=False
        ).input_ids             

        pooler_token_id = self.tokenizer(
            "<|endoftext|>",
            add_special_tokens=False
        ).input_ids[0]

        concept_positions = [0] * self.tokenizer.model_max_length
        for i, tok_id in enumerate(prompt_ids):
            if tok_id == concept_ids[0] and prompt_ids[i:i + len(concept_ids)] == concept_ids:
                concept_positions[i:i + len(concept_ids)] = [1]*len(concept_ids)
            if self.use_pooler and tok_id == pooler_token_id:
                concept_positions[i] = 1
        example["concept_positions"] = torch.tensor(concept_positions)[None]               

        if self.with_prior_preservation:
            class_image, class_prompt = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["preserve_images"] = self.image_transforms(class_image)
            example["preserve_prompt_ids"] = self.tokenizer(
                class_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
        # 提取实体并生成嵌入
        if self.knowledge_graph and self.entity_embedding:
            entities = self.extract_entities(instance_prompt)
            if entities:
                entity_texts = [e[0] for e in entities]
                entity_embeddings = self.entity_embedding(entity_texts)
                example["entity_embeddings"] = entity_embeddings
                example["entity_info"] = entities
            else:
                example["entity_embeddings"] = None
                example["entity_info"] = None
        else:
            example["entity_embeddings"] = None
            example["entity_info"] = None

        if self.knowledge_graph and self.entity_embedding:
            print("\n=== 调试信息 ===")
            print("提示词:", instance_prompt)
            print("spaCy模型:", self.kg_processor.meta["name"])
            entities = self.extract_entities(instance_prompt)
            print("提取的实体:", entities)

        return example
    
