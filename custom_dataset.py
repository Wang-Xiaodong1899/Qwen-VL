from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import json
import random
import torch
import transformers
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import copy
import os


IGNORE_INDEX = -100
eod_tag = '<|endoftext|>'


class CoCODataset(Dataset):  
    def __init__(self, args, tokenizer: PreTrainedTokenizer):  
        self.tokenizer = tokenizer  
        self.max_seq_len = args.model_max_length
        
        img_start_id = tokenizer.img_start_id
        img_end_id = tokenizer.img_end_id
        img_pad_id = tokenizer.img_pad_id
        eod_id = tokenizer.eod_id
        
        # can pass image path
        with open("/f_data/G/dataset/mscoco/train_data.json", "r") as file:  
            self.data = json.load(file)
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        try:
            item = self.data[index]
            name = item['image_name']
            path = os.path.join('/f_data/G/dataset/mscoco/train2017', name)
            caption = random.choice(item['captions'])
            
            # input end
            query = self.tokenizer.from_list_format([
                    {'image': path},
                    {'text': 'Describe this image in English:'},
                ])
            
            sequence = self.tokenizer.from_list_format([
                    {'image': path},
                    {'text': 'Describe this image in English:' + caption + eod_tag},
                ])
            # sample: 'Picture 1:<img>coco/1.jpg</img>\nDescribe this image in English: a dog <|endoftext|>'
            
            source = self.tokenizer(
                query, 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_seq_len,
                return_tensors="pt",
            )
            source_input_ids = source['input_ids'][0]
            source_attn_mask = source['attention_mask'][0]
            source_len = (source_attn_mask==1).sum().item()
            strings = self.tokenizer(
                sequence, 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_seq_len,
                return_tensors="pt",
            )
            input_ids = strings['input_ids'][0]
            labels = copy.deepcopy(input_ids)
            attn_mask = strings['attention_mask'][0]
            string_len = (attn_mask==1).sum().item()
            attn_mask[-1] = 0 # EOS token not attend
            labels[:source_len] = IGNORE_INDEX
            labels[string_len:] = IGNORE_INDEX
            return {
                'input_ids': input_ids,
                'label': labels,
                'attention_mask': attn_mask,
            }
            
        except Exception as e:
            # print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))