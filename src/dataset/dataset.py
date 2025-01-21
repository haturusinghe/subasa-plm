from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from ast import literal_eval
from sinling import SinhalaTokenizer, POSTagger
import random

from transformers import XLMRobertaTokenizer
import copy
import os
import json
import numpy as np
import emoji
import sys
import argparse

from src.utils.helpers import add_tokens_to_tokenizer, get_token_rationale
from src.utils.logging_utils import setup_logging

class SOLDDataset(Dataset):
    def __init__(self, args, mode='train', tokenizer=None):
        self.train_dataset_path = 'SOLD_DATASET/sold_train_split.json' 
        self.test_dataset_path = 'SOLD_DATASET/sold_test_split.json'

        self.label_list = ['NOT' , 'OFF']
        self.label_count = [0, 0]
        self.logger = setup_logging()
        self.tokenizer = tokenizer

        if mode == 'test':
            with open(self.test_dataset_path, 'r') as f:
                self.dataset = list(json.load(f))
            # Sort dataset by a unique identifier to ensure consistent ordering
            self.dataset.sort(key=lambda x: x['post_id'])
        elif mode == 'train' or mode == 'val':
            with open(self.train_dataset_path, 'r') as f:
                self.dataset = list(json.load(f))
            # Sort dataset by a unique identifier to ensure consistent ordering
            self.dataset.sort(key=lambda x: x['post_id'])

            #use train_test_split to split the train set into train and validation
            train_set, val_set = train_test_split(self.dataset, test_size=0.1, random_state=args.seed)

            if mode == 'train':
                self.dataset = train_set
            elif mode == 'val':
                self.dataset = val_set

            
            for d in self.dataset:
                for i in range(len(self.label_list)):
                    if d['label'] == self.label_list[i]:
                        self.label_count[i] += 1

        # if args.check_errors == True:
        #     # this flag is to make the datasets very small to quickly run the training to see errors
        #     self.dataset = self.dataset[:100]

        if args.intermediate and args.skip_empty_rat:
            rm_idxs = []
            removed_items = []
            for idx, d in enumerate(self.dataset):
                if 1 not in json.loads(d['rationales']) and d['label'] == "OFF":
                    rm_idxs.append(idx)
            rm_idxs.sort(reverse=True)
            for j in rm_idxs:
                removed_items.append(self.dataset[j])
                del self.dataset[j]
            self.logger.info(f"[DATASET] [MODE: {mode}] [SKIP_EMPTY_RAT] Removed {len(removed_items)} items")
            self.logger.info(f"[DATASET] [MODE: {mode}] [REMOVED_ITEMS] {str(removed_items)}")
        
        self.mode = mode
        self.intermediate = args.intermediate

        tokenizer = XLMRobertaTokenizer.from_pretrained(args.pretrained_model)
        self.tokenizer = add_tokens_to_tokenizer(args, tokenizer)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        id = self.dataset[idx]['post_id']
        text = self.dataset[idx]['tokens'] #use tokens key instead of text because the length of rationales is the same as tokens
        label = self.dataset[idx]['label']
        cls_num = self.label_list.index(label)
        
        if self.intermediate == 'mrp' or self.intermediate == 'rp':
            raw_rationale_from_ds = self.dataset[idx]['rationales'] #this is as a string (of a list) in the dataset
            rationales = literal_eval(raw_rationale_from_ds) # converts the raw string to a list of integers

            length_of_rationales = len(rationales)
            length_of_text = len(text.split())
            
            # # convert ratianles back to a string and make sure its same as the original raw_rationale_from_ds
            # back_to_str = "[" + ", ".join([str(r) for r in rationales]) + "]"
            # assert raw_rationale_from_ds == back_to_str, "Rationales are not the same after conversion"

            # there are items in the dataset with label is NOt and the rationale list is an empty arry. so we need to make sure that the length of the rationales is the same as the length of the text
            if label == "NOT" and len(rationales) == 0:
                # create a list of zeros with the same length as the text
                rationales = [0] * length_of_text

            length_of_rationales = len(rationales)
            if length_of_rationales != length_of_text:
                self.logger.error(f"[ERROR] [RAT_LEN] {length_of_rationales} [TEXT_LEN] {length_of_text} [ID] {id}")
                sys.exit(1)

            text_str_split_to_tokens = text.split(' ')
            final_rationale_tokens, text_after_tokenizer = get_token_rationale(self.tokenizer, copy.deepcopy(text_str_split_to_tokens), copy.deepcopy(rationales), copy.deepcopy(id))

            tmp = []
            for r in final_rationale_tokens:
                tmp.append(str(r))
            final_rationales_str = ','.join(tmp)
            return (text, cls_num, final_rationales_str)

        elif self.intermediate == 'mlm':
            encoding = self.tokenizer(text, return_tensors=None, padding=True)
            return encoding

        elif self.intermediate == False:  # hate speech detection
            return (text, cls_num, id)
        
        else:
            return ()


class SOLDAugmentedDataset(SOLDDataset):
    def __init__(self, args, mode='train', tokenizer=None):
        super().__init__(args, mode, tokenizer)
        self.output_dir = "json_dump"
        self.initialize_data_structures()
        self.load_and_process_data()
        self.process_offensive_words()
        self.generate_augmented_data()

    def initialize_data_structures(self):
        """Initialize all data structures used by the class."""
        self.offensive_data_only = []
        self.non_offensive_data_only = []
        self.offensive_ngram_list = []
        self.categoried_offensive_phrases = {}
        self.augmented_data = []
        self.non_offensive_data_selected = []
        self.offensive_data_with_pos_tags = []
        self.non_offensive_data_with_pos_tags = []
        self.pos_tagger = POSTagger()
        self.final_non_offensive_data = []
        self.final_offensive_data = []
        self.offensive_single_word_list_with_pos_tags = []

    def load_and_process_data(self):
        """Load and separate offensive and non-offensive data."""
        for item in self.dataset:
            if item['label'] == 'OFF' and item['rationales'] != "[]":
                self.offensive_data_only.append(item)
            elif item['label'] == 'NOT':
                self.non_offensive_data_only.append(item)
        
        self.offensive_data_only.sort(key=lambda x: x['post_id'])
        self.non_offensive_data_only.sort(key=lambda x: x['post_id'])
        
        self._process_pos_tags()

    def _process_pos_tags(self):
        """Process POS tags for both offensive and non-offensive data."""
        for item in self.offensive_data_only:
            text_tokens = item['tokens'].split()
            rationale = literal_eval(item['rationales'])
            positions_of_offensive_tokens = [i for i, r in enumerate(rationale) if r == 1]
            pos_tags = self.pos_tagger.predict([text_tokens])[0]
            post_tags_offensive_only = [pos_tags[i] for i in positions_of_offensive_tokens]
            self.offensive_data_with_pos_tags.append(pos_tags)
            self.offensive_single_word_list_with_pos_tags.extend(post_tags_offensive_only)

        for item in self.non_offensive_data_only:
            text_tokens = item['tokens'].split()
            pos_tags = self.pos_tagger.predict([text_tokens])[0]
            self.non_offensive_data_with_pos_tags.append(pos_tags)

    def process_offensive_words(self):
        """Extract and categorize offensive phrases."""
        self._extract_offensive_ngrams()
        self.offensive_ngram_list = list(dict.fromkeys(self.offensive_ngram_list))
        self.offensive_single_word_list_with_pos_tags = self.remove_duplicates(self.offensive_single_word_list_with_pos_tags)
        self.categoried_offensive_phrases = self.categorize_offensive_phrases(
            self.offensive_single_word_list_with_pos_tags, # self.offensive_word_list, 
            self.pos_tagger
        )
        self._save_processed_data()

    def _extract_offensive_ngrams(self):
        """Extract offensive phrases from the dataset."""
        for item in self.offensive_data_only:
            text_tokens = item['tokens'].split()
            raw_rationale_tokens = literal_eval(item['rationales'])
            offensive_phrases = self.extract_offensive_phrases(
                text_tokens, 
                raw_rationale_tokens,
                max_ngram=1
            )
            self.offensive_ngram_list.extend(offensive_phrases.keys())

    def _save_processed_data(self):
        """Save all processed data to JSON files."""
        os.makedirs(self.output_dir, exist_ok=True)
        data_to_save = {
            'offensive_phrases.json': self.categoried_offensive_phrases,
            'non_offensive_data_with_pos_tags.json': self.non_offensive_data_with_pos_tags,
            'offensive_data_with_pos_tags.json': self.offensive_data_with_pos_tags,
            'offensive_word_list.json': self.offensive_single_word_list_with_pos_tags
        }
        
        for filename, data in data_to_save.items():
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
    
    def _save_augmented_data(self):
        """Save augmented data to a JSON file."""
        os.makedirs(self.output_dir, exist_ok=True)
        data_to_save = {
            'augmented_data.json': self.augmented_data,
            'non_offensive_data_selected.json': self.non_offensive_data_selected,
            'final_datsaset.json': self.dataset
        }

        for filename, data in data_to_save.items():
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
        


    def generate_augmented_data(self):
        """Generate augmented offensive data from non-offensive sentences."""
        for item in self.non_offensive_data_only[:]:  # Create a copy to iterate
            text_tokens = item['tokens'].split()
            pos_tags = self.pos_tagger.predict([text_tokens])[0]
            
            augmented_tokens, augmented_rationale = self.offensive_token_insertion(text_tokens, pos_tags)
            if augmented_tokens and augmented_rationale:
                augmented_sentence = ' '.join(augmented_tokens)
            
                new_item = {
                    'post_id': f"{item['post_id']}_aug",
                    'text': augmented_sentence,
                    'tokens': augmented_sentence,
                    'rationales': str(augmented_rationale),
                    'label': 'OFF',
                }
                self.augmented_data.append(new_item)
                self.non_offensive_data_selected.append(item)
        
        # make copy of the augmented data and extend it with non_offensive_data_selected to create the final dataset
        copy_of_augmented_data = copy.deepcopy(self.augmented_data)
        copy_of_augmented_data.extend(self.non_offensive_data_selected)
        random.shuffle(copy_of_augmented_data)
        self.dataset = copy_of_augmented_data

        # Print length of each dataset
        self.logger.info(f"[AUGMENTED] [AUGMENTED_DATA_LEN] {len(self.augmented_data)}")
        self.logger.info(f"[AUGMENTED] [NON_OFFENSIVE_DATA_SELECTED_LEN] {len(self.non_offensive_data_selected)}")
        self.logger.info(f"[AUGMENTED] [FINAL_DATASET_LEN] {len(self.dataset)}")

        self._save_augmented_data()

    def offensive_token_insertion(self, tokens, pos_tags):
        """
        Insert offensive tokens into a non-offensive sentence based on POS patterns.
        Returns modified tokens or None if no valid insertions possible.
        """
        if not tokens or not pos_tags:
            return None
            
        modified_tokens = tokens.copy()
        offensive_lexicon = self.categoried_offensive_phrases
        inserted_positions = set()
        count_of_inserted = 0
        
        # Define insertion probabilities
        NOUN_MODIFIER_PROB = 0.5
        VERB_INTENSIFIER_PROB = 0.3
        INTERJECTION_PROB = 0.2
        MAX_NEW_PHRASES_ALLOWED = 3
        
        for i, (token, tag) in enumerate(pos_tags):

            if count_of_inserted >= MAX_NEW_PHRASES_ALLOWED:
                break

            if i in inserted_positions:
                continue
                
            if tag.startswith('NN') and offensive_lexicon['noun_modifiers']:
                if random.random() < NOUN_MODIFIER_PROB:
                    offensive_modifier = random.choice(offensive_lexicon['noun_modifiers'])
                    modified_tokens.insert(i, offensive_modifier)
                    inserted_positions.add(i)
                    count_of_inserted += 1
                    
            elif tag.startswith('VB') and offensive_lexicon['verb_intensifiers']:
                if random.random() < VERB_INTENSIFIER_PROB:
                    offensive_intensifier = random.choice(offensive_lexicon['verb_intensifiers'])
                    modified_tokens.insert(i + 1, offensive_intensifier)
                    inserted_positions.add(i + 1)
                    count_of_inserted += 1
        
        if count_of_inserted <= MAX_NEW_PHRASES_ALLOWED:
            if random.random() < INTERJECTION_PROB and offensive_lexicon['interjections']:
                if random.choice([True, False]):
                    modified_tokens.insert(0, random.choice(offensive_lexicon['interjections']))
                    # if 0 is already in the inserted_positions, make it 1
                    if 0 in inserted_positions:
                        inserted_positions.add(1)

                    inserted_positions.add(0)
                else:
                    modified_tokens.append(random.choice(offensive_lexicon['interjections']))
                    inserted_positions.add(len(modified_tokens) - 1)
        
        pre_modified_tokens = modified_tokens.copy()
        raw_rationale_tokens = [0] * len(pre_modified_tokens)
        after_modification_rationale_tokens = []

        for i, (token, rationale_token) in enumerate(zip(pre_modified_tokens, raw_rationale_tokens)):
            if i in inserted_positions:
                if len(token.split()) > 1:
                    rationale_token = [1] * len(token.split())
                else:
                    rationale_token = [1]
            else:
                rationale_token = [0]
            after_modification_rationale_tokens.extend(rationale_token)
        
        new_setence = ' '.join(pre_modified_tokens)

        if new_setence.split() == tokens:
            return None, None
        
        return pre_modified_tokens, after_modification_rationale_tokens

    @staticmethod
    def extract_offensive_phrases(tokens, rationales, max_ngram=1):

        offensive_phrases = {}

        # Find consecutive offensive tokens
        for n in range(1, max_ngram + 1):
            for i in range(len(tokens) - n + 1):
                # Check if all tokens in this window are marked offensive
                rat_portion = rationales[i:i+n]
                str_portion = tokens[i:i+n]
                if all(rat_portion):
                    phrase = ' '.join(str_portion)
                    if phrase not in offensive_phrases:
                        offensive_phrases[phrase] = 1
                    else:
                        offensive_phrases[phrase] += 1
                        
        return offensive_phrases

    @staticmethod
    def categorize_offensive_phrases_old(phrases, pos_tagger):
        categorized = {
            'noun_phrases': [],
            'verb_phrases': [],
            'adjective_phrases': [],
            'mixed_phrases': []
        }
        
        for phrase in phrases:
            tokens = phrase.split()
            pos_tags = pos_tagger.predict([tokens])[0]
            
            # Determine phrase type based on POS pattern
            if all(tag[1].startswith('N') for tag in pos_tags):
                categorized['noun_phrases'].append(tuple(pos_tags))
            elif all(tag[1].startswith('V') for tag in pos_tags):
                categorized['verb_phrases'].append(tuple(pos_tags))
            elif all(tag[1].startswith('J') for tag in pos_tags):
                categorized['adjective_phrases'].append(tuple(pos_tags))
            else:
                categorized['mixed_phrases'].append(tuple(pos_tags))

        categorized['noun_phrases'] = list(dict.fromkeys(categorized['noun_phrases']))
        categorized['verb_phrases'] = list(dict.fromkeys(categorized['verb_phrases']))
        categorized['adjective_phrases'] = list(dict.fromkeys(categorized['adjective_phrases']))
        categorized['mixed_phrases'] = list(dict.fromkeys(categorized['mixed_phrases']))
                
        return categorized

    @staticmethod
    def categorize_offensive_phrases(offensive_single_word_list_with_pos_tags, pos_tagger):
        categorized = {
            'noun_modifiers': [],      # words that modify nouns (adjectives, determiners)
            'verb_intensifiers': [],   # words that intensify verbs (adverbs, particles)
            'interjections': [],       # standalone expressions, exclamations
            'offensive_nouns': [],     # offensive nouns for direct replacement
            'offensive_verbs': [],     # offensive verbs for action replacement
            'pronouns': [],           # offensive pronouns (තෝ, උඹ)
            'particles': [],          # emphatic particles that add offensive tone
            'adjectives': [],         # offensive adjectives
            'adverbs': []            # offensive adverbs
        }


        
        for word,tag in offensive_single_word_list_with_pos_tags:
            
            # Comprehensive single token classification
            if tag.startswith('N'):
                categorized['offensive_nouns'].append(word)
                # Nouns can also act as modifiers in Sinhala
                if tag == 'NNC':
                    categorized['noun_modifiers'].append(word)
                    
            elif tag.startswith('V'):
                categorized['offensive_verbs'].append(word)
                # Some verb forms can act as intensifiers
                if tag in ['VNN', 'VNF']:
                    categorized['verb_intensifiers'].append(word)
                    
            elif tag.startswith('J'):
                categorized['adjectives'].append(word)
                categorized['noun_modifiers'].append(word)
                
            elif tag == 'RB':
                categorized['adverbs'].append(word)
                categorized['verb_intensifiers'].append(word)
                
            elif tag == 'DET' or tag == 'PRP':
                categorized['pronouns'].append(word)
                # Determiners and pronouns can modify nouns
                categorized['noun_modifiers'].append(word)
                
            elif tag in ['UH', 'FS', 'INJ']:
                categorized['interjections'].append(word)
                
            elif tag == 'RP':
                categorized['particles'].append(word)
                # Particles can intensify both nouns and verbs
                categorized['verb_intensifiers'].append(word)
                categorized['noun_modifiers'].append(word)
                
            # Words that can serve multiple functions
            if tag in ['JJ', 'RB', 'RP', 'UH']:
                # These can often be used as standalone offensive expressions
                categorized['interjections'].append(word)

        # Remove duplicates while preserving order
        for category in categorized:
            categorized[category] = list(dict.fromkeys(categorized[category]))
            
        return categorized




    def offensive_token_insertion_old(self, tokens, pos_tags):
        """
        Insert offensive tokens into a non-offensive sentence based on POS patterns
        
        Args:
            tokens: List of original sentence tokens
            pos_tags: List of POS tags for each token [(word1, tag1), (word2, tag2),...]
            offensive_lexicon: Dictionary with categories of offensive words
                {
                    'noun_modifiers': [...],  # offensive words that can modify nouns
                    'verb_intensifiers': [...],  # offensive words that can intensify verbs
                    'interjections': [...],  # standalone offensive expressions
                    'offensive_nouns': [...],  # offensive nouns for replacement
                }
        """
        import random
        modified_tokens = tokens.copy()
        offensive_lexicon = self.categoried_offensive_phrases
        
        # Track insertion positions to avoid multiple insertions at same spot
        inserted_positions = set()
        
        # Iterate through tokens and their POS tags
        for i, (token, tag) in enumerate(pos_tags):
            # Skip if we already inserted at this position
            if i in inserted_positions:
                continue
                
            # Case 1: Insert before nouns
            if tag[1].startswith('NN'):
                if random.random() < 0.5:  # 50% chance to insert
                    offensive_modifier = random.choice(offensive_lexicon['noun_modifiers'])
                    modified_tokens.insert(i, offensive_modifier)
                    inserted_positions.add(i)
                    
            # Case 2: Insert after verbs
            elif tag[1].startswith('VB'):
                if random.random() < 0.3:  # 30% chance to insert
                    offensive_intensifier = random.choice(offensive_lexicon['verb_intensifiers'])
                    modified_tokens.insert(i + 1, offensive_intensifier)
                    inserted_positions.add(i + 1)
        
        # Case 3: Add interjection at the beginning or end (20% chance)
        if random.random() < 0.2:
            if random.choice([True, False]):  # randomly choose start or end
                modified_tokens.insert(0, random.choice(offensive_lexicon['interjections']))
            else:
                modified_tokens.append(random.choice(offensive_lexicon['interjections']))
        
        return modified_tokens
    
    @staticmethod
    def remove_duplicates(list_of_pairs):
        seen = {}
        return list(dict.fromkeys(map(tuple, list_of_pairs)).keys())


