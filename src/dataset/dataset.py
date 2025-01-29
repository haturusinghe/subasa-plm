from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from ast import literal_eval
from sinling import POSTagger
import random

from transformers import XLMRobertaTokenizer
import copy
import os
import json
import sys

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
            # Subset for debug
            if args.debug:
                subset_size = len(self.dataset) // 20
                self.dataset = self.dataset[:subset_size]
        elif mode == 'train' or mode == 'val':
            with open(self.train_dataset_path, 'r') as f:
                self.dataset = list(json.load(f))
            # Sort dataset by a unique identifier to ensure consistent ordering
            self.dataset.sort(key=lambda x: x['post_id'])
            # Subset for debug before splitting
            if args.debug:
                subset_size = len(self.dataset) // 20
                self.dataset = self.dataset[:subset_size]

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
        
        self.mode = mode
        self.intermediate = args.intermediate

        tokenizer = XLMRobertaTokenizer.from_pretrained(args.pretrained_model)
        self.tokenizer = add_tokens_to_tokenizer(args, tokenizer)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        id =  666 #self.dataset[idx]['post_id']
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


from typing import List, Dict, Set, Tuple
from pathlib import Path

class SOLDAugmentedDataset(SOLDDataset):
    # Configuration constants
    MAX_NEW_PHRASES_ALLOWED = 3
    MAX_NEW_SENTENCES_GENERATED = 2
    AUGMENTATION_STRATEGIES = [
        "Noun-Based Insertions",
        "Adjective Replacement",
        "Verb Modification",
        "Proper Noun Modification",
        "Hybrid Approach",
        "Insert for PUNC",
        "Compound Verb Modification",
        "Extended Verb Patterns",
        "Case Marker Insertion",
        "Proper Noun Punctuation"
    ]


    def __init__(self, args, mode='train', tokenizer=None):
        super().__init__(args, mode, tokenizer)
        self.output_dir = Path("json_dump")
        self.output_dir.mkdir(exist_ok=True)
        self.initialize_data_structures()
        self.load_and_process_data()
        self.process_offensive_words()
        self.generate_augmented_data()

    def initialize_data_structures(self) -> None:
        """Initialize all data structures used by the class."""
        self.offensive_data_only: List[Dict] = []
        self.non_offensive_data_only: List[Dict] = []
        self.offensive_ngram_list: List[str] = []
        self.categoried_offensive_phrases: Dict = {}
        self.augmented_data: List[Dict] = []
        self.non_offensive_data_selected: List[Dict] = []
        self.offensive_data_with_pos_tags: List[List[Tuple[str, str]]] = []
        self.non_offensive_data_with_pos_tags: List[List[Tuple[str, str]]] = []
        self.pos_tagger = POSTagger()
        self.offensive_single_word_list_with_pos_tags: List[Tuple[str, str]] = []

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
        self.categoried_offensive_phrases = self.categorize_offensive_phrases(
            self.offensive_single_word_list_with_pos_tags, # self.offensive_word_list, 
        )
        self.categoried_offensive_phrases = self.filter_low_count_words(self.categoried_offensive_phrases, frequency_threshold=4)
        self._save_processed_data()
    
    @staticmethod
    def filter_low_count_words(pos_dict, frequency_threshold=2):
        """Remove words with count < 2 from POS dictionary"""

        filtered_data = {
            outer_key: {inner_key: freq for inner_key, freq in inner_dict.items() if freq >= frequency_threshold}
            for outer_key, inner_dict in pos_dict.items()
        }

        return filtered_data

    def _extract_offensive_ngrams(self):
        # TODO: Implement this method if needed
        pass

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
    
    @staticmethod
    def shuffle_digits(number):
        digits = list(str(number))
        random.shuffle(digits)
        shuffled_number = ''.join(digits)
        return int(shuffled_number)

    def generate_augmented_data(self):
        """Generate augmented offensive data from non-offensive sentences."""
        for item in self.non_offensive_data_only[:]:  # Create a copy to iterate
            text_tokens = item['tokens'].split()
            pos_tags = self.pos_tagger.predict([text_tokens])[0]
            
            augmented_tokens_list, augmented_rationale_list = self.offensive_token_insertion(text_tokens, pos_tags)

            for augmented_tokens, augmented_rationale in zip(augmented_tokens_list, augmented_rationale_list):
                if augmented_tokens and augmented_rationale:
                    augmented_sentence = ' '.join(augmented_tokens)
                
                    new_item = {
                        'post_id': self.shuffle_digits(item['post_id']),
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

        # Apply debug subsetting here
        if self.args.debug:
            subset_size = len(self.dataset) // 20
            self.dataset = self.dataset[:subset_size]

        # Print length of each dataset
        self.logger.info(f"[AUGMENTED] [AUGMENTED_DATA_LEN] {len(self.augmented_data)}")
        self.logger.info(f"[AUGMENTED] [NON_OFFENSIVE_DATA_SELECTED_LEN] {len(self.non_offensive_data_selected)}")
        self.logger.info(f"[AUGMENTED] [FINAL_DATASET_LEN] {len(self.dataset)}")

        self._save_augmented_data()

    @staticmethod
    def categorize_offensive_phrases(offensive_single_word_list_with_pos_tags):
        """
            POS Tag and Meaning:

            NNC - Common Noun  
            NNP - Proper Noun  
            PRP - Pronoun  
            QUE - Questioning Pronoun  
            NDT - Deterministic Pronoun  
            QBE - Question Based Pronoun  
            VFM - Verb Finite  
            VP - Verb Particle  
            VNN - Verbal Noun  
            AUX - Modal Auxiliary  
            VNF - Verb Non Finite  
            NCV - Noun in Compound Verb  
            JCV - Adjective in Compound Verb  
            RRPCV - Particle in Compound Verb  
            JJ - Adjective  
            NNJ - Adjectival Noun  
            RB - Adverbs  
            POST - Postposition  
            CC - Conjunction  
            RP - Particle  
            DET - Determiner  
            CM - Case Maker  
            NVB - Noun in Sentence Ending  
            NUM - Number  
            ABB - Abbreviation  
            FS - Full Stop  
            PUNC - Punctuation  
            FRW - Foreign Word  
            UNK - Undefined

        """

        categorized = {}
        # Initialize the structure with an empty dict for each tag we encounter
        for _, tag in offensive_single_word_list_with_pos_tags:
            if tag not in categorized:
                categorized[tag] = {}

        # Count occurrences of each word for each POS tag
        for word, tag in offensive_single_word_list_with_pos_tags:
            if word not in categorized[tag]:
                categorized[tag][word] = 0
            categorized[tag][word] += 1
            
        return categorized

    def offensive_token_insertion(
        self, 
        tokens: List[str], 
        pos_tags: List[Tuple[str, str]]
    ) -> Tuple[List[str], List[int]]:
        """Generate augmented offensive sentences using various strategies."""
        if not tokens or not pos_tags:
            return [], []

        new_offensive_sentences_tokens = []
        new_offensive_sentences_rationale = []
        failed_strategies: Set[str] = set()
        
        while len(new_offensive_sentences_tokens) < self.MAX_NEW_SENTENCES_GENERATED:
            modified_tokens = tokens.copy()
            inserted_positions: Set[int] = set()
            count_inserted = 0
            
            available_strategies = [s for s in self.AUGMENTATION_STRATEGIES if s not in failed_strategies]
            if not available_strategies:
                break
                
            strategy = random.choice(available_strategies)
            trigrams = list(zip(pos_tags[:-2], pos_tags[1:-1], pos_tags[2:]))
            trigrams = trigrams

            for i, trigram in enumerate(trigrams):
                if count_inserted >= self.MAX_NEW_PHRASES_ALLOWED:
                    break
                    
                modified_tokens, inserted_positions, new_count = self._apply_augmentation_strategy(
                    strategy, tokens, trigram, i, modified_tokens, 
                    inserted_positions, self.categoried_offensive_phrases
                )
                count_inserted += new_count

            rationale = self._generate_rationale(modified_tokens, inserted_positions)
            new_sentence = ' '.join(modified_tokens)

            if new_sentence.split() != tokens:
                new_offensive_sentences_tokens.append(new_sentence.split())
                new_offensive_sentences_rationale.append(rationale)
            else:
                failed_strategies.add(strategy)
            
        return new_offensive_sentences_tokens, new_offensive_sentences_rationale

    def _generate_rationale(
        self, 
        tokens: List[str], 
        inserted_positions: Set[int]
    ) -> List[int]:
        """Generate rationale tokens for the augmented sentence."""
        rationale = []
        for i, token in enumerate(tokens):
            if i in inserted_positions:
                rationale.extend([1] * len(token.split()))
            else:
                rationale.append(0)
        return rationale

    def _apply_augmentation_strategy(
        self, 
        strategy: str, 
        tokens: List[str], 
        trigram: Tuple[Tuple[str, str], ...], 
        i: int,
        modified_tokens: List[str],
        inserted_positions: Set[int],
        offensive_lexicon: Dict
    ) -> Tuple[List[str], Set[int], int]:
        """Apply a specific augmentation strategy and return modified data."""
        count_inserted = 0
        t1, t2, t3 = trigram

        strategy_handlers = {
                "Noun-Based Insertions": self._handle_noun_insertions,
                "Adjective Replacement": self._handle_adjective_replacement,
                "Verb Modification": self._handle_verb_modification,
                "Proper Noun Modification": self._handle_proper_noun_modification,
                "Hybrid Approach": self._handle_hybrid_approach,
                "Insert for PUNC": self._handle_punctuation_noun_pattern,
                "Compound Verb Modification": self._handle_compound_verb_modification,
                "Case Marker Insertion": self._handle_case_marker_insertion,
                "Proper Noun Punctuation": self.handle_proper_noun_punctuation
            }


        if strategy in strategy_handlers:
            modified_tokens, inserted_positions, count_inserted = strategy_handlers[strategy](
                tokens, trigram, i, modified_tokens, inserted_positions, offensive_lexicon
            )

        return modified_tokens, inserted_positions, count_inserted

    def _handle_noun_insertions(
        self, 
        tokens: List[str], 
        trigram: Tuple[Tuple[str, str], ...], 
        i: int,
        modified_tokens: List[str],
        inserted_positions: Set[int],
        offensive_lexicon: Dict
    ) -> Tuple[List[str], Set[int], int]:
        t1, t2, t3 = trigram
        count_inserted = 0
        
        # Pattern: NNC -> NNC -> NNC
        if all(t[1] == "NNC" for t in [t1, t2, t3]):
            offensive_noun = random.choice(list(offensive_lexicon['NNC'].keys()))
            modified_tokens.insert(i+1, offensive_noun)  # Insert between first two nouns
            inserted_positions.add(i+1)
            count_inserted += 1
        return modified_tokens, inserted_positions, count_inserted

    def _handle_adjective_replacement(
        self, 
        tokens: List[str], 
        trigram: Tuple[Tuple[str, str], ...], 
        i: int,
        modified_tokens: List[str],
        inserted_positions: Set[int],
        offensive_lexicon: Dict
    ) -> Tuple[List[str], Set[int], int]:
        t1, t2, t3 = trigram
        count_inserted = 0
        
        # Pattern: NNC -> JJ -> NNC or JJ -> JJ -> NNC
        if ((t1[1] == "NNC" and t2[1] == "JJ" and t3[1] == "NNC") or
            (t1[1] == "JJ" and t2[1] == "JJ" and t3[1] == "NNC")):
            offensive_adj = random.choice(list(offensive_lexicon['JJ'].keys()))

            if t1[1] == "JJ" and t2[1] == "JJ":
                # Insert between consecutive adjectives
                modified_tokens.insert(i+1, offensive_adj)
                inserted_positions.add(i+1)
            else:
                # Replace existing adjective
                modified_tokens[i+1] = offensive_adj
                inserted_positions.add(i+1)
            count_inserted += 1
        return modified_tokens, inserted_positions, count_inserted

    def _handle_verb_modification(
        self, 
        tokens: List[str], 
        trigram: Tuple[Tuple[str, str], ...], 
        i: int,
        modified_tokens: List[str],
        inserted_positions: Set[int],
        offensive_lexicon: Dict
    ) -> Tuple[List[str], Set[int], int]:
        t1, t2, t3 = trigram
        count_inserted = 0
        
        # Patterns: NNC -> VP -> NNC or NNC -> NNC -> VP
        if ((t1[1] == "NNC" and t2[1] == "VP" and t3[1] == "NNC") or
            (t1[1] == "NNC" and t2[1] == "NNC" and t3[1] == "VP")):
            offensive_verb = random.choice(list(offensive_lexicon['VP'].keys()))

            if t2[1] == "VP":
                modified_tokens[i+1] = offensive_verb
                inserted_positions.add(i+1)
            else:
                modified_tokens.insert(i+2, offensive_verb)
                inserted_positions.add(i+2)
            count_inserted += 1
        return modified_tokens, inserted_positions, count_inserted

    def _handle_hybrid_approach(
        self, 
        tokens: List[str], 
        trigram: Tuple[Tuple[str, str], ...], 
        i: int,
        modified_tokens: List[str],
        inserted_positions: Set[int],
        offensive_lexicon: Dict
    ) -> Tuple[List[str], Set[int], int]:
        t1, t2, t3 = trigram
        count_inserted = 0
        
        # Pattern: JJ -> NNC -> VP
        if t1[1] == "JJ" and t2[1] == "NNC" and t3[1] == "VP":
            # Replace adjective
            offensive_adj = random.choice(list(offensive_lexicon['JJ'].keys()))
            modified_tokens[i] = offensive_adj
            inserted_positions.add(i)
            
            # Replace verb
            offensive_verb = random.choice(list(offensive_lexicon['VP'].keys()))
            modified_tokens[i+2] = offensive_verb
            inserted_positions.add(i+2)
            
            count_inserted += 2
        return modified_tokens, inserted_positions, count_inserted

    def _handle_proper_noun_modification(
        self, 
        tokens: List[str], 
        trigram: Tuple[Tuple[str, str], ...], 
        i: int,
        modified_tokens: List[str],
        inserted_positions: Set[int],
        offensive_lexicon: Dict
    ) -> Tuple[List[str], Set[int], int]:
        t1, t2, t3 = trigram
        count_inserted = 0
        
        # Pattern: NNP -> NNP -> NNP
        if all(t[1] == "NNP" for t in [t1, t2, t3]):
            offensive_noun = random.choice(list(offensive_lexicon['NNP'].keys()))
            modified_tokens.insert(i+1, offensive_noun)
            inserted_positions.add(i+1)
            count_inserted += 1
        return modified_tokens, inserted_positions, count_inserted
    
    def _handle_punctuation_noun_pattern(
        self, 
        tokens: List[str], 
        trigram: Tuple[Tuple[str, str], ...], 
        i: int,
        modified_tokens: List[str],
        inserted_positions: Set[int],
        offensive_lexicon: Dict
    ) -> Tuple[List[str], Set[int], int]:
        t1, t2, t3 = trigram
        count_inserted = 0
        
        # Pattern: PUNC -> NNC -> PUNC
        if t1[1] == "PUNC" and t2[1] == "NNC" and t3[1] == "PUNC":
            # Randomly decide between inserting before, after, or both with 1/3 probability each
            choice = random.random()
            
            if choice < 1/3:
                # Insert offensive adjective before noun
                offensive_adj = random.choice(list(offensive_lexicon['JJ'].keys()))
                modified_tokens.insert(i+1, offensive_adj)
                inserted_positions.add(i+1)
                count_inserted += 1
            elif choice < 2/3:
                # Insert offensive noun after existing noun
                offensive_noun = random.choice(list(offensive_lexicon['NNC'].keys()))
                modified_tokens.insert(i+2, offensive_noun)
                inserted_positions.add(i+2)
                count_inserted += 1
            else:
                # Insert both before and after
                offensive_adj = random.choice(list(offensive_lexicon['JJ'].keys()))
                offensive_noun = random.choice(list(offensive_lexicon['NNC'].keys()))
                modified_tokens.insert(i+1, offensive_adj)
                modified_tokens.insert(i+2, offensive_noun)
                inserted_positions.add(i+1)
                inserted_positions.add(i+2)
                count_inserted += 2
            
        # Pattern: NPP -> NPP -> PUNC    
        elif t1[1] == "NPP" and t2[1] == "NPP" and t3[1] == "PUNC":
            choice = random.random()

            if choice < 1/3:
                # Insert offensive adjective before noun
                offensive_adj = random.choice(list(offensive_lexicon['JJ'].keys()))
                modified_tokens.insert(i+1, offensive_adj)
                inserted_positions.add(i+1)
                count_inserted += 1
            elif choice < 2/3:
                # Insert offensive noun after existing noun
                offensive_noun = random.choice(list(offensive_lexicon['NNC'].keys()))
                # j is random choice either 1 or 2
                j = random.choice([1, 2])
                modified_tokens.insert(i+j, offensive_noun)
                inserted_positions.add(i+j)
                count_inserted += 1
            else:
                # Insert both before and after
                offensive_adj = random.choice(list(offensive_lexicon['JJ'].keys()))
                offensive_noun = random.choice(list(offensive_lexicon['NNC'].keys()))
                modified_tokens.insert(i+1, offensive_adj)
                modified_tokens.insert(i+2, offensive_noun)
                inserted_positions.add(i+1)
                inserted_positions.add(i+2)
                count_inserted += 2
           
         
        return modified_tokens, inserted_positions, count_inserted

    def _handle_compound_verb_modification(
        self, 
        tokens: List[str], 
        trigram: Tuple[Tuple[str, str], ...], 
        i: int,
        modified_tokens: List[str],
        inserted_positions: Set[int],
        offensive_lexicon: Dict
    ) -> Tuple[List[str], Set[int], int]:
        t1, t2, t3 = trigram
        count_inserted = 0
        
        # Pattern: NCV -> RRPCV or JCV -> RRPCV
        if ((t1[1] == "NCV" and t2[1] == "RRPCV") or 
            (t1[1] == "JCV" and t2[1] == "RRPCV")):
            if t1[1] == "NCV" and 'NNC' in offensive_lexicon:
                offensive_word = random.choice(list(offensive_lexicon['NNC'].keys()))
            elif t1[1] == "JCV" and 'JJ' in offensive_lexicon:
                offensive_word = random.choice(list(offensive_lexicon['JJ'].keys()))
            modified_tokens[i] = offensive_word
            inserted_positions.add(i)
            count_inserted += 1
        return modified_tokens, inserted_positions, count_inserted

    def _handle_case_marker_insertion(
        self, 
        tokens: List[str], 
        trigram: Tuple[Tuple[str, str], ...], 
        i: int,
        modified_tokens: List[str],
        inserted_positions: Set[int],
        offensive_lexicon: Dict
    ) -> Tuple[List[str], Set[int], int]:
        t1, t2, t3 = trigram
        count_inserted = 0
        
        # Pattern: NNC -> CM or NNC -> POST
        if t1[1] == "NNC" and (t2[1] == "CM" or t2[1] == "POST"):
            if 'NNC' in offensive_lexicon:
                offensive_noun = random.choice(list(offensive_lexicon['NNC'].keys()))
                modified_tokens.insert(i+1, offensive_noun)
                inserted_positions.add(i+1)
                count_inserted += 1
        return modified_tokens, inserted_positions, count_inserted

    def handle_proper_noun_punctuation(
        self, 
        tokens: List[str], 
        trigram: Tuple[Tuple[str, str], ...], 
        i: int,
        modified_tokens: List[str],
        inserted_positions: Set[int],
        offensive_lexicon: Dict
    ) -> Tuple[List[str], Set[int], int]:
        t1, t2, t3 = trigram
        count_inserted = 0
        
        if t1[1] == "NNP" and t2[1] == "NNP" and t3[1] == "PUNC":
            # Insert offensive adjective between proper nouns
            if 'JJ' in offensive_lexicon:
                offensive_adj = random.choice(list(offensive_lexicon['JJ'].keys()))
                modified_tokens.insert(i+1, offensive_adj)
                inserted_positions.add(i+1)
                count_inserted += 1
                
        return modified_tokens, inserted_positions, count_inserted



