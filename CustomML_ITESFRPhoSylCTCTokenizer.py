# coding=utf-8
# Copyright 2021 The Facebook Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization class for Wav2Vec2Phoneme."""

import json
import os
import sys
from itertools import groupby
from typing import Any, Dict, List, Optional, Tuple, Union
import re
from transformers.file_utils import requires_backends
from transformers.tokenization_utils import PreTrainedTokenizer, _insert_one_token_to_ordered_list
from transformers.tokenization_utils_base import AddedToken
from transformers.utils import (ModelOutput,
    is_flax_available,
    is_tf_available,
    is_torch_available,
    logging,
    requires_backends,
    to_py_obj,
)


logger = logging.get_logger(__name__)



VOCAB_FILES_NAMES = {   
    "vocab_file": "vocab.json",   
    "tokenizer_config_file": "tokenizer_config.json",  
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {   
        "facebook/wav2vec2-lv-60-espeak-cv-ft": "https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/resolve/main/vocab.json",
    },
    "tokenizer_config_file": {       
        "facebook/wav2vec2-lv-60-espeak-cv-ft": "https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/resolve/main/tokenizer_config.json",
    },
}

# Wav2Vec2Phoneme has no max input length
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/wav2vec2-lv-60-espeak-cv-ft": sys.maxsize}


class HybridML_ITESFRPhoSylCTCTokenizer(PreTrainedTokenizer):

    """
    Constructs a Wav2Vec2PhonemeCTC tokenizer.
    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.
    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sentence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        do_phonemize (`bool`, *optional*, defaults to `True`):
            Whether the tokenizer should phonetize the input or not. Only if a sequence of phonemes is passed to the
            tokenizer, `do_phonemize` should be set to `False`.
        phonemizer_lang (`str`, *optional*, defaults to `"en-us"`):
            The language of the phoneme set to which the tokenizer should phonetize the input text to.
        phonemizer_backend (`str`, *optional*. defaults to `"espeak"`):
            The backend phonetization library that shall be used by the phonemizer library. Defaults to `espeak-ng`.
            See the [phonemizer package](https://github.com/bootphon/phonemizer#readme). for more information.
        **kwargs
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        phone_delimiter_token=" ",
        word_delimiter_token=" | ",
        do_phonemize=False,   
        phonemizer_lang="it",  
        phonemizer_backend="espeak",
        do_wb_format = True, ### added function to get the text in the roght format
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            word_delimiter_token=word_delimiter_token,
            phone_delimiter_token=phone_delimiter_token,
            do_phonemize=do_phonemize, 
            phonemizer_lang=phonemizer_lang,
            phonemizer_backend=phonemizer_backend,
            do_wb_format = do_wb_format, ### added
            **kwargs,
        )

        self._word_delimiter_token = word_delimiter_token
        self._phone_delimiter_token = phone_delimiter_token
        self.do_phonemize = do_phonemize
        self.do_wb_format = do_wb_format ### added
        self.phonemizer_lang = phonemizer_lang
        self.phonemizer_backend = phonemizer_backend

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.decoder)

    def get_vocab(self) -> Dict:
        return dict(self.encoder, **self.added_tokens_encoder)

    def prepare_for_tokenization(
        self,
        text: str,
        is_split_into_words: bool = True,
        phonemizer_lang: Optional[str] = None,
        do_phonemize: Optional[bool] = None,
        do_wb_format:  bool = True, ### added
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Performs any necessary transformations before tokenization.
        This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the
        `kwargs` at the end of the encoding process to be sure all the arguments have been used.
        Args:
            text (`str`):
                The text to prepare.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            phonemizer_lang (`str`, *optional*):
                The language of the phoneme set to which the tokenizer should phonetize the input text to.
            do_phonemize (`bool`, *optional*):
                Whether the tokenizer should phonetize the input text or not. Only if a sequence of phonemes is passed
                to the tokenizer, `do_phonemize` should be set to `False`.
        Returns:
            `Tuple[str, Dict[str, Any]]`: The prepared text and the unused kwargs.
        """
        if is_split_into_words:
            text = " " + text

        

        if do_wb_format is not None:   ### added
            # print('entered wb_format')
            self.do_wb_format = do_wb_format

        # set whether tokenizer should phonemize or not

        if do_phonemize is not None:
            self.do_phonemize = do_phonemize

        # set the correct phonemizer language
        if phonemizer_lang is not None:
            self.phonemizer_lang = phonemizer_lang

        return (text, {})

    def _tokenize(self, text, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer.
        """

        # make sure whitespace is stripped to prevent <unk>
        text = text.strip()
        # print('_tokenize_funct: ', text)  

        # phonemize
        if self.do_phonemize:
          text = text.lower()

          # create list of phonemes
          text = self.phonemize(text, self.phonemizer_lang)

        if self.do_wb_format: ### added
          text = self.wb_format(text)

        # make sure ' ' is between phonemes
        tokens = text.split(" ")         
    
        tokens = list(filter(lambda p: p.strip() != "", tokens))
        return tokens


    def wb_format(self, text: str) -> str:  ### added function
      MLTnucleus = ['a', 'e', 'o', 'i', 'u', '@', 'E', 'O', 'y', '2', '9', 'A', 'e~', 'a~', 'o~', '9~']

      MLTphoClasses = {'e': 0, 'i': 0, 'o': 0,'u': 0, 'a': 0, '@': 0, 'E':0, 'O':0, 'y':0, '2':0, '9':0, # vowels
                 'A':0, 'e~':0, 'a~':0, 'o~':0, '9~':0, '~': 11,
                 'j': 1, 'w': 1, 'H':1,
                 'r': 2, 'R': 2,
                 'l': 3,'L': 3,
                 'm': 4, 'n': 4, 'J': 4, 'N': 4,
                 's': 5, 'z': 5,
                 'v': 6, 'f': 6,
                 'B': 7, 'D': 7, 'G': 7, 'Z': 7, 'S': 7, 'T': 7, 'x': 7, 'X': 7,
                 'ddz': 8, 'dz': 8, 'ts': 8, 'tts': 8, 'ddZ' : 8, 'dZ' : 8, 'tS' : 8, 'ttS' : 8,
                 'b': 9, 'd': 9, 'g': 9, 'k': 9, 'p': 9,'t': 9}
        

      # set the word separator

      listlist = [ ]

      # print('splitting sent') # words in list to avoid syllabification outside word boundaries

      splitted_sent = text.split(" ")

      word_delimiter_tok = self.word_delimiter_token + ' '
      pho_delimiter_tok = self.phone_delimiter_token

      #### EXAMPLE ------------------- 'nessun lavoro sErjo di applikattsjone E denari in abbondantsa'

      splitted_sent = text.split(pho_delimiter_tok)
      # print('splitted_sent:', splitted_sent)  ### --------------- ['nessun', 'lavoro', 'sErjo', 'di', 'applikattsjone', 'E', 'denari', 'in', 'abbondantsa']
        
      raw_MOP_sent = [ ] # splits after each vowel

      for item in splitted_sent: # sillabification starts here

        raw_MOP_w = ""

        for seg in item:
          if seg in MLTnucleus:
            raw_MOP_w += seg + ' '
          elif seg == '~':
            raw_MOP_w = raw_MOP_w[ :-1] + '~' + ' '
            
            # print('raw_MOP~',  raw_MOP_w)

          else:
            raw_MOP_w += seg
    
        raw_MOP_sent.append(raw_MOP_w.strip())

      # print('raw MOP___: ', raw_MOP_sent)   


      # syllables in list to check if they're valid or violate SSP

      for i, rawSyl_w in enumerate(raw_MOP_sent):
        raw_MOP_sent_list = rawSyl_w.split(pho_delimiter_tok)
        # print('word to check ', i, raw_MOP_sent_list) ### ---------------  0 ['ne', 'ssu', 'n']


        # SSP check

        """every segment is mappend into a class tht has a numeric ID according to the sonority.
        Onset segments of the syllables of the rough MOP tokenization are asigned with a sonority ID
        that is appended to a list; if the list matches with one of the allowed combinatios the syllable
        is valid (SSP = True), otherwise the problematic onsets that violates the SSP are appended as coda of the previous syllable"""

        ok_MOP_sent = [ ]


        SSP_allowed = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], # CV
                    [1, 0, 11], [2, 0, 11], [3, 0, 11], [4, 0, 11], [5, 0, 11], [6, 0, 11],  # CV~
                    [7, 0, 11], [8, 0, 11], [9, 0, 11], [5, 1, 0, 11], [5, 1, 0],            # CV~  
                    [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], # geminates
                    [9 ,9, 3] ,[9 ,9, 2], # geminates + liquids          
                    [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1],# everything + approximants
                    [5, 2], [6, 2], [7, 2], [9, 2], # fric plosives + /r/  # ------------------- ok 
                    [5, 3], [6, 3], [7, 3], [9, 3],  # fric, plosives + /l/ # ------------------- ok 
                    [5, 4], [7, 4], [9, 4],   # + nasali - con le plosive ci sono eccezioni tolto [8, 4] - pneumatico
                    [9, 5],  # + /s/ /z/ - con le plosive ci sono eccezioni psicologo
                    [7, 6], [7, 9, 2],  [8, 6],  # + /S/ /Z/ - con le plosive ci sono eccezioni 
                    [5, 9], [5, 6],  # s impura
                    [5, 6, 1],  [5, 6, 2], [5, 6, 3], [5, 9, 1],  [5, 9, 2], [5, 9, 3], # s impura + CC --------- ok
                    [5, 6, 2, 1], [5, 6, 3, 1], [5, 9, 2, 1], [5, 9, 3, 1], # s impura + CC + glide --------- ok
                    [9, 9, 5], [9, 9, 6], [9, 9, 5, 1], [9, 9, 6, 1], 
                      [9, 5, 9], [6, 9]] # per gestire affricate


        current_syl = '' 

        for i, syl in enumerate(raw_MOP_sent_list):
          seg_class = [ ]     
          onset = syl[:-1]

          if len(onset) > 1:
            for ch in onset:
              seg_class.append(MLTphoClasses[ch]) ###class del carattere
            
            if seg_class in SSP_allowed:
              MOP_ok = True
            else:
              MOP_ok = False 

          else:
            MOP_ok = True # if the onset is only one C it is always legit
          
          ok_MOP_sent.append([syl, MOP_ok])



        # print('ok_MOP_sent: ', ok_MOP_sent)    
                                              


        # fixing syl that violate the MOP


        final_syl_sent = [ ]

        current_syl = None

        for i, syl_MOP in enumerate(ok_MOP_sent):
          if syl_MOP[1] == True:
            current_syl = syl_MOP[0]
            final_syl_sent.append(current_syl) 

          if syl_MOP[1] == False:
            try:
              prev_syll = final_syl_sent[-1]
              probl = syl_MOP[0][0]  
              prev_syll = prev_syll+probl
              final_syl_sent.pop()
              final_syl_sent.append(prev_syll)

              ok_syl = syl_MOP[0].replace(probl, "")
              current_syl = ok_syl
              final_syl_sent.append(current_syl)
          
            except IndexError: # fixes if onset is unseparable consonants(exceptions, names)
                current_syl = syl_MOP[0]
                final_syl_sent.append(current_syl)

          
          
        if len(final_syl_sent[-1]) == 1 and final_syl_sent[-1] not in MLTnucleus: # fixes if last syllable is a consonant alone
          probl_coda = final_syl_sent[-1]
          try:
            final_syl_sent.pop()
            last_s = final_syl_sent[-1]
            final_syl_sent.pop()
            fixed_last = last_s + probl_coda
            final_syl_sent.append(fixed_last)
          except IndexError:
            # print('i have a problem with', probl_coda)
                if probl_coda in 'dl':  #
                  final_syl_sent.append(probl_coda) #
                else: 
                  final_syl_sent.append(probl_coda+'e') #
        

        if final_syl_sent[-1] not in 'dl' and not any(substring in final_syl_sent[-1] for substring in MLTnucleus):
          
          last_Csyl =  final_syl_sent[-1]
          # print(' ---------- CHECK PROBLEM: last_Csyl', last_Csyl) ##### added 3/01
          
          ### added -------------------

          try:

            final_syl_sent.pop()
            last_s = final_syl_sent[-1]

            final_syl_sent.pop()
            fixed_last = last_s + last_Csyl

            final_syl_sent.append(fixed_last)

          except IndexError:
            print('index error for :', last_Csyl)
            pass



        else:
          pass


        #print('final_syl_sent before vocab check____:', final_syl_sent)

        final_syl_hybr = [ ]

        for sillabified_w in final_syl_sent:
          if sillabified_w in self.encoder:
            #print('im a freq syllable', sillabified_w )
            final_syl_hybr.append(sillabified_w)
          else:
           # print('ops, gotta get the phonemes', sillabified_w)
            str_space = pho_delimiter_tok.join(sillabified_w)

            # let's manage the affricates to keep the chr together
            if "d d z" in str_space:
              str_space = str_space.replace("d d z", 'ddz')
            else:
              str_space = str_space
            if "d d Z"  in str_space:
              str_space = str_space.replace("d d Z",  'ddZ')
            else:
              str_space = str_space 
            if "d z"  in str_space:
              str_space = str_space.replace("d z",  'dz')
            else:
              str_space = str_space  
            if "d Z"  in str_space:
              str_space = str_space.replace("d Z",  'dZ')
            else:
              str_space = str_space  
            if "t t s"  in str_space:
              str_space = str_space.replace("t t s",  'tts')
            else:
              str_space = str_space  
            if "t t S"  in str_space:
              str_space = str_space.replace("t t S",  'ttS')
            else:
              str_space = str_space  

            if "t s"  in str_space:
              str_space = str_space.replace("t s",  'ts')
            else:
              str_space = str_space  
            if "t S"  in str_space:
              str_space = str_space.replace("t S",  'tS')
            else:
              str_space = str_space 


            if "a ~"  in str_space:
              str_space = str_space.replace("a ~",  'a~')
            else:
              str_space = str_space  
            if "o ~"  in str_space:
              str_space = str_space.replace("o ~",  'o~')
            else:
              str_space = str_space  

            if "e ~"  in str_space:
              str_space = str_space.replace("e ~",  'e~')
            else:
              str_space = str_space  
            if "9 ~"  in str_space:
              str_space = str_space.replace("9 ~",  '9~')
            else:
              str_space = str_space 

            final_syl_hybr.append(str_space)



        syl_w_str = pho_delimiter_tok.join(final_syl_hybr) # word as string with space-separated syllables
        #print('syl_w_str_________', syl_w_str)
        


      ## FINAL PART
        
        listlist.append(syl_w_str+' '+ word_delimiter_tok)
        
      #print('listlist___', listlist)
      text = ''.join(listlist)
      #print('tokenized_text___', text, type(text))
      return text



    def phonemize(self, text: str, phonemizer_lang: Optional[str] = None) -> str: ############################ this won't be called
        requires_backends(self, "phonemizer")

        from phonemizer import phonemize
        from phonemizer.separator import Separator

        word_delimiter = self.word_delimiter_token + " " if self.word_delimiter_token is not None else ""
        phonemizer_lang = phonemizer_lang if phonemizer_lang is not None else self.phonemizer_lang

        separator = Separator(phone=self.phone_delimiter_token, word=word_delimiter, syllable="")
        phonemes = phonemize(
            text,
            language=phonemizer_lang,
            backend=self.phonemizer_backend,
            separator=separator,
            language_switch="remove-flags",
        )
        phonemes = phonemes.strip()

        return phonemes

    @property
    def word_delimiter_token(self) -> str:
        """
        `str`: Word delimiter token. Log an error if used while not having been set.
        """
        if self._word_delimiter_token is None and self.verbose:
            return None
        return str(self._word_delimiter_token)

    @property
    def word_delimiter_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the word_delimiter_token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        if self._word_delimiter_token is None:
            return None
        return self.convert_tokens_to_ids(self.word_delimiter_token)

    @word_delimiter_token.setter
    def word_delimiter_token(self, value):
        self._word_delimiter_token = value

    @word_delimiter_token_id.setter
    def word_delimiter_token_id(self, value):
        self._word_delimiter_token = self.convert_tokens_to_ids(value)

    @property
    def phone_delimiter_token(self) -> str:
        """
        `str`: Word delimiter token. Log an error if used while not having been set.
        """
        if self._phone_delimiter_token is None and self.verbose:
            logger.error("Using phone_delimiter_token, but it is not set yet.")
            return None
        return str(self._phone_delimiter_token)

    @property
    def phone_delimiter_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the phone_delimiter_token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        if self._phone_delimiter_token is None:
            return None
        return self.convert_tokens_to_ids(self.phone_delimiter_token)

    @phone_delimiter_token.setter
    def phone_delimiter_token(self, value):
        self._phone_delimiter_token = value

    @phone_delimiter_token_id.setter
    def phone_delimiter_token_id(self, value):
        self._phone_delimiter_token = self.convert_tokens_to_ids(value)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an index (integer) using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        result = self.decoder.get(index, self.unk_token)
        return result

    def convert_tokens_to_string(
        self,
        tokens: List[str],
        group_tokens: bool = True,
        spaces_between_special_tokens: bool = False,
        filter_word_delimiter_token: bool = True,
    ) -> str:
        """
        Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
        """
        # group same tokens into non-repeating tokens in CTC style decoding
        if group_tokens:
            tokens = [token_group[0] for token_group in groupby(tokens)]

        # filter self.pad_token which is used as CTC-blank token
        filtered_tokens = list(filter(lambda token: token != self.pad_token, tokens))

        # also filter self.word_delimiter_token if not not
        if filter_word_delimiter_token and self.word_delimiter_token is not None:
            filtered_tokens = list(filter(lambda token: token != self.word_delimiter_token, filtered_tokens))

        string = " ".join(filtered_tokens).strip()

        return string

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        group_tokens: bool = True,
        filter_word_delimiter_token: bool = True,
        spaces_between_special_tokens: bool = False,
    ) -> str:
        """
        special _decode function is needed for Wav2Vec2PhonemeTokenizer because added tokens should be treated exactly
        the same as tokens of the base vocabulary and therefore the function `convert_tokens_to_string` has to be
        called on the whole token list and not individually on added tokens
        """
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        result = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            result.append(token)

        text = self.convert_tokens_to_string(
            result,
            group_tokens=group_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            filter_word_delimiter_token=filter_word_delimiter_token,
        )

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        return (vocab_file,)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary.
        Args:
            new_tokens (`List[str]`or `List[tokenizers.AddedToken]`):
                Token(s) to add in vocabulary. A token is only added if it's not already in the vocabulary (tested by
                checking if the tokenizer assign the index of the `unk_token` to them).
            special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the tokens should be added as special tokens.
        Returns:
            `int`: The number of tokens actually added to the vocabulary.
        Examples:
        ```python
        # Let's see how to increase the vocabulary of Bert model and tokenizer
        tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
        model = Wav2Vec2PhonemeForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
        num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
        print("We have added", num_added_toks, "tokens")
        # Note: resize_token_embeddings expects to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))
        ```"""
        new_tokens = [str(tok) for tok in new_tokens]

        tokens_to_add = []
        for token in new_tokens:
            if not isinstance(token, str):
                raise ValueError(f"Token {token} has to be of type string, but is " f"of type {type(token)}.")
            assert isinstance(token, str)
            if (
                token != self.unk_token
                and self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token)
                and token not in tokens_to_add
            ):
                tokens_to_add.append(token)
                if self.verbose:
                    logger.info(f"Adding {token} to the vocabulary")

        added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(tokens_to_add))
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.added_tokens_decoder.update(added_tok_decoder)

        # Make sure we don't split on any special tokens (even they were already in the vocab before)
        for token in tokens_to_add:
            if len(token) > 1:
                self._additional_special_tokens.append(AddedToken(token))
                _insert_one_token_to_ordered_list(self.unique_no_split_tokens, token)

        self._create_trie(self.unique_no_split_tokens)

        return len(tokens_to_add)