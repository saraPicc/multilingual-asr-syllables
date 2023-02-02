# Multilingual end-to-end ASR system with phonological syllables as subwords


The aim of this project is to build a multilingual ASR system in which phonological syllables are considered as subwords. 
There are multiple advantages of considering syllables as subwords: several patterns are observed cross-linguistically, meaning that languages share an important part of their syllable inventories; they vehiculate acoustic information, because the distribution of segments represents the variation of energy in the signal; moreover, they can be easily automatically generated according to the syllabification rules.

To build the ASR we fine-tune the model WavLM-large [(Chen et al., 2021)](https://arxiv.org/abs/2110.13900) with multilingual data extracted from the [Mozilla Common Voice](https://commonvoice.mozilla.org/it?gclid=Cj0KCQiA2-2eBhClARIsAGLQ2RlkVJtTFkEemoK3FvlpTxtFwuXvAHGOHadvXjzcbrx-R2Jw9eNdES8aAhcPEALw_wcB) dataset.

To obtain syllables as subwords we need to build a custom tokenizer based on the class [Wav2Vec2PhonemeCTCTokenizer](https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py#L94) that works according to the main syllabification rules, the [Sonority Sequencing Principle](http://www.ai.mit.edu/projects/dm/featgeom/clements90.pdf) and the [Maximal Onset Principle](https://dspace.mit.edu/handle/1721.1/16397).




