# multilingual-asr-syllables

The aim of this project is to build a multilingual ASR in which phonological syllables are considered as subwords. 

The ASR model used for the fine-tuning is Wavlm-large.

To achieve the purpose a syllabifier algorithm has been designed and implemented in the tokenizer of the ASR model.
Italian, Spanish and French were the languages considered for the monolingual fine-tunings and for the training of the multilingual model.

The subword inventory will be composed by phonological syllables, linguistics units that vehiculate acoustic information.
This should lead to stronger associations between audio frames and labels and facilitate the recognition. 
In addition, syllables are considered by the majority of scholars as a linguistic universal and their structure is largely shared among languages.
This aspect, in the perspective of multilingual ASR, translates into a synthetic subword vocabulary efficiently exploitable for more languages at the same time.


