import re
import pickle as pkl
import random
import json
import pandas as pd

variants_nums =  ['('+str(el)+')' for el in range(11)]

def digraph_fix(str_space):

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

    if "e ~"  in str_space:
        str_space = str_space.replace("e ~",  'e~')
    else:
        str_space = str_space

    if "i ~"  in str_space:
        str_space = str_space.replace("i ~",  'i~')
    else:
        str_space = str_space
    if "o ~"  in str_space:
        str_space = str_space.replace("o ~",  'o~')
    else:
        str_space = str_space

    if "u ~"  in str_space:
        str_space = str_space.replace("u ~",  'u~')
    else:
        str_space = str_space

    if "9 ~"  in str_space:
        str_space = str_space.replace("9 ~",  '9~')
    else:
        str_space = str_space
    
    return str_space

def ref2phonseq(phonl_sent, dataset = True):

    """dataset True works on HF dataframe,
    dataset False works ok list of sentences"""

    sentPhoSeqL = [ ]

    if dataset == False:
        nospace = phonl_sent.replace(" ", "")
        str_space = nospace.replace("", " ")
        str_space = digraph_fix(str_space)
        sentPhoSeqL.append(str_space)
        sentPhoSeqL = sentPhoSeqL[0] 
        return sentPhoSeqL

    

    else:   
        for el in phonl_sent:
            nospace = el.replace(" ", "")
            str_space = nospace.replace("", " ")
            str_space = digraph_fix(str_space)
            sentPhoSeqL.append(str_space)

    
        return sentPhoSeqL

def checkLex(word, langdepLex):
    """Checks if a word is found in language dependent lexicon and updates the language ID counter. Called inside back2wordsMLT"""
    found = 0
    lang = langdepLex[:2].lower()
    with open(langdepLex, "rb") as read_file:
        lex = pkl.load(read_file)

    if word in lex:
        found =+1
    else:
        pass

    return found

# ---------------------------------------------  back2wordsIT

def back2wordsIT(decoded_str, voc = None,  graph = True):
    """Takes a decoded string and build back words in a graphemic
    representation if the grap parameter is True (firstly relying on
    an external lexicon and then on ortographical rules) or in phonemes
    if the parameter is set to False"""
    variants_nums =  ['('+str(el)+')' for el in range(11)]
  
    w_l = decoded_str.split('|')

    ok_words = []
    for el in w_l:
        word = el.replace(" ", "")
        ok_words.append(word) 

    if graph == False:  
        converted_sent = ' '.join(ok_words)
        return converted_sent

    else:
        transcribed = [ ]
        with open(voc, "rb") as read_file:
            voc = pkl.load(read_file)

        for word in ok_words:

            if word in voc:
                if len(voc[word]) > 1:
                    random_gr_w = random.choice(voc[word])
            
                else:
                    gr_w = voc[word][0] 
                    transcribed.append(gr_w)
                    
            else:
                x = word
                x = re.sub('ttSi', 'cci', x)         
                x = re.sub('ttSe', 'cce', x)
                x = re.sub('ttS', 'cci', x) 
                x = re.sub('tSi', 'ci', x)
                x = re.sub('tSe', 'cce', x)
                x = re.sub('tSE', 'cce', x)
                x = re.sub('ddZi', 'ggi', x)
                x = re.sub('dZi', 'gi', x)
                x = re.sub('ddZe', 'gge', x)
                x = re.sub('ddZE', 'gge', x)
                x = re.sub('dZe', 'ge', x)
                x = re.sub('dZE', 'ge', x)    
                x = re.sub('tS', 'ci', x)
                x = re.sub('dZ', 'gi', x)
                x = re.sub('tts', 'zz', x)
                x = re.sub('ddz', 'zz', x)
                x = re.sub('ts', 'zz', x)
                x = re.sub('dz', 'zz', x)
                x = re.sub('ki', 'chi', x)
                x = re.sub('ke', 'che', x)
                x = re.sub('ku', 'qu', x)
                x = re.sub('kw', 'qu', x)
                x = re.sub('SSe', 'sce', x)
                x = re.sub('SSE', 'sce', x)
                x = re.sub('SE', 'sce', x)
                x = re.sub('Se', 'sce', x)
                x = re.sub('LL', 'gl', x)
                x = re.sub('JJ', 'gn', x)

                if 'k' in x:
                    x = x.replace('k', 'c')
                
                if 'L' in x:
                    x = x.replace('L', 'gl')

                if 'N' in x:
                    x = x.replace('N', 'n')

                if 'S' in x:
                    x = x.replace('S', 'sci')

                if 'J' in x:
                    x = x.replace('J', 'gn')

                if 'j' in x:
                    x = x.replace('j', 'i')

                if 'w' in x:
                    x = x.replace('w', 'u')
            
                transcribed.append(x.lower())

        gr_sent = ' '.join(transcribed)


        # correzione d e l

        if " d " in gr_sent:    
            d_index = gr_sent.index(" d ")
            try:
                if gr_sent[d_index + 3] not in 'aeiou':
                    gr_sent = gr_sent.replace(" d ", " di ")
                else:
                    gr_sent = gr_sent.replace(" d ", " d' ")
                pass
            except IndexError:
                print(gr_sent)
                pass


        if " l' " in gr_sent:    
            l_index = gr_sent.index(" l' ")
            try:
                if gr_sent[l_index + 4] not in 'haeiou':
                    gr_sent = gr_sent.replace(" l' ", " il ")
                else:
                    pass
            except IndexError:
                print(gr_sent)
                pass


            # correzione spazi apostrofo

        if "'" in gr_sent:
            
            apo_index = gr_sent.index("'")
            if gr_sent[apo_index - 1] not in 'aeiou':
                gr_sent = gr_sent.replace("' ", "'")
            else:
                pass

        for el in variants_nums:
            if el in gr_sent:
                gr_sent = gr_sent.replace(el, "")
            else:
                pass

    return gr_sent

# ---------------------------------------------  back2wordsES

def back2wordsES(decoded_str, voc = None,  graph = True):

    variants_nums =  ['('+str(el)+')' for el in range(11)]

    """Takes a decoded string and build back words in a graphemic
    representation if the grap parameter is True (firstly relying on
    an external lexicon and then on ortographical rules) or in phonemes
    if the parameter is set to False"""

    LL_list = ['y', 'll']
    xe_list = ['ge', 'je']
    xi_list = ['gi', 'ji']
    dZ_list = ['j' , 'll', 'y']
    B_list = ['b', 'v']

    w_l = decoded_str.split('|')

    ok_words = []
    for el in w_l:
        word = el.replace(" ", "")
        ok_words.append(word) 
        
    if graph == False:  
        converted_sent = ' '.join(ok_words)
        return converted_sent

    else:
        transcribed = [ ]
        with open(voc, "rb") as read_file:
            voc = pkl.load(read_file)

        for word in ok_words:

            if word in voc:
                if len(voc[word]) > 1:
                    random_gr_w = random.choice(voc[word])
            
                else:
                    gr_w = voc[word][0] 
                    transcribed.append(gr_w)
            
            else:
                x = word

                if x[:2] == 'rr':
                    x = re.sub('rr', 'r', x)  # ok

                if 'z' in x:
                    x = x.replace('z', 's') # allofono

                x = re.sub('Te', 'ce', x) # ok
                x = re.sub('Ti', 'ci', x) # ok
                x = re.sub('Ta', 'za', x) # ok
                x = re.sub('To', 'zo', x) # ok
                x = re.sub('Tu', 'zu', x)   # ok
                x = re.sub('ttS', 'ch', x)  # ok
                x = re.sub('tS', 'ch', x)   # ok
                x = re.sub('xe', random.choice(xe_list), x) #ok
                x = re.sub('xi', random.choice(xi_list), x) #ok
                x = re.sub('xa', 'ja', x) # ok
                x = re.sub('xo', 'jo', x) # ok
                x = re.sub('xu', 'ju', x) # ok
                x = re.sub('ki', 'qui', x) # ok
                x = re.sub('ke', 'que', x) # ok
                x = re.sub('LL' , random.choice(LL_list), x) # ok
                x = re.sub('JJ', 'ñ', x) # ok

                x = re.sub('ddZe', 'ye', x)
                x = re.sub('dZe', 'ye', x)
                x = re.sub('dZ', random.choice(dZ_list), x)
                x = re.sub('tts', 'z', x) # parole straniere
                x = re.sub('ddz', 'z', x) # parole straniere
                x = re.sub('ts', 'z', x) # parole straniere
                x = re.sub('dz', 'z', x) # parole straniere

                x = re.sub('ku', 'cu', x)
                x = re.sub('kw', 'cu', x)
                x = re.sub('Gw', 'gu', x)


                x = re.sub('B' , random.choice(B_list), x)
                x = re.sub('gwj', 'güi', x)


                if 'D' in x:              # ok
                    x = x.replace('D', 'd')

                if 'G' in x:               # ok
                    x = x.replace('G', 'g')

                if 'k' in x:
                    x = x.replace('k', 'c')

                if 'L' in x:
                    x = x.replace('L',  random.choice(LL_list)) # ok

                if 'N' in x:
                    x = x.replace('N', 'n')

                if 'S' in x:
                    x = x.replace('S', 'sh')

                if 'J' in x:
                    x = x.replace('J', 'ñ')

                if 'j' in x:
                    x = x.replace('j', 'i')

                if 'w' in x:
                    x = x.replace('w', 'u')

                if 'Z' in x:
                    x = x.replace('Z', 'y') # argentino


                transcribed.append(x.lower())

        gr_sent = ' '.join(transcribed)

        for el in variants_nums:
            if el in gr_sent:
                gr_sent = gr_sent.replace(el, "")
            else:
                pass

    return gr_sent


  # ---------------------------------------------  back2wordsFR

def back2wordsFR(decoded_str, voc = None,  graph = True):

    variants_nums =  ['('+str(el)+')' for el in range(11)]
    """Takes a decoded string and build back words in a graphemic
    representation if the grap parameter is True (firstly relying on
    an external lexicon and then on ortographical rules) or in phonemes
    if the parameter is set to False"""

    w_l = decoded_str.split('|')

    ok_words = []

    for el in w_l:
        word = el.replace(" ", "")
        ok_words.append(word) 
        
    if graph == False:  
        converted_sent = ' '.join(ok_words)
        return converted_sent

    else:
        transcribed = [ ]
        with open(voc, "rb") as read_file:
            voc = pkl.load(read_file)

        for word in ok_words:

            if word in voc:
                if len(voc[word]) > 1:
                    random_gr_w = random.choice(voc[word])
                    transcribed.append(random_gr_w)
            
                else:
                    gr_w = voc[word][0] 
                    transcribed.append(gr_w)
            
            else:
                x = word

            # ----- approximative rules pho2graph ----- #

                nas_E = ['in', 'ain', 'en', 'aim'] # ok
                nas_o = ['on', 'om' ] # ok
                nas_a = [ 'em', 'en', 'an', 'am'] # ok
                or_o = ['o', 'au'] # ok
                or_a = ['a', 'e'] # ok - non in uso perche' la e e[ meno frequente ...?
                or_E = ['e', 'è' ,'ai', 'ei'] # ok
                fin_or_e = ['e','er', 'ez']
                fin_or_o = ['o','eau', 'eaux', 'eaux']
                fin_or_E = ['e', 'et', 'ait']
                fin_n_a = ['ent', 'an', 'ans']
                fin_n_o = ['ong', 'ont']
                nasE2 = ['in', 'en']
                nasau = ['on', 'om']
                

                cs_i = ['ci', 'si']
                cs_e = ['ce', 'se']

                # reducing word-initial geminates

                if len(x) > 1 and x[0] == x[1]:
                    x = re.sub(x[0:2], x[0], x)

                # vowels

                x = re.sub('y', 'u', x)
                x = re.sub('u', 'ou', x)

                if 'e' in x and 'e' == x[-1]:
                    x = re.sub('e', random.choice(fin_or_e), x)
                    x = re.sub('e', 'é', x)

                if 'o'  in x and 'o' == x[-1]:
                    x = re.sub('o', random.choice(fin_or_o), x)
                    x = re.sub('o', random.choice(or_o), x)

                x = re.sub('2', 'eu', x)

                x = re.sub('E', random.choice(or_E), x)
                x = re.sub('9', 'eu', x)
                x = re.sub('O', 'o', x)
                x = re.sub('A', 'â', x)
                x = re.sub('9~', 'un', x)
                x = re.sub('e~',  random.choice(nas_E), x)
                x = re.sub('@', 'eu', x) ### PROBLEM when is in phonetic chain?

                # ~ digraphs

                x = re.sub('si', random.choice(cs_i), x)
                x = re.sub('se', random.choice(cs_e), x)
                x = re.sub('ks', 'x', x)


                # others
                x = re.sub('wa', 'oi', x)
                x = re.sub('wâ', 'oi', x)

                x = re.sub('ki', 'qui', x)
                x = re.sub('ke', 'que', x)
                x = re.sub('ku', 'qu', x)

                x = re.sub('JJ', 'gn', x)

                x = re.sub('ddZ', 'j', x)
                x = re.sub('dZ', 'j', x)
                x = re.sub('tts', 'z', x)
                x = re.sub('ddz', 'z', x)
                x = re.sub('ts', 'z', x)
                x = re.sub('dz', 'z', x)
                x = re.sub('tS', 'ch', x)
                x = re.sub('ttS', 'ch', x)


                x = re.sub('kw', 'qu', x)
                x = re.sub('Gw', 'gu', x)

                x = re.sub('gwj', 'gui', x)

                # nasal vowels

                if 'a~' in x[-2: ]:
                    x = re.sub('a~', random.choice(fin_n_a), x)
                else:
                    x = re.sub('a~', random.choice(nas_a), x)

                if 'o~' in x[-2: ]:
                    x = re.sub('o~', random.choice(fin_n_o), x)
                else:
                    x = re.sub('o~', random.choice(nas_o), x)

                if 'k' in x:
                    x = x.replace('k', 'c')

                # glides

                if 'j' in x:
                    x = x.replace('j', 'i')

                if 'w' in x:
                    x = x.replace('w', 'ou')

                if 'H' in x:
                    x = x.replace('H', 'ue')


                # fricatives

                if 'Z' in x:
                    x = x.replace('Z', 'j')

                if 'S' in x:
                    x = x.replace('S', 'ch')

                if 'N' in x:
                    x = x.replace('N', 'n')

                if 'J' in x:
                    x = x.replace('J', 'gn')

                x = re.sub('é~', random.choice(nasE2), x)
                x = re.sub('au~', random.choice(nasau), x)
                x = re.sub('eu~', 'in', x)

                transcribed.append(x.lower())


        gr_sent = ' '.join(transcribed)

        # correzione d e l

        d_fr = ['de', 'des']

        if " d " in gr_sent:    
            d_index = gr_sent.index(" d ")
            try:
                if gr_sent[d_index + 3] not in 'haeiou':
                    gr_sent = gr_sent.replace(" d ", random.choice(d_fr))
                else:
                    gr_sent = gr_sent.replace(" d ", " d' ")
                    pass
            except IndexError:
                print(gr_sent)
                pass


        # correzione spazi apostrofo

        if "'" in gr_sent:
            apo_index = gr_sent.index("'")
            if gr_sent[apo_index - 1] not in 'haeiou':
                gr_sent = gr_sent.replace("' ", "'")
            else:
                pass

        # numeri di trascrizioni alternative che continuano a saltare fuori

        for el in variants_nums:
            if el in gr_sent:
                gr_sent = gr_sent.replace(el, "")
            else:
                pass

    return gr_sent



  # ---------------------------------------------  back2wordsPT

def back2wordsPT(decoded_str, voc = None,  graph = True):
  
    ambs = ['c', 's', 'x'] # conciso
    ambS = ['ch', 'x'] # conciso
    nase = ['en', 'em'] # ok
    ambz = ['z', 's', 'x', 'ç'] # ok
    ambZ = ['g', 'j'] # ok
    amba = ['á', 'à']


    w_l = decoded_str.split('|')

    ok_words = []

    for el in w_l:
        word = el.replace(" ", "")
        ok_words.append(word) 
        
    if graph == False:  
        converted_sent = ' '.join(ok_words)
        return converted_sent

    else:
        transcribed = [ ]
        with open(voc, "rb") as read_file:
            voc = pkl.load(read_file)

        for word in ok_words:

            if word in voc:
                if len(voc[word]) > 1:
                    random_gr_w = random.choice(voc[word])
            
                else:
                    gr_w = voc[word][0] 
                    transcribed.append(gr_w)

            else:
                x = word
                x = re.sub('tS', 'ch', x)     # ok    
                x = re.sub('ttS', 'ch', x)    # ok
                x = re.sub('L', 'lh', x)      # ok
                x = re.sub('LL', 'lh', x)     # ok
                x = re.sub('J', 'nh', x)      # ok
                x = re.sub('JJ', 'nh', x)     # ok
                x = re.sub('dZ', 'd', x)      # ok
                x = re.sub('O', 'ó', x) # ok
                x = re.sub('gi', 'gui', x)    # ok
                x = re.sub('ge', 'gue', x)    # ok
                x = re.sub('ku', 'cu', x)     # ok
                x = re.sub('kw', 'qu', x)     # ok
                x = re.sub('gw', 'gu', x)     # ok
                x = re.sub('s', random.choice(ambs), x)    # ok
                x = re.sub('S', random.choice(ambS), x)    # ok
                x = re.sub('z', random.choice(ambz), x)    # ok
                x = re.sub('Z', random.choice(ambZ), x)    # ok
                x = re.sub('e~',  random.choice(nase), x)     # ok
                x = re.sub('a~','ã', x)     # ok
                x = re.sub('o~','õ', x)     # ok
                x = re.sub('E','é', x)     # ok # la parte di diacritici e' complessa,

                # x = re.sub('e','ê', x)     # ok  per ora si lascia da parte
                # x = re.sub('i','í', x)     # ok
            
                if 'k' in x:
                    x = x.replace('k', 'c')          # ok
                
                if 'N' in x:                       # ok
                    x = x.replace('N', 'n')

                if 'j' in x:                       # ok
                    x = x.replace('j', 'i')

                if 'w' in x:                       # ok
                    x = x.replace('w', 'l')
            
                transcribed.append(x.lower())

        gr_sent = ' '.join(transcribed)

        for el  in gr_sent:
            if el == "(":
                par_index = gr_sent.index("(")
                x = par_index + 3
                nums = gr_sent[par_index : par_index + 3]
                gr_sent = gr_sent.replace(nums, "")
            else:
                pass

    return gr_sent

# ---------------------------------------------  back2wordsCA

def back2wordsCA(decoded_str, voc = None,  graph = True):

    # pho con + corrispondenze  grafemiche

    se_list = ['ce', 'se']
    si_list = ['ci', 'si']
    scA_list = ['sa', 'ça']
    scO_list = ['so', 'ço']
    scU_list = ['su', 'çu']
    Ze_list = [ 'gge', 'ge']
    Zi_list = [ 'ggi', 'gi']
    z_list = ['z', 's']

    w_l = decoded_str.split('|')

    ok_words = []

    for el in w_l:
        word = el.replace(" ", "")
        ok_words.append(word) 
        
    if graph == False:  
        converted_sent = ' '.join(ok_words)
        return converted_sent

    else:

        transcribed = [ ]

        with open(voc, "rb") as read_file:
            voc = pkl.load(read_file)

        for word in ok_words:

            if word in voc:

                if len(voc[word]) > 1:

                    random_gr_w = random.choice(voc[word])
                    transcribed.append(random_gr_w)

                else:
                    gr_w = voc[word][0]
                    transcribed.append(gr_w)


            else:
                x = word
                
                # vowels

                x = re.sub('E','e', x)
                x = re.sub('O', 'o', x) # ok
                x = re.sub('@', 'e', x) 

                x = re.sub('sa', random.choice(scA_list), x) # ok
                x = re.sub('so', random.choice(scO_list), x) # ok
                x = re.sub('su', random.choice(scU_list), x) # ok
                x = re.sub('se', random.choice(se_list), x) # ok
                x = re.sub('si', random.choice(si_list), x) # ok 
                x = re.sub('Ze', random.choice(Ze_list), x) # ok
                x = re.sub('Zi', random.choice(Zi_list), x) # ok
                x = re.sub('Z', 'j', x)    # ok          
                x = re.sub('tS', 'g', x)   # ok
                x = re.sub('ku', 'qu', x)  # ok
                x = re.sub('kw', 'qu', x)  # ok
                x = re.sub('LL', 'll', x)  # ok
                x = re.sub('L', 'll', x)  # ok
                x = re.sub('ki', 'qui', x) # ok
                x = re.sub('ke', 'que', x) # ok
                x = re.sub('S', 'x', x) # ok
                x = re.sub('JJ', 'ny', x) # ok
                x = re.sub('J', 'ny', x) # ok      

                if 'k' in x:   # ok
                    x = x.replace('k', 'c')

                if 'N' in x:
                    x = x.replace('N', 'n')

                if 'w' in x:
                    x = x.replace('w', 'u')


                transcribed.append(x.lower())

        gr_sent = ' '.join(transcribed)

        if " l' " in gr_sent:    
            l_index = gr_sent.index(" l' ")
            try:
                if gr_sent[l_index + 4] not in 'haeiou':
                    gr_sent = gr_sent.replace(" l' ", " il ")
                else:
                    pass
            except IndexError:
                print(gr_sent)
                pass


            # correzione spazi apostrofo

        if "'" in gr_sent:
            
            apo_index = gr_sent.index("'")
            if gr_sent[apo_index - 1] not in 'aeiou':
                gr_sent = gr_sent.replace("' ", "'")
            else:
                pass

        for el in variants_nums:
            if el in gr_sent:
                gr_sent = gr_sent.replace(el, "")
            else:
                pass

    return gr_sent


# ---------------------------------------------  back2wordsMLT

## this version works with ES, FR, IT
## languages can be added as long as there are a language dependent back2words function and lexicon

def back2wordsMLT(decoded_str, MLTlex = None, *LangDeplex,  graph = True, LID = False):
    """Establishes the language of the recognized sentence and relies to
    the correspondant lexicon to build the sentence in graphemes"""

    w_l = decoded_str.split('|')
    ok_words = []

    for el in w_l:
        word = el.replace(" ", "")
        ok_words.append(word) 
        
    if graph == False:  
        converted_sent = ' '.join(ok_words)
        return converted_sent
        
    else:
        if LID == True:
            with open(MLTlex, "rb") as read_file:
                vocMLT = pkl.load(read_file)

            for word in ok_words:

                # language ID identification

                langID = { }
                langs = [lang[:2].lower() for lang in LangDeplex]

                for lng in langs:
                    langID[lng] = 0

                for word in ok_words:
                    if word in vocMLT:
                        LID = vocMLT[word][0]
                        langID[LID] += 2         # if the word is found it is very probable that the sentence is in that language

                    else:
                        for ldepLex in LangDeplex:
                            lang = ldepLex[:2].lower()
                            LIDcount = checkLex(word, ldepLex)
                            langID[lang] += LIDcount

            predLang = max(langID, key=langID.get)

                # language is known, let's call the correspondant back2words function

            if predLang == 'it':
                grapheme_sent = back2wordsIT(decoded_str, 'ITpho2gr.pkl')

            if predLang == 'es':
                grapheme_sent = back2wordsES(decoded_str, 'ESpho2gr.pkl')

            if predLang == 'fr':
                grapheme_sent = back2wordsFR(decoded_str, 'FRpho2gr.pkl')

            return grapheme_sent

        else:
            print('working with indexes')



# ---------------------------------------------  GRAPHEME2PHONEME 

# ---------------------------------------------  gr2phoIT

def gr2phoIT(decoded_str, voc = None):
  
  transcribed = [ ]

  words_in_list = decoded_str.split(' ')     # (1) splitta parole in stringa secondo spazio 


  with open(voc, "rb") as read_file:       # (2) qui per chiamare la variabile, uncomment later 
      voc = pkl.load(read_file)

  for word in words_in_list:                # (3) controlla se la parola e' tra i values (graph) del dic

    if any(word in v for v in voc.values()):
        # print('w in voc val list!')
        gr_w = [i for i in voc if word in voc[i]]  # (4) appende a lista trascrizione k se word in v > 1
        # print('transcr_w', gr_w)

        transcribed.append(gr_w)

    elif [word] in voc.values():         # (5) appende a lista trascrizione k se word == v
        # print('w in voc unique!')
        gr_w = [i for i in voc if voc[i]== [word]]
        # print('transcr_w', gr_w)

        transcribed.append(gr_w)

          
    else:

      x = word
      x = re.sub('cci','ttSi',  x)         
      x = re.sub('cce','ttSe',  x)
      x = re.sub('cci','ttS',  x) 
      x = re.sub('ci','tSi',  x)
      x = re.sub('cce','tSe',  x)
      x = re.sub('cce', 'tSE', x)
      x = re.sub('ggi','ddZi',  x)
      x = re.sub('gi','dZi',  x)
      x = re.sub('gge','ddZe',  x)
      x = re.sub('gge','ddZE',  x)
      x = re.sub( 'ge','dZe', x)
      x = re.sub('ge','dZE',  x)    
      x = re.sub('ci','tS',  x)
      x = re.sub('gi','dZ', x)
      x = re.sub('zz','tts',  x)
      x = re.sub('zz','ddz',  x)
      x = re.sub('zz','ts',  x)
      x = re.sub('zz','dz',  x)
      x = re.sub('chi','ki',  x)
      x = re.sub('che','ke',  x)
      x = re.sub('qu','ku',  x)
      x = re.sub('qu','kw',  x)
      x = re.sub('sce','SSe',  x)
      x = re.sub('sce','SSE',  x)
      x = re.sub('sce','SE',  x)
      x = re.sub('sce','Se',  x)
      x = re.sub('gli','LL',  x)
      x = re.sub('gn','JJ',  x)
      x = re.sub('nc','Nk',  x)
      x = re.sub('ng','Ng',  x)
      x = re.sub('ia','ja',  x)       # problematico
      x = re.sub('io','jo',  x)
      x = re.sub('iu','ju',  x)
      x = re.sub('ua','wa',  x)
      x = re.sub('uo','wo',  x)  
      x = re.sub('ue','we',  x)
      x = re.sub('ui','wi',  x)     # problematico
      x = re.sub('ò','o',  x)
      x = re.sub('à','a',  x)
      x = re.sub('é','e',  x)
      x = re.sub('è','E',  x)
      x = re.sub('ù','u',  x)
      x = re.sub('ì','i',  x)
      x = re.sub('h','',  x)

      if 'c' in x:
        x = x.replace('c', 'k')
      
      if 'gl' in x:
        x = x.replace('gl', 'L')

      if 'sci' in x:
        x = x.replace('sci', 'S')

      if 'gn' in x:
        x = x.replace('gn', 'J')

      if "d'" in x:
        x = x.replace("d'", 'd')

      if "l'" in x:
        x = x.replace("l'", 'l')

      # print(x)
      transcribed.append(x)

  # print(transcribed)
  defintive_pho_tr = []
  for el in transcribed:
    if type(el) == list:
      pho_word = el[0]
      defintive_pho_tr.append(pho_word)
    else:
      pho_word = el
      defintive_pho_tr.append(pho_word)

     
  pho_sent = ' '.join(defintive_pho_tr)
  

  return pho_sent


# ---------------------------------------------  gr2phoES

def gr2phoES(decoded_str, voc = None):
 
  transcribed = [ ]
  ye_list = ['je' , 'ddZe', 'LLe']
  yo_list = ['jo' , 'ddZo', 'LLo']

  words_in_list = decoded_str.split(' ')     # (1) splitta parole in stringa secondo spazio

  with open(voc, "rb") as read_file:       # (2) qui per chiamare la variabile, uncomment later 
      voc = pkl.load(read_file)

  for word in words_in_list:                # (3) controlla se la parola e' tra i values (graph) del dic


    if any(word in v for v in voc.values()):
        # print('w in voc val list!')
        gr_w = [i for i in voc if word in voc[i]]  # (4) appende a lista trascrizione k se word in v > 1
        # print('transcr_w', gr_w)

        transcribed.append(gr_w)

    elif [word] in voc.values():         # (5) appende a lista trascrizione k se word == v
        # print('w in voc unique!')
        gr_w = [i for i in voc if voc[i]== [word]]
        # print('transcr_w', gr_w)

        transcribed.append(gr_w)

          
    else:

      x = word

      # if 'z' in x:
      #   x = x.replace('z', 's') # allofono

      x = re.sub('ce', 'Te',  x) # ok
      x = re.sub('ci','Ti',  x) # ok
      x = re.sub('za','Ta',  x) # ok
      x = re.sub('zo', 'To', x) # ok
      x = re.sub('zu','Tu',  x)   # ok
      x = re.sub('ch','ttS', x)  # ok
      x = re.sub('ch','tS',  x)   # ok
      x = re.sub('ge','xe',  x)   # ok
      x = re.sub('je','xe',  x)   # ok
      x = re.sub('gi','xi',  x)   # ok
      x = re.sub('ji','xi',  x)   # ok
      x = re.sub('ja','xa', x) # ok
      x = re.sub('jo','xo',  x) # ok
      x = re.sub('ju','xu',  x) # ok
      x = re.sub('qui','ki',  x) # ok
      x = re.sub('que','ke',  x) # ok
      x = re.sub('ll','LL', x) # ok
      x = re.sub('ñ','JJ',  x) # ok

      x = re.sub('ye', random.choice(ye_list), x)
      x = re.sub('z', 's', x) # parole straniere

      x = re.sub('cu','ku',  x)
      x = re.sub('cu','kw',  x)
      x = re.sub('gu','Gw',  x)
      x = re.sub('v' , 'B', x)
      x = re.sub('b' , 'B', x)
      x = re.sub('güi','gwj',  x)
      x = re.sub('d' , 'D', x)
      x = re.sub('g' , 'G', x)
      x = re.sub('c' , 'k', x)
      x = re.sub('nc','Nk',  x)
      x = re.sub('ng','Ng',  x)
      x = re.sub('j' , 'i', x)
      x = re.sub('ñ' , 'J', x)
      x = re.sub('ua','wa',  x)
      x = re.sub('uo','wo',  x)  
      x = re.sub('ue','we',  x)
      x = re.sub('ui','wi',  x)     # problematico
      x = re.sub('á','a',  x) 
      x = re.sub('é','e',  x) 
      x = re.sub('í','i',  x) 
      x = re.sub('ó','o',  x) 
      x = re.sub('ú','u',  x) 
  
      transcribed.append(x)

  # print(transcribed)
  defintive_pho_tr = []
  for el in transcribed:
    if type(el) == list:
      pho_word = el[0]
      defintive_pho_tr.append(pho_word)
    else:
      pho_word = el
      defintive_pho_tr.append(pho_word)

     
  pho_sent = ' '.join(defintive_pho_tr)
  

  return pho_sent

# ---------------------------------------------  gr2phoFR

def gr2phoFR(decoded_str, voc = None):
  
  transcribed = [ ]

  words_in_list = decoded_str.split(' ')     # (1) splitta parole in stringa secondo spazio

  with open(voc, "rb") as read_file:       # (2) qui per chiamare la variabile, uncomment later 
      voc = pkl.load(read_file)

  for word in words_in_list:                # (3) controlla se la parola e' tra i values (graph) del dic
    if any(word in v for v in voc.values()):
        # print('w in voc val list!')
        gr_w = [i for i in voc if word in voc[i]]  # (4) appende a lista trascrizione k se word in v > 1
        # print('transcr_w', gr_w)

        transcribed.append(gr_w)

    elif [word] in voc.values():         # (5) appende a lista trascrizione k se word == v
        # print('w in voc unique!')
        gr_w = [i for i in voc if voc[i]== [word]]
        # print('transcr_w', gr_w)

        transcribed.append(gr_w)
          
    else:

      x = word

      # vowels

      # x = re.sub('y', 'u', x) # problematico
      x = re.sub('u', 'ou', x)


      if 'er' in x and 'er' == x[ :-2]:
        x = re.sub('er', 'e', x)
      if 'ez' in x and 'ez' == x[ :-2]:
        x = re.sub('ez', 'e', x)
   

      x = re.sub('eau', 'o', x)
      x = re.sub('eaux', 'o', x)
      x = re.sub('au', 'o', x)
  
      x = re.sub('eu', '2', x)

      x = re.sub('è', 'E', x)
      x = re.sub('é','e',  x)
      x = re.sub('ai', 'E', x)
      x = re.sub('ei', 'E', x)
      x = re.sub('eu','9',  x)
  

      x = re.sub('un','9~',  x)
      
      x = re.sub('in','e~', x)
      x = re.sub('ain','e~', x)
      x = re.sub('en','e~', x)
      x = re.sub('aim','e~', x)
      x = re.sub('eu', '@', x) ### PROBLEM when is in phonetic chain?


      x = re.sub('ci', 'si', x) 
      x = re.sub('ce', 'se', x) 
      x = re.sub('se', 'se', x) 
      x = re.sub('x','ks',  x)


      x = re.sub('oi','wa',  x)
      x = re.sub('oi','wâ',  x)

      x = re.sub( 'qui','ki', x)
      x = re.sub( 'que','ke', x)
      x = re.sub('qu', 'ku',  x)


      x = re.sub('j','ddZ',  x)
      x = re.sub('j','dZ',  x)
      x = re.sub('z','tts',  x)
      x = re.sub('z', 'ddz', x)
      x = re.sub('z','ts',  x)
      x = re.sub('z','dz',  x)
      x = re.sub('ch', 'tS', x)
      x = re.sub('ch','ttS',  x)

      x = re.sub('qu','kw',  x)
      x = re.sub('gu','Gw',  x)

      x = re.sub('gui','gwj',  x)

      # nasal vowels

      x = re.sub('ent','a~',  x)
      x = re.sub('an','a~',  x)
      x = re.sub('am','a~',  x)
      x = re.sub('en','a~',  x)
      x = re.sub('em','a~',  x)
      x = re.sub('ans','a~',  x)

      x = re.sub('on','o~',  x)
      x = re.sub('om','o~',  x)
      x = re.sub('ong','o~',  x)
      x = re.sub('ont','o~',  x)

      x = re.sub('in','e~',  x)
      x = re.sub('en','e~',  x)
      x = re.sub('ait','E',  x)
      x = re.sub('et','E',  x)
      x = re.sub('ai','E',  x)
      x = re.sub('ei','E',  x)

      
      x = re.sub('c','k',  x)


      x = re.sub('ou','w',  x)
 
      x = re.sub('ue', 'H', x)
      x = re.sub('j', 'Z', x)
      x = re.sub('ch', 'S', x)

      x = re.sub('nc','Nk',  x)
      x = re.sub('ng','Ng',  x)
      x = re.sub('gn','J',  x)
      x = re.sub('des', 'de', x)
      x = re.sub('ia','ja',  x)       # problematico
      x = re.sub('io','jo',  x)
      x = re.sub('iu','ju',  x)
      x = re.sub('ua','wa',  x)
      x = re.sub('uo','wo',  x)  
      x = re.sub('ue','we',  x)
      x = re.sub('ui','wi',  x)     # problematico
      x = re.sub('ò','o',  x)
      x = re.sub('à','a',  x)
      x = re.sub('é','e',  x)
      x = re.sub('è','E',  x)
      x = re.sub('ù','u',  x)
      x = re.sub('ì','i',  x)
      x = re.sub('â','a',  x)
      x = re.sub('h','',  x)

      if "d'" in x:
        x = x.replace("d'", 'd')

      if "l'" in x:
        x = x.replace("l'", 'l')

      transcribed.append(x.lower())


  # print(transcribed)
  defintive_pho_tr = []
  for el in transcribed:
    if type(el) == list:
      pho_word = el[0]
      defintive_pho_tr.append(pho_word)
    else:
      pho_word = el
      defintive_pho_tr.append(pho_word)

     
  pho_sent = ' '.join(defintive_pho_tr)
  

  return pho_sent

# ---------------------------------------------  gr2phoPT

def gr2phoPT(decoded_str, voc = None):
  
  schwa = ['E', 'O', 'a']
  

  transcribed = [ ]

  words_in_list = decoded_str.split(' ')     # (1) splitta parole in stringa secondo spazio

  with open(voc, "rb") as read_file:       # (2) qui per chiamare la variabile, uncomment later 
      voc = pkl.load(read_file)

  for word in words_in_list:                # (3) controlla se la parola e' tra i values (graph) del dic

    if any(word in v for v in voc.values()):
        # print('w in voc val list!')
        gr_w = [i for i in voc if word in voc[i]]  # (4) appende a lista trascrizione k se word in v > 1
        # print('transcr_w', gr_w)

        transcribed.append(gr_w)

    elif [word] in voc.values():         # (5) appende a lista trascrizione k se word == v
        # print('w in voc unique!')
        gr_w = [i for i in voc if voc[i]== [word]]
        # print('transcr_w', gr_w)

        transcribed.append(gr_w)

          
    else:

      x = word
        
      x = re.sub('ch','ttS', x)    # ok
      x = re.sub('ch','S',  x)     # ok  
      x = re.sub('lh','LL',  x)      # ok
      x = re.sub('lh','L',  x)     # ok
      x = re.sub('nh','JJ',  x)     # ok
      x = re.sub('nh','J',  x)      # ok      
      x = re.sub('d','dZ',  x)      # ok
      x = re.sub('ó','O',  x) # ok
      x = re.sub('gui','gi',  x)    # ok
      x = re.sub('gue','ge',  x)    # ok
      x = re.sub('cu','ku',  x)     # ok
      x = re.sub('qu','kw',  x)     # ok
      x = re.sub('gu','gw',  x)     # ok
      x = re.sub('ce', 'se',  x) # ok
      x = re.sub('ci','si',  x) # ok
      x = re.sub('ge', 'Ze',  x) # ok
      x = re.sub('gi','Zi',  x) # ok

      x = re.sub('ç','s',  x) # ok
      x = re.sub('x','S',  x) # ok
      x = re.sub('en','e~',  x)
      x = re.sub('em','e~',  x)

      x = re.sub('ã','a~', x)     # ok
      x = re.sub('õ','o~', x)     # ok
      x = re.sub('é','E', x)     # ok # la parte di diacritici e' complessa,
      x = re.sub('á','a',  x) # ok
      x = re.sub('à','a',  x) # ok
      x = re.sub('ê','e', x)     # ok  per ora si lascia da parte
      x = re.sub('í','i', x)     # ok
     
      x = re.sub('c' , 'k', x)
      x = re.sub('nc','Nk',  x)
      x = re.sub('ng','Ng',  x)
      # x = re.sub('ia','ja',  x)       # problematico
      x = re.sub('io','jo',  x)
      x = re.sub('iu','ju',  x)
      x = re.sub('ua','wa',  x)
      x = re.sub('uo','wo',  x)  
      x = re.sub('ue','we',  x)
      x = re.sub('ui','wi',  x)     # problematico
     
      
      # problema con l vocalizzata a fine sillaba

      # print(x)
      transcribed.append(x)

  # print(transcribed)
  defintive_pho_tr = []
  for el in transcribed:
    if type(el) == list:
      pho_word = el[0]
      defintive_pho_tr.append(pho_word)
    else:
      pho_word = el
      defintive_pho_tr.append(pho_word)

     
  pho_sent = ' '.join(defintive_pho_tr)
  

  return pho_sent



# ---------------------------------------------  gr2phoCA

def gr2phoCA(decoded_str, voc = None):
  

  
  transcribed = [ ]

  words_in_list = decoded_str.split(' ')     # (1) splitta parole in stringa secondo spazio

  with open(voc, "rb") as read_file:       # (2) qui per chiamare la variabile, uncomment later 
      voc = pkl.load(read_file)

  for word in words_in_list:                # (3) controlla se la parola e' tra i values (graph) del dic

    if any(word in v for v in voc.values()):
        # print('w in voc val list!')
        gr_w = [i for i in voc if word in voc[i]]  # (4) appende a lista trascrizione k se word in v > 1
        # print('transcr_w', gr_w)

        transcribed.append(gr_w)

    elif [word] in voc.values():         # (5) appende a lista trascrizione k se word == v
        # print('w in voc unique!')
        gr_w = [i for i in voc if voc[i]== [word]]
        # print('transcr_w', gr_w)

        transcribed.append(gr_w)

          
    else:

      schwa_e = ['e', '@']
      schwa_o = ['o',  '@']

      x = word
        
      # x = re.sub('E','e', x) si sacrifica il contrasto aperta-chiusa
      # x = re.sub('O', 'o', x) si sacrifica il contrasto aperta-chiusa
 

      x = re.sub('dj','ddZ', x)
      x = re.sub('gge','dZe', x)
      x = re.sub('gi','Zi', x)
      x = re.sub('ggi','dZi', x)
      x = re.sub('ç','s', x)
      x = re.sub('ce','se', x)
      x = re.sub('ci','si', x)
     
      x = re.sub('e', random.choice(schwa_e), x) # problematico per acenti non controllati 
      x = re.sub('o', random.choice(schwa_o), x) # ok 
   

      x = re.sub('qui','ki',  x) # ok
      x = re.sub('que','ke',  x) # ok
      x = re.sub('gi','Zi',  x)   # ok
      x = re.sub('ja','Za',  x)       
      x = re.sub('jo','Zo',  x)
      x = re.sub('ju','Zu',  x)

      x = re.sub('qu','ku',  x)  # ok
      x = re.sub('qu','kw', x)  # ok
      x = re.sub('ll','LL',  x)  # ok
      x = re.sub('ll','L',  x)  # ok
      
      x = re.sub('x','S',  x) # ok
      x = re.sub('ny','JJ',  x) # ok
      x = re.sub('ny', 'J', x) # ok      

      x = re.sub('c' , 'k', x)
      x = re.sub('nc','Nk',  x)
      x = re.sub('ng','Ng',  x)
     
      x = re.sub('ia','ja',  x)       # problematico
      x = re.sub('io','jo',  x)
      x = re.sub('iu','ju',  x)
      x = re.sub('ua','wa',  x)
      x = re.sub('uo','wo',  x)  
      x = re.sub('ue','we',  x)
      x = re.sub('ui','wi',  x)     # problematico
      x = re.sub('ò','o',  x)
      x = re.sub('à','a',  x)
      x = re.sub('é','e',  x)
      x = re.sub('è','E',  x)
      x = re.sub('ù','u',  x)
      x = re.sub('ì','i',  x)
      x = re.sub('ï','i',  x)
      x = re.sub('h','',  x)
      x = re.sub("''",'',  x)
     


      # print(x)
      transcribed.append(x)

  # print(transcribed)
  defintive_pho_tr = []
  for el in transcribed:
    if type(el) == list:
      pho_word = el[0]
      defintive_pho_tr.append(pho_word)
    else:
      pho_word = el
      defintive_pho_tr.append(pho_word)

     
  pho_sent = ' '.join(defintive_pho_tr)
  

  return pho_sent