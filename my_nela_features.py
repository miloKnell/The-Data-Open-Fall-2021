class FeatureFunctions(object):
    '''
    Functions for individual features or to support feature groups.
    '''
    # Constructor
    def __init__(self):
        # Compile our regular expressions
        for i in range(len(FALLBACK_SUBSYL)):
            FALLBACK_SUBSYL[i] = re.compile(FALLBACK_SUBSYL[i])
        for i in range(len(FALLBACK_ADDSYL)):
            FALLBACK_ADDSYL[i] = re.compile(FALLBACK_ADDSYL[i])

        # Read our syllable override file and stash that info in the cache
        for line in SPECIALSYLLABLES_EN.splitlines():
            line = line.strip()
            if line:
                toks = line.split()
                assert len(toks) == 2
                FALLBACK_CACHE[self._normalize_word(toks[0])] = int(toks[1])

    # Helper Functions
    def _normalize_word(self, word):
        return word.strip().lower()

    def get_filtered_words(self, tokens):
        special_chars = list(string.punctuation)
        filtered_words = []
        for tok in tokens:
            if tok in special_chars or tok == " ":
                continue
            else:
                new_word = "".join([c for c in tok if c not in special_chars])
                if new_word == "" or new_word == " ":
                    continue
                filtered_words.append(new_word)
        return filtered_words

    # Style Functions
    def LIWC(self, tokens):
        counts_dict = {k:0 for k in LIWC_CAT_DICT.keys()}
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(t) for t in tokens]
        for stem in LIWC_STEM_DICT:
            count = stemmed_tokens.count(stem.replace("*", ""))
            if count > 0:
                for cat in LIWC_STEM_DICT[stem]:
                    counts_dict[cat] += count
        counts_dict_norm_with_catnames = {LIWC_CAT_DICT[k]:float(c)/len(tokens) for (k,c) in counts_dict.items()}
        return counts_dict_norm_with_catnames

    def POS_counts(self, words):
        pos_tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT",
					"POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "WP$", "WRB", "VB", "VBD", "VBG",
					"VBN", "VBP", "VBZ", "WDT", "WP", "$", "''", "(", ")", ",", "--", ".", ":", "``"]
        tag_to_count = {t:0 for t in pos_tags} # init dict
        tagged_words = pos_tag(words)
        for word,tag in tagged_words: #count tags
            tag_to_count[tag] += 1
        tag_to_count_norm = {t:float(n)/len(words) for (t,n) in tag_to_count.items()} #normalize counts by num words in article
        return tag_to_count_norm

    def puncs_caps_stops(self, tokens):
        puncs = set(string.punctuation)
        quotes = float((tokens.count("\"") + tokens.count('``') + tokens.count("''"))) / len(tokens)
        exclaim = float(tokens.count("!")) / len(tokens)
        allpunc = 0
        for p in puncs:
        	allpunc += tokens.count(p)
        allpunc = float(allpunc) /  len(tokens)
        words_upper = 0
        words_upper = sum([1 for w in tokens if w.isupper()])
        allcaps = float(words_upper) / len(tokens)
        stopwords_eng = set(stopwords.words('english'))
        stops = float(len([s for s in tokens if s in stopwords_eng]))/len(tokens)
        return quotes, exclaim, allpunc, allcaps, stops

    # Complexity Functions
    def ttr(self, words):
        dif_words = len(set(words))
        tot_words = len(words)
        ttr = (float(dif_words) / tot_words)
        return ttr

    def count_syllables(self, word):
        word = self._normalize_word(word)
        if not word:
            return 0
        # Check for a cached syllable count
        count = FALLBACK_CACHE.get(word, -1)
        if count > 0:
            return count
        # Remove final silent 'e'
        if word[-1] == "e":
            word = word[:-1]
        # Count vowel groups
        count = 0
        prev_was_vowel = 0
        for c in word:
            is_vowel = c in ("a", "e", "i", "o", "u", "y")
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        # Add & subtract syllables
        for r in FALLBACK_ADDSYL:
            if r.search(word):
                count += 1
        for r in FALLBACK_SUBSYL:
            if r.search(word):
                count -= 1
        # Cache the syllable count
        FALLBACK_CACHE[word] = count
        return count

    def count_complex_words(self, tokens, sentences):
        words = tokens
        complex_words = 0
        found = False
        cur_word = []
        for word in words:
            if self.count_syllables(word)>= 3:
                #Checking proper nouns. If a word starts with a capital letter
                #and is NOT at the beginning of a sentence we don't add it
                #as a complex word.
                if not(word[0].isupper()):
                    complex_words += 1
                else:
                    for sentence in sentences:
                        if str(sentence).startswith(word):
                            found = True
                            break
                    if found:
                        complex_words += 1
                        found = False
        return complex_words

    def flesch_kincaid_grade_level(self, text, words, sentences):
        score = 0.0
        word_count = len(words)
        sentence_count = len(sentences)
        avg_words_p_sentence = word_count/sentence_count
        syllableCount = 0
        for word in words:
            syllableCount += self.count_syllables(word)
        if word_count > 0.0:
            score = 0.39 * (avg_words_p_sentence + 11.8 * (syllableCount/word_count)) - 15.59
        rounded_score = round(score, 4)
        return rounded_score

    def smog_index(self, text, words, sentences):
        score = 0.0
        word_count = len(words)
        sentence_count = len(sentences)
        complex_word_count = self.count_complex_words(words, sentences)
        if word_count > 0.0:
            score = (math.sqrt(complex_word_count*(30/sentence_count)) + 3)
        return score

    def coleman_liau_index(self, text, words, sentences):
        score = 0.0
        word_count = len(words)
        sentence_count = len(sentences)
        characters = 0
        for word in words:
            characters += len(word)
        if word_count > 0.0:
            score = (5.89*(characters/word_count))-(30*(sentence_count/word_count))-15.8
        rounded_score = round(score, 4)
        return rounded_score

    def lix(self, text, words, sentences):
        longwords = 0.0
        score = 0.0
        word_count = len(words)
        sentence_count = len(sentences)
        if word_count > 0.0:
            for word in words:
                if len(word) >= 7:
                    longwords += 1.0
            score = word_count / sentence_count + float(100 * longwords) / word_count
        return score

    # Affect Functions
    def vadersent(self, text): #dependent on vaderSentiment
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(text)
        return vs['neg'], vs['neu'], vs['pos']

    def acl_affect(self, words):
        wneg_count = float(sum([words.count(n) for n in ACL13_DICT['wneg']])) / len(words)
        wpos_count = float(sum([words.count(n) for n in ACL13_DICT['wpos']])) / len(words)
        wneu_count = float(sum([words.count(n) for n in ACL13_DICT['wneu']])) / len(words)
        sneg_count = float(sum([words.count(n) for n in ACL13_DICT['sneg']])) / len(words)
        spos_count = float(sum([words.count(n) for n in ACL13_DICT['spos']])) / len(words)
        sneu_count = float(sum([words.count(n) for n in ACL13_DICT['sneu']])) / len(words)
        return wneg_count, wpos_count, wneu_count, sneg_count, spos_count, sneu_count

    # Bias Functions
    def bias_words(self, words):
        bigrams = [" ".join(bg) for bg in ngrams(words, 2)]
        trigrams = [" ".join(tg) for tg in ngrams(words, 3)]
        bias = float(sum([words.count(b) for b in ACL13_DICT['bias_words']])) / len(words)
        assertatives = float(sum([words.count(a) for a in ACL13_DICT['assertatives']])) / len(words)
        factives = float(sum([words.count(f) for f in ACL13_DICT['factives']])) / len(words)
        hedges = sum([words.count(h) for h in ACL13_DICT['hedges']]) + \
            sum([bigrams.count(h) for h in ACL13_DICT['hedges']]) + \
            sum([trigrams.count(h) for h in ACL13_DICT['hedges']])
        hedges = float(hedges) / len(words)
        implicatives = float(sum([words.count(i) for i in ACL13_DICT['implicatives']])) / len(words)
        report_verbs = float(sum([words.count(r) for r in ACL13_DICT['report_verbs']])) / len(words)
        positive_op = float(sum([words.count(p) for p in ACL13_DICT['positive']])) / len(words)
        negative_op = float(sum([words.count(n) for n in ACL13_DICT['negative']])) / len(words)
        return bias, assertatives, factives, hedges, implicatives, report_verbs, positive_op, negative_op

    # Moral Functions
    def moral_foundations(self, words):
        foundation_counts_norm = {}
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(t) for t in words]
        for key in MORAL_FOUNDATION_DICT.keys():
        	foundation_counts_norm[key] = float(sum([stemmed_tokens.count(i) for i in MORAL_FOUNDATION_DICT[key]])) / len(words)
        return foundation_counts_norm

    # Event functions
    def get_continuous_NE_chunks(self, tokens):
        chunked = ne_chunk(pos_tag(tokens))
        continuous_chunk = []
        current_chunk = []
        for i in chunked:
         if hasattr(i, 'label'):
             if i.label() == 'GPE' or i.label() == 'LOC':
                 if type(i) == Tree:
                         current_chunk.append(" ".join([token for token, pos in i.leaves()]))
                 elif current_chunk:
                    named_entity = " ".join(current_chunk)
                    if named_entity not in continuous_chunk:
                        continuous_chunk.append(named_entity)
                        current_chunk = []
                 else:
                    continue
        norm_number_gpe_and_loc = float(len(continuous_chunk))/len(tokens)
        return norm_number_gpe_and_loc

    def count_dates(self, text, words):
        matches = list(datefinder.find_dates(text))
        norm_num_dates = float(len(matches))/len(words)
        return norm_num_dates

class NELAFeatureExtractor(object):
    '''
    Extract NELA features by group or all.
    '''
    # Constructor
    def __init__(self):
        self.Functions = FeatureFunctions()

    def extract_LIWC(self, tokens):
        normed_LIWC_count_dict = self.Functions.LIWC(tokens)
        return normed_LIWC_count_dict

    def extract_style(self, text, tokens=None, words=None, normed_LIWC_count_dict=None):
        if tokens == None:
            tokens = word_tokenize(text)
        if words == None:
            words = self.Functions.get_filtered_words(tokens)
        quotes, exclaim, allpunc, allcaps, stops = self.Functions.puncs_caps_stops(tokens)
        normed_POS_count_dict = self.Functions.POS_counts(words)
        if normed_LIWC_count_dict == None:
            normed_LIWC_count_dict = self.Functions.LIWC(tokens)
        liwc_feats_to_keep = ['funct', 'pronoun', 'ppron', 'i', 'we', 'you',
            'shehe', 'they', 'ipron', 'article', 'verb', 'auxverb', 'past',
            'past', 'future', 'adverb', 'preps', 'conj', 'negate', 'quant',
            'number', 'cogmech', 'insight', 'cause', 'discrep', 'incl',
            'excl', 'assent', 'nonfl', 'filler']
        # Liwc dictionary filter is only needed with keep large amounts for output
        liwc_count_dict_filt = {k:v for (k,v) in normed_LIWC_count_dict.items() if k in liwc_feats_to_keep}

        #build final vector and final names
        names = ['quotes', 'exclaim', 'allpunc', 'allcaps', 'stops']
        vect = [quotes, exclaim, allpunc, allcaps, stops]
        for (k,v) in normed_POS_count_dict.items():
            names.append(k)
            vect.append(v)
        for (k,v) in liwc_count_dict_filt.items():
            names.append(k)
            vect.append(v)
        return vect, names

    def extract_complexity(self, text, tokens=None, sentences=None, words=None):
        if tokens == None:
            tokens = word_tokenize(text)
        if sentences == None:
            sentences = sent_tokenize(text)
        if words == None:
            words = self.Functions.get_filtered_words(tokens)
        ttr = self.Functions.ttr(words)
        avg_wordlen = float(sum([len(w) for w in words]))/len(words)
        wc = len(words)
        fkgl = self.Functions.flesch_kincaid_grade_level(text, words, sentences)
        smog = self.Functions.smog_index(text, words, sentences)
        cli = self.Functions.coleman_liau_index(text, words, sentences)
        lix = self.Functions.lix(text, words, sentences)

        #build final vector and final names
        names = ['ttr', 'avg_wordlen', 'word_count',
            'flesch_kincaid_grade_level', 'smog_index', 'coleman_liau_index',
            'lix']
        vect = [ttr, avg_wordlen, wc, fkgl, smog, cli, lix]
        return vect, names

    def extract_bias(self, text, tokens=None, words=None, normed_LIWC_count_dict=None):
        if tokens == None:
            tokens = word_tokenize(text)
        if words == None:
            words = self.Functions.get_filtered_words(tokens)
        words = self.Functions.get_filtered_words(tokens)
        bias, assertatives, factives, hedges, implicatives, report_verbs, \
            positive_op, negative_op = self.Functions.bias_words(words)
        if normed_LIWC_count_dict == None:
            normed_LIWC_count_dict = self.Functions.LIWC(tokens)
        liwc_feats_to_keep = ['tentat', 'certain']
        # Liwc dictionary filter is only needed with keep large amounts for output
        #liwc_count_dict_filt = {k:v for (k,v) in normed_LIWC_count_dict.items() if k in liwc_feats_to_keep}

        #build final vector and final names
        names = ['bias_words', 'assertatives', 'factives', 'hedges',
            'implicatives', 'report_verbs', 'positive_opinion_words',
            'negative_opinion_words', 'tentat', 'certain']
        vect = [bias, assertatives, factives, hedges, implicatives, report_verbs,
            positive_op, negative_op, normed_LIWC_count_dict['tentat'],
            normed_LIWC_count_dict['certain']]
        return vect, names

    def extract_affect(self, text, tokens=None, words=None, normed_LIWC_count_dict=None):
        if tokens == None:
            tokens = word_tokenize(text)
        if words == None:
            words = self.Functions.get_filtered_words(tokens)
        vadneg, vadneu, vadpos = self.Functions.vadersent(text)
        wneg, wpos, wneu, sneg, spos, sneu = self.Functions.acl_affect(words)
        if normed_LIWC_count_dict == None:
            normed_LIWC_count_dict = self.Functions.LIWC(tokens)
        liwc_feats_to_keep = ['swear', 'affect', 'posemo', 'negemo', 'anx',
            'anger', 'sad']
        # Liwc dictionary filter is only needed with keep large amounts for output
        liwc_count_dict_filt = {k:v for (k,v) in normed_LIWC_count_dict.items() if k in liwc_feats_to_keep}

        #build final vector and final names
        names = ['vadneg', 'vadneu', 'vadpos', 'wneg', 'wpos', 'wneu', 'sneg',
        'spos', 'sneu', 'swear', 'affect', 'posemo', 'negemo', 'anx', 'anger',
        'sad']
        vect = [vadneg, vadneu, vadpos, wneg, wpos, wneu, sneg, spos, sneu]
        [vect.append(liwc_count_dict_filt[k]) for k in liwc_count_dict_filt.keys()]
        return vect, names

    def extract_moral(self, text, words=None):
        if words == None:
            tokens = word_tokenize(text)
            words = self.Functions.get_filtered_words(tokens)
        normed_moral_count_dict = self.Functions.moral_foundations(words)

        #build final vector and final names
        names = list(normed_moral_count_dict.keys())
        vect = list(normed_moral_count_dict.values())
        return vect, names

    def extract_event(self, text, tokens=None, words=None, normed_LIWC_count_dict=None):
        names = []
        if tokens == None:
            tokens = word_tokenize(text)
        if normed_LIWC_count_dict == None:
            normed_LIWC_count_dict = self.Functions.LIWC(tokens)
        if words == None:
            words = self.Functions.get_filtered_words(tokens)
        liwc_feats_to_keep = ['time']
        # Liwc dictionary filter is only needed with keep large amounts for output
        #liwc_count_dict_filt = {k:v for (k,v) in normed_LIWC_count_dict.items() if k in liwc_feats_to_keep}
        percent_GPE_and_LOC = self.Functions.get_continuous_NE_chunks(tokens)
        percent_dates = self.Functions.count_dates(text, words)

        #build final vector and final names
        names = ['time-words', 'num_locations', 'num_dates']
        vect = [normed_LIWC_count_dict['time'], percent_GPE_and_LOC, percent_dates]
        return vect, names

    def extract_all(self, text):
        '''
        Compute each feature group, merge computed vectors and feature names for
        each vector. The names list can be used to create headers in a csv or a db.
        '''
        # Pretokenize to speed up, compute LIWC before hand to speed up
        tokens = word_tokenize(text)
        sentences = sent_tokenize(text)
        words = self.Functions.get_filtered_words(tokens)
        normed_LIWC_count_dict = self.extract_LIWC(tokens)

        # Get each feature group
        svect, snames = self.extract_style(text, tokens, normed_LIWC_count_dict)
        cvect, cnames = self.extract_complexity(text, tokens, sentences)
        bvect, bnames = self.extract_bias(text, tokens, normed_LIWC_count_dict)
        avect, anames = self.extract_affect(text, tokens, words, normed_LIWC_count_dict)
        mvect, mnames = self.extract_moral(text, tokens)
        #evect, enames = self.extract_event(text, tokens, words, normed_LIWC_count_dict)

        # Produce final vector and names
        computed_vectors = [svect, cvect, bvect, avect, mvect]
        extracted_names = [snames, cnames, bnames, anames, mnames]
        vect = [item for sublist in computed_vectors for item in sublist]
        names = [item for sublist in extracted_names for item in sublist]
        return vect, names

if __name__ == "__main__":
    '''
    Example text and functions below
    '''
    newsarticle = "Ireland Expected To Become World's First Country To Divest \
    From Fossil Fuels The Republic of Ireland took a crucial step Thursday \
    toward becoming the first country in the world to divest from fossil fuels.\
    Lawmakers in the Dail, the lower house of parliament, advanced a bill \
    requiring the Irish government's more than $10 billion national investment \
    fund to sell off stakes in coal, oil, gas and peat  and to do so \
    \"as soon as practicable.\"The bill now heads to the upper chamber, \
    known as Seanad, where it is expected to pass easily when it's taken up, \
    likely in September."

    nela = NELAFeatureExtractor()

    #extract by group
    #feature_vector, feature_names = nela.extract_style(newsarticle) # <--- tested
    #feature_vector, feature_names = nela.extract_complexity(newsarticle) # <--- tested
    #feature_vector, feature_names = nela.extract_bias(newsarticle) # <--- tested
    #feature_vector, feature_names = nela.extract_affect(newsarticle) # <--- tested
    #feature_vector, feature_names = nela.extract_moral(newsarticle) # <--- tested
    #feature_vector, feature_names = nela.extract_event(newsarticle) # <--- tested

    #extract all groups
    feature_vector, feature_names = nela.extract_all(newsarticle) # <--- tested