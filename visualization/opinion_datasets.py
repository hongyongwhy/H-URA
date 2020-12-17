import xml.etree.ElementTree as ET
from collections import Counter


class Datasets(object):

    def SemEval2014(self, _domain, _dataset):

        if _domain == 'laptop':
            file_name = 'Data/Laptop_Train_v2.xml' if _dataset == 'train' else 'Data/Laptops_test_Gold.xml'
            root = ET.parse(file_name).getroot()
            sentences = []
            for sent in root:
                sent_id = sent.attrib['id']
                sent_text = ''
                aspect_terms = []
                for child in sent.getchildren():
                    if child.tag == 'text':
                        sent_text = child.text
                    elif child.tag == 'aspectTerms':
                        for aspect in child:
                            aspect_terms.append([aspect.attrib['term'], aspect.attrib['from'], aspect.attrib['to'], aspect.attrib['polarity']])
                        # aspect_terms = [[aspect.attrib['term'], aspect.attrib['from'], aspect.attrib['to'], aspect.attrib['polarity']] for aspect in child]
                sentences.append([sent_id, sent_text, aspect_terms])
            return sentences

        elif _domain == 'restaurant':
            file_name = 'Data/Restaurants_Train_v2.xml' if _dataset == 'train' else 'Data/Restaurants_test_Gold.xml'
            root = ET.parse(file_name).getroot()
            sentences = []
            for idx, sent in enumerate(root):
                sent_id = sent.attrib['id']
                sent_text = ''
                aspect_terms, category_terms = [], []
                for child in sent.getchildren():
                    if child.tag == 'text':
                        sent_text = child.text
                    elif child.tag == 'aspectTerms':
                        for aspect in child:
                            aspect_terms.append([aspect.attrib['term'], aspect.attrib['from'], aspect.attrib['to'], aspect.attrib['polarity']])
                        # aspect_terms = [[aspect.attrib['term'], aspect.attrib['from'], aspect.attrib['to'], aspect.attrib['polarity']] for aspect in child]
                    elif child.tag == 'aspectCategories':
                        for aspect in child:
                            category_terms.append([aspect.attrib['category'], aspect.attrib['polarity']])
                        # category_terms = [[aspect.attrib['category'], aspect.attrib['polarity']] for aspect in child]

                sentences.append([sent_id, sent_text, aspect_terms, category_terms])
            return sentences

# Laptop domain

# format: sentence {id} => {text, aspectTerms => {term, polarity, from, to}}
# {'positive': 987, 'negative': 866, 'neutral': 460, 'conflict': 45}
# tree = ET.parse('Data/Laptop_Train_v2.xml')
# root = tree.getroot()
# sentences = []
# polarities = []
#
# for idx, sent in enumerate(root):
#     sent_id = sent.attrib['id']
#     childs = sent.getchildren()
#     for child in childs:
#         tag = child.tag
#         sent_text = ''
#         aspect_terms = []
#         if tag == 'text':
#             sent_text = child.text
#         elif tag == 'aspectTerms':
#             aspect_terms = [[aspect.attrib['term'], aspect.attrib['from'], aspect.attrib['to'], aspect.attrib['polarity']] for aspect in child]
#     sentences.append([sent_id, sent_text, aspect_terms])
#     polarities.extend([aspt[3] if len(aspt) == 4 else 'no' for aspt in aspect_terms])
#
# ob_sent = sum(1 if len(sent[2]) > 0 else 0 for sent in sentences)
# sub_sent = sum(1 if len(sent[2]) == 0 else 0 for sent in sentences)
# print('{}/{}'.format(ob_sent, sub_sent))
# print(idx)
# print(Counter(polarities))

# restaurant domain
# format: sentence {id} => {text, [aspectTerms => {term, polarity, from, to}], [aspectCategories => aspectCategory{category, polarity}]}

# tree = ET.parse('Data/Restaurants_Train_v2.xml')
# root = tree.getroot()
# sentences = []
# polarities = []
#
# for idx, sent in enumerate(root):
#     sent_id = sent.attrib['id']
#     childs = sent.getchildren()
#     for child in childs:
#         tag = child.tag
#         sent_text = ''
#         aspect_terms = []
#         if tag == 'text':
#             sent_text = child.text
#         elif tag == 'aspectTerms':
#             aspect_terms = [[aspect.attrib['term'], aspect.attrib['from'], aspect.attrib['to'], aspect.attrib['polarity']] for aspect in child]
#         elif tag == 'aspectCategories':
#             category_terms = [[aspect.attrib['category'], aspect.attrib['polarity']] for aspect in child]
#
#         sentences.append([sent_id, sent_text, aspect_terms])
#         polarities.extend([aspt[3] if len(aspt) == 4 else 'no' for aspt in aspect_terms])
#
# ob_sent = sum(1 if len(sent[2]) > 0 else 0 for sent in sentences)
# sub_sent = sum(1 if len(sent[2]) == 0 else 0 for sent in sentences)
# print('{}/{}'.format(ob_sent, sub_sent))
# print(idx)
# print(Counter(polarities))


class Opinion_Lexicons(object):
    # load all the sentiment lexicons_
    # 1. in-domian lexicons_:
    #   1) yelp affLex, NegLex
    #   2) amazon affLex, NegLex
    # 2. out-domain lexicons_, automatically created lexicons_:
    #   3) NRC Hashtag Sentiment Lexicon : NRC_Hashtag_Sentiment_AffLex_Neg
    #   4) NRC Emoticon Lexicon(Sentiment 140 Lexicon)
    # 3. out-domain lexicons_, three manually created sentiment lexicons_,
        # 5). NRC emotion lexicon
        # 6). Bing liu's lexicon
        # 7). MPQA subjectivity lexicon

    def load_yelp_aff_neg_lex(self):
        uni_lex_corpus = dict()
        bi_lex_corpus = dict()
        # <term> <tab> <score> <tab> <Npos> <tab> <Nneg>
        # _NEGFIRST, _NEG
        # Both suffixes are attached only to nouns, verbs, adjectives, and adverbs. All other parts of speech do not get these suffixes attached.

        file_names = ['lexicons/Yelp-restaurant-reviews/NRC-Yelp-restaurant-sentiment-AFFLEX-NEGLEX-bigrams.txt',
                      'lexicons/Yelp-restaurant-reviews/NRC-Yelp-restaurant-sentiment-AFFLEX-NEGLEX-unigrams.txt']

        with open(file_names[0], 'r', encoding='utf-8') as bi_reader,  open(file_names[1], 'r', encoding='utf-8') as uni_reader:
            for line in bi_reader:
                [term, score, Npos, Neg] = line.rstrip().split('\t')
                bi_term_key = term.replace('_NEGFIRST', '').replace('_NEG', '')
                bi_lex_corpus[bi_term_key] = score

            for line in uni_reader:
                [term, score, Npos, Neg] = line.rstrip().split('\t')
                uni_term_key = term.replace('_NEGFIRST', '').replace('_NEG', '')
                uni_lex_corpus[uni_term_key] = score

        return uni_lex_corpus, bi_lex_corpus

    def load_amazon_aff_neg_lex(self):
        uni_lex_corpus = dict()
        bi_lex_corpus = dict()
        # <term> <tab> <score> <tab> <Npos> <tab> <Nneg>
        # _NEGFIRST, _NEG
        # Both suffixes are attached only to nouns, verbs, adjectives, and adverbs. All other parts of speech do not get these suffixes attached.

        file_names = ['lexicons_/NRC-Amazon-laptops-sentiment-AFFLEX-NEGLEX/NRC-Amazon-laptops-sentiment-AFFLEX-NEGLEX-bigrams.txt',
                      'lexicons_/NRC-Amazon-laptops-sentiment-AFFLEX-NEGLEX/NRC-Amazon-laptops-sentiment-AFFLEX-NEGLEX-unigrams.txt']

        with open(file_names[0], 'r', encoding='utf-8') as bi_reader, open(file_names[1], 'r', encoding='utf-8') as uni_reader:
            for line in bi_reader:
                [term, score, Npos, Neg] = line.rstrip().split('\t')
                bi_lex_corpus[term] = score

            for line in uni_reader:
                [term, score, Npos, Neg] = line.rstrip().split('\t')
                uni_lex_corpus[term] = score

        return uni_lex_corpus, bi_lex_corpus

    def load_NRC_hashtag_sentiment_lex(self):
        uni_lex_corpus = dict()
        bi_lex_corpus = dict()
        # <term> <tab> <score> <tab> <Npos> <tab> <Nneg>
        # _NEGFIRST, _NEG
        # Both suffixes are attached only to nouns, verbs, adjectives, and adverbs. All other parts of speech do not get these suffixes attached.

        file_names = ['lexicons_/NRC-Sentiment-Emotion-lexicons/NRC-Sentiment-Emotion-lexicons/AutomaticallyGeneratedLexicons/NRC-Hashtag-Sentiment-AffLexNegLex-v1.0/HS-AFFLEX-NEGLEX-bigrams.txt',
                      'lexicons_/NRC-Sentiment-Emotion-lexicons/NRC-Sentiment-Emotion-lexicons/AutomaticallyGeneratedLexicons/NRC-Hashtag-Sentiment-AffLexNegLex-v1.0/HS-AFFLEX-NEGLEX-unigrams.txt']

        with open(file_names[0], 'r', encoding='utf-8') as bi_reader, open(file_names[1], 'r', encoding='utf-8') as uni_reader:
            for line in bi_reader:
                [term, score, Npos, Neg] = line.rstrip().split('\t')
                bi_lex_corpus[term] = score

            for line in uni_reader:
                [term, score, Npos, Neg] = line.rstrip().split('\t')
                uni_lex_corpus[term] = score

        return uni_lex_corpus, bi_lex_corpus

    def load_NRC_emoticon_lex(self):
        uni_lex_corpus = dict()
        bi_lex_corpus = dict()
        # <term> <tab> <score> <tab> <Npos> <tab> <Nneg>
        # _NEGFIRST, _NEG
        # Both suffixes are attached only to nouns, verbs, adjectives, and adverbs. All other parts of speech do not get these suffixes attached.

        file_names = ['lexicons_/NRC-Sentiment-Emotion-lexicons/NRC-Sentiment-Emotion-lexicons/AutomaticallyGeneratedLexicons/NRC-Emoticon-AffLexNegLex-v1.0/Emoticon-AFFLEX-NEGLEX-bigrams.txt',
                      'lexicons_/NRC-Sentiment-Emotion-lexicons/NRC-Sentiment-Emotion-lexicons/AutomaticallyGeneratedLexicons/NRC-Emoticon-AffLexNegLex-v1.0/Emoticon-AFFLEX-NEGLEX-unigrams.txt']

        with open(file_names[0], 'r', encoding='utf-8') as bi_reader, open(file_names[1], 'r', encoding='utf-8') as uni_reader:
            for line in bi_reader:
                [term, score, Npos, Neg] = line.rstrip().split('\t')
                bi_lex_corpus[term] = score

            for line in uni_reader:
                [term, score, Npos, Neg] = line.rstrip().split('\t')
                uni_lex_corpus[term] = score

        return uni_lex_corpus, bi_lex_corpus

    def load_manual_NRC_emotion_lex(self):

        pass

    def load_liu_lex(self):
        lex_corpus = dict()
        # <term> <neg>
        # _NEGFIRST, _NEG
        # Both suffixes are attached only to nouns, verbs, adjectives, and adverbs. All other parts of speech do not get these suffixes attached.

        file_names = ['lexicons/opinion-lexicon-English_Liu/negative-words.txt',
                      'lexicons/opinion-lexicon-English_Liu/positive-words.txt']
        neg_words = [line.rstrip() for line in open(file_names[0], 'r', encoding='utf-8')][36:]
        pos_words = [line.rstrip() for line in open(file_names[1], 'r', encoding='utf-8')][36:]

        for word in neg_words:
            lex_corpus[word] = 'neg'
        for word in pos_words:
            lex_corpus[word] = 'pos'

        return lex_corpus

    def load_mpqa(self):
        # [type,  len, word1, pos1, stemmed1, priorpolarity]
        file_name = 'lexicons/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'
        lst_stemmed1 = []
        with open(file_name, 'r', encoding='utf-8') as reader:
            for line in reader:
                fields = line.rstrip().split()
                try:
                    [_type, _len, word1, pos1, stemmed1, priorpolarity] = [f.split('=')[1] for f in fields]
                    lst_stemmed1.append(stemmed1)
                except Exception as e:
                    print(line)

        print(Counter(lst_stemmed1))


if __name__ == '__main__':
    lex = Opinion_Lexicons()
    uni_lex_corpus, bi_lex_corpus = lex.load_yelp_aff_neg_lex()
    liu_lex_corpus = lex.load_liu_lex()
    print()