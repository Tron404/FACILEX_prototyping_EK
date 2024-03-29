#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:20:13 2019

@author: kolawole
"""
import nltk, string
#from gensim.models import LdaModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import document_encoder as encoder
#import document_processor as doc_p
import os, re, datetime, math, pandas as pd, numpy as np, networkx as nx
#from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.stem.porter import PorterStemmer
from gensim.models import*
from nltk.tokenize import *
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
from sklearn.cluster import KMeans


#ELMO_MODULE_URL = '/home/kolawole/Desktop/BERT/ELMO'
##DOC= doc_p.Document()
#EMBEDDER = encoder.Embedding()

punkt_param = PunktParameters()
abbreviation = ['u.s.a', 'fig', 'e.t.c', 'etc', 'mr', 'mrs', 'u.k', 'u.a.e', 'eg', 'ie', 'e.g', 'dr', 'i.e', 's.a', 'e.u', 'ph.d', 'm.a.', 'm.sc',  'b.sc', 'b.a', 'prof', 'apt', 'ave', 'blvd', 'ct', 'hwy', 'ln', 'mt', 'rd', 'st', 'ste', 'capt', 'col', 'cpl', 'gen', 'gov', 'jr', 'lt', 'sr', 'sgt', 'no', 'assoc', 'dept', 'inc', 'ltd' ]
numbers = [str(i) for i in range(1000)]
abbreviation.extend(numbers)
punkt_param.abbrev_types = set(abbreviation)
sent_tokenizer = PunktSentenceTokenizer(punkt_param)

stopwords = set([ "i", "a", "about", "an", "are", "as", "at", "be", "by", "for", "from",
        "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
        "was", "what", "when", "where", "who", "will", "with", "the", "'s", "did",
        "have", "has", "had", "were", "'ll"])
stops = nltk.corpus.stopwords.words('english')
stops.extend(string.punctuation)
stops.append('')
  
class Summarizer(encoder.Embedding):

    def __init__(self):  
        super().__init__()
        return
    
    
    def get_cosine_of_vectors(self, vecs): 
        '''
        compute cosine similarity between vectors
        '''
        return cosine_similarity(vecs)
    
    
    def get_top_informative_sentences(self, ranked_sents, summary_size):
        summary = []
        
        # Extract top n sentences as the summary
        if len(ranked_sents) >= summary_size:
            #print('INFO:: Rule 1 applicable')
            for i in range(summary_size):
                summary.append(ranked_sents[i][1])
            summary = ' '.join(summary).replace('\n', '')
            return summary
        
        if len(ranked_sents) == 1:
            #print('[INFO]:: Rule 2 applicable')
            ranked  = ranked_sents[0]
            score, sent = ranked
            summary.append(sent)
            summary = ' '.join(summary).replace('\n', '')
            return summary
        
        #print('INFO:: Rule 3 applicable')
        for i in range(len(ranked_sents)):
            summary.append(ranked_sents[i][1])
        summary = ' '.join(summary).replace('\n', '')
        return summary

    def remove_duplicate_sentences(self, summary):
        seq = sent_tokenizer.tokenize(summary)
        #seq = sent_tokenize(summary)
        seen = set()
        seen_add = seen.add
        return ' '.join([x for x in seq if not (x in seen or seen_add(x))])



    def get_sent_rank(self, orig_sents, embedded_sents):
        # similarity matrix
        #sim_mat = np.zeros([len(embedded_sents), len(embedded_sents)])
    
        if len(embedded_sents) < 3:
            #If the number of sentences to be ranked is less than  3 (i.e., 1 or 2)
            #perhaps the bests thing is to return the exact input
            ranked_sentences = []
            for i in orig_sents:
                ranked_sentences.append((0.0, i))
            return ranked_sentences
    
        #for i in range(len(embedded_sents)):
            #for j in range(len(embedded_sents)):
                #if i != j:
                    #sim_mat[i][j] = cosine_similarity(embedded_sents[i].reshape(1,-1), embedded_sents[j].reshape(1,-1))[0,0]
                    
        sim_mat = self.get_cosine_of_vectors(embedded_sents)
        
        try:
            nx_graph = nx.from_numpy_array(sim_mat)
            scores = nx.pagerank(nx_graph)   
            ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(orig_sents)), reverse=True)
        except:
            #print('[ERROR HANDLING INFO]:: Succesfully handled PowerIterationFailedConvergence Error while generating summary')
            # factorize/decompose similarity matrix with SVD to obtain the non-negative real numbers of the diagonals of the decomposed matrix which when ranked gives the linear depency of each vector (sentence) in the matrix
            vecs = np.row_stack([v for v in sim_mat])
            #eps = np.finfo(np.linalg.norm(vecs).dtype).eps
            #tolerance =  max(eps * np.array(vecs.shape))
            U, scores, V = np.linalg.svd(vecs)
            ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(orig_sents)), reverse=True)
            
        return ranked_sentences


    def get_top_informative_sentences__(self, ranked_sents, summary_size):
        summary = []
        
        #summary_size = int(summary_size_pct)
        # Extract top n sentences as the summary
        if len(ranked_sents) >= summary_size:
            #print('INFO:: Rule 1 applicable')
            for i in range(summary_size):
                summary.append(ranked_sents[i][1])
            summary = ' '.join(summary).replace('\n', '')
            return summary
        
        if len(ranked_sents) == 1:
            #print('[INFO]:: Rule 2 applicable')
            ranked  = ranked_sents[0]
            score, sent = ranked
            summary.append(sent)
            summary = ' '.join(summary).replace('\n', '')
            return summary
        
        #print('INFO:: Rule 3 applicable')
        for i in range(len(ranked_sents)):
            summary.append(ranked_sents[i][1])
        summary = ' '.join(summary).replace('\n', '')
        return summary
    
    
    def summarize(self, mode = '', segments = [], stemmed = False, text_encoder = 'glove', segment_rank_threshold = 3, absolute_rank_threshold = 5):
        
        general_summary = []
        
        if text_encoder.lower() == 'glove':
            text_encoder = 'glove'
        if text_encoder.lower() == 'lda':
            text_encoder = 'lda'
        if mode.lower().strip() == 'full':
            mode = 'full'
        if mode.lower().strip() == 'segment':
            mode = 'segment'
        if mode.lower().strip() == 'paragraph':
            mode = 'paragraph'
        
        
        #validate whether to exit early, if the input is empty, exit. 
        #also if the input contyains only one sentence, return the exact input
        
        if isinstance(segments, list) and len(segments) < 1:
            return ''
        
        if isinstance(segments, list) and len(segments) < 2 and len(sent_tokenizer.tokenize(segments[0].strip())) < 2:
            return segments[0].strip()
        
        if isinstance(segments, str) and len(sent_tokenizer.tokenize(segments.strip())) < 2:
            return segments
        
        #print ('\n******segments\n', segments, '\n')
        
        if mode == 'full':
            
            clean_sentences, original_sentences = self.prepare_sentences(segments) 
            
            #print('\n[SUMMARIZER INFO]:: Document prepared for Processing.')
            if text_encoder == 'glove':  
                sentence_vectors = self.encode_with_glove_embeddings(clean_sentences)
                #print('\n[ENCODER INFO]:: Document Representation >> Document encoded with GLOVE Word Embedding')
            elif text_encoder == 'lda': 
                #print ('\n******clean_sentences\n', clean_sentences, '\n',clean_sentences[0], '\n', 'clean_sentences is a list:', isinstance(clean_sentences, list),  '\n******\n')
                sentence_vectors = self.topic_encoding(sentence_list = clean_sentences, stemmed = False)
                #print('\n[ENCODER INFO]:: Document Representation >> Document encoded with Topic Embedding\n')
            else:
                sentence_vectors = self.encode_with_glove_embeddings(clean_sentences)
               #print('\n[ENCODER INFO]:: Document Representation >> Document encoded with GLOVE Word Embedding')
                
            #print('\n******vectors\n', sentence_vectors, '\n******\n')
            
            ranked_sents = self.get_sent_rank(original_sentences, sentence_vectors)
            #print('\nINFO:: doc ranked \n')
            #segment_rank_threshold = math.ceil(segment_rank_threshold * len(clean_sentences))
            summary_list =  self.get_top_informative_sentences(ranked_sents, segment_rank_threshold)
            #print('\nINFO:: informative parts extracted \n')
            
            general_summary_text = ''.join(summary_list)
            #print('\n*************INFO:: inner summary_1:', general_summary_text, '********\n')
            
            #print('\n***************general summary text ***************\n', general_summary_text, '\n********************\n')
            
            g_clean_sentences, g_original_sentences = self.prepare_sentences(general_summary_text) 
            
            if text_encoder == 'glove':  
                general_sentence_vectors = self.encode_with_glove_embeddings(g_clean_sentences)
                
            elif text_encoder == 'lda': 
                general_sentence_vectors = self.topic_encoding(sentence_list = g_clean_sentences, stemmed = False)
                
            else:
                general_sentence_vectors = self.encode_with_glove_embeddings(clean_sentences)
    
            #general_sentence_vectors = encode_with_w2v_embeddings(g_clean_sentences)
            g_ranked_sents = self.get_sent_rank(g_original_sentences, general_sentence_vectors)
            #absolute_rank_threshold = math.ceil(absolute_rank_threshold * len(g_clean_sentences))
            summary_list =  self.get_top_informative_sentences(g_ranked_sents, absolute_rank_threshold)
            summary_text = ''.join(summary_list)
            #print('\n*************INFO:: inner summary_2:', summary_text, '********\n')
            return self.remove_duplicate_sentences(summary_text)
        
        if len(segments) ==  1:
            
            clean_sentences, original_sentences = self.prepare_sentences(segments[0])  
            
            if text_encoder == 'glove':  
                sentence_vectors = self.encode_with_glove_embeddings(clean_sentences)
                
            elif text_encoder == 'lda': 
                sentence_vectors = self.topic_encoding(sentence_list = clean_sentences, stemmed = False)
                
            else:
                sentence_vectors = self.encode_with_glove_embeddings(clean_sentences)
            
            ranked_sents = self.get_sent_rank(original_sentences, sentence_vectors)
            #segment_rank_threshold = math.ceil(segment_rank_threshold * len(clean_sentences))
            summary_list =  self.get_top_informative_sentences(ranked_sents, segment_rank_threshold)
            summary_text = ''.join(summary_list) 
            return self.remove_duplicate_sentences(summary_text)
        
        #seg_count = 0
        #for  segment in segments:
            #print (seg_count, '-->>>', segment, '\n')
            #seg_count += 1
            
        #seg_count = 0
        for  segment in segments:
            
            clean_sentences, original_sentences = self.prepare_sentences(segment) 
            
            if text_encoder == 'glove':  
                sentence_vectors = self.encode_with_glove_embeddings(clean_sentences)
                
            elif text_encoder == 'lda': 
                sentence_vectors = self.topic_encoding(sentence_list = clean_sentences, stemmed = False)
                
            else: 
                sentence_vectors = self.encode_with_glove_embeddings(clean_sentences)
                
            
            #sentence_vectors = encode_with_w2v_embeddings(clean_sentences)
            ranked_sents = self.get_sent_rank(original_sentences, sentence_vectors)
            #segment_rank_threshold = math.ceil(segment_rank_threshold * len(clean_sentences))
            summary_list =  self.get_top_informative_sentences(ranked_sents,segment_rank_threshold)
            #print(summary_list)
            summary_text = ''.join(summary_list)
            #print(summary_text)
            general_summary.append(summary_text)
            
            #print (seg_count, segment, '-->>>', summary_text, '\n')
            #seg_count +=  1
        
        general_summary_text = ' '.join(general_summary)
        
        g_clean_sentences, g_original_sentences = self.prepare_sentences(general_summary_text) 
        
        if text_encoder == 'glove':  
            general_sentence_vectors = self.encode_with_glove_embeddings(g_clean_sentences)
            
        elif text_encoder == 'lda': 
            general_sentence_vectors = self.topic_encoding(sentence_list = g_clean_sentences, stemmed = False)
            
        else:
            general_sentence_vectors = self.encode_with_glove_embeddings(clean_sentences)
            
        g_ranked_sents = self.get_sent_rank(g_original_sentences, general_sentence_vectors)
        summary_list =  self.get_top_informative_sentences(g_ranked_sents, absolute_rank_threshold)
        summary_text = ''.join(summary_list)
        
        return self.remove_duplicate_sentences(summary_text)
        
            
    
    def process_from_directory(self, input_dir = '', splitter_model = 1, stemmed = False, text_encoder = 'glove', mode = 'segment', segment_rank_threshold = 3, absolute_rank_threshold = 5):
        
       
        #mode can be either of : segment, paragrapgh, or full
        logpd = '_Summary_LOG.csv'
        doc = '_Summary.csv'
        doclog = ''
        logdoc = ''
        sents_pick_desc = '_Selected_Top_' +str(absolute_rank_threshold) + '_informative sentences_'
        encoder_desc = '_Encoder_is_' + text_encoder + '_embeddings_'
        mode_desc = 'Text_Chunk_is_' + mode 
        
        timestamp = datetime.datetime.now().timestamp()
        if text_encoder == 'w2v':
            text_encoder = 'glove'
            
        if mode.lower() == 'segment':
            mode = 'segment'
        elif mode.lower() == 'paragraph':
            mode = 'paragraph'
        elif mode.lower() == 'full':
            mode = 'full'
        else:
            mode = 'segment'
            #print('[SUMMARIZER INFO]:: Wrong mode selected. Mode should be either of full (whole document), segment (splitted segemnts), or paragraph (paragraph-tokenized document) \n')
            #print('[SUMMARIZER INFO]:: Switching to mode:>> [Segment]')
           
        filenames = [f[:] for f in os.listdir(input_dir) if re.search('\.pdf',f)]
        filenames.extend([f[:] for f in os.listdir(input_dir) if re.search('\.PDF',f)])
        filenames.extend([f[:] for f in os.listdir(input_dir) if re.search('\.Pdf',f)])
        filenames.extend([f[:] for f in os.listdir(input_dir) if re.search('\.docx',f)])
        filenames.extend([f[:] for f in os.listdir(input_dir) if re.search('\.DOCX',f)])
        filenames.extend([f[:] for f in os.listdir(input_dir) if re.search('\.Docx',f)])
        filenames.extend([f[:] for f in os.listdir(input_dir) if re.search('\.doc',f)])
        filenames.extend([f[:] for f in os.listdir(input_dir) if re.search('\.Doc',f)])
        filenames.extend([f[:] for f in os.listdir(input_dir) if re.search('\.DOC',f)])
        
        summary = ''
        filename_log = []
        summary_log = []
        summaries = []
       
        
        for i, filename in enumerate(tqdm(filenames)):
            #print ('SPLITTER PROGRESS INFO:: processing document {} >>> {} of {} documents \n'.format(filename,i+1, len(filenames)) )
            filepath = input_dir + "/" + filename
            if os.path.isdir(filepath):
                continue
        
            if mode == 'segment':
                try:
                    if splitter_model == 1:
                        full_text, splitted_segment, para_segment = self.read_document(filepath = filepath, read_mode = 'segment', splitter_model = 1)
                    elif splitter_model == 4:
                        full_text, splitted_segment, para_segment = self.read_document(filepath = filepath, read_mode = 'segment', splitter_model = 4)
                    elif splitter_model == 6:
                        full_text, splitted_segment, para_segment = self.read_document(filepath = filepath, read_mode = 'segment', splitter_model = 6)
                    elif splitter_model not in [1,4,6]:
                        full_text, splitted_segment, para_segment = self.read_document(filepath = filepath, read_mode = 'segment', splitter_model = 1)
                    
                    if len(splitted_segment) < 5:
                        mode = 'paragraph'
                except:
                    #print ('[ERROR INFO]:: Error reading document {} \n'.format(filename))
                    continue
                
            if mode == 'paragraph':
                try:
                    full_text, splitted_segment, para_segment = self.read_document(filepath = filepath, read_mode = 'paragraph', splitter_model = 1)
                    
                    if len(para_segment) < 4:
                        mode = 'full'
                except:
                    #print ('[ERROR INFO]:: Error reading document {} \n'.format(filename))
                    continue
                
            if mode == 'full':
                try:
                    full_text, splitted_segment, para_segment = self.read_document(filepath = filepath, read_mode = 'full', splitter_model = 1)
                except:
                    #print ('[ERROR INFO]:: Error reading document {} \n'.format(filename))
                    continue
            
            
            try:
                if mode == 'segment':
                    #print('\n[SUMMARIZER INFO]:: Found {} sections in document: {}'.format(len(splitted_segment),filename))   
                    #summary = self.summarize(segments = splitted_segment, stemmed = False, text_encoder = 'w2v', segment_rank_threshold = 2, absolute_rank_threshold = 5)
                    summary = self.summarize(mode = mode, segments = splitted_segment, stemmed = stemmed, text_encoder = text_encoder, segment_rank_threshold = segment_rank_threshold, absolute_rank_threshold = absolute_rank_threshold)

                if mode == 'paragraph':
                    #print('\n[SUMMARIZER INFO]:: Found {} paragraphs in document: {}'.format(len(para_segment),filename)) 
                    summary = self.summarize(mode = mode, segments = para_segment, stemmed = stemmed, text_encoder = text_encoder, segment_rank_threshold = segment_rank_threshold, absolute_rank_threshold = absolute_rank_threshold)
        
                if mode == 'full':
                    #print('\n[SUMMARIZER INFO]:: Using whole text from document: {}'.format(filename)) 
                    summary = self.summarize(mode = mode, segments = full_text, stemmed = stemmed, text_encoder = text_encoder, segment_rank_threshold = segment_rank_threshold, absolute_rank_threshold = absolute_rank_threshold)
            
                #print ('\n[SUMMARIZER INFO]:: Finished summary extraction for document {}.\n'.format(filename))
                summary_ = sent_tokenizer.tokenize(summary)
                #summary_ = sent_tokenize(summary)
                summary_ = '\n'.join(summary_)
                docdict = {"Summary":summary_}
                index = [0]
                docframe = pd.DataFrame(docdict, columns=['Summary'], index =index)
                fname = filename[:-4].upper()
                
                sents_pick_desc = '_Selected_Top_' +str(absolute_rank_threshold) + '_informative sentences_'
                encoder_desc = '_Encoder_is_' + text_encoder + '_embeddings_'
                mode_desc = 'Text_Chunk_is_' + mode 
                #logdoc = mode + '_' + text_encoder + '_' + fname + doc
                logdoc = mode_desc + encoder_desc  + sents_pick_desc + fname + doc
                doclog =  input_dir + "/" + logdoc           
                docframe.to_csv(doclog, index=False)
            
            except:
                #print ('\n[ERROR INFO]:: Error generating summary for document {}. Skipping document. \n'.format(filename))
                continue
            summaries.append(summary)
            filename_log.append(filename)
            summary_log.append(summary_)
    
        logger = str(timestamp) +  '_' + mode + '_' + text_encoder + logpd
        logframepath = input_dir + "/" + logger
    
        logdict = {"Filename": filename_log, "Summary": summary_log}  
        #index = [0]
        logframe = pd.DataFrame(logdict, columns=['Filename', 'Summary'])
        logframe = logframe.sort_values(by= ['Filename'], ascending=True)
        logframe.to_csv(logframepath, index=False)  
        
        try:
            writer = pd.ExcelWriter(logframepath[:-4] + '.xlsx', engine='xlsxwriter')
            logframe.to_excel(writer, sheet_name='Document Summary Log')
            writer.save()
        except:
            pass
        #print('\n[SUMMARIZER INFO]:: Summary extraction finished. Summary logged at {}'.format(logframepath))
        
        return  summaries
    
    
    
    def summarize_single_document(self, filepath = '', splitter_model = 1, stemmed = False, text_encoder = 'glove', mode = 'segment', segment_rank_threshold = 2, absolute_rank_threshold = 7):
        
        #EMBEDDER = Embedding()
        #mode can be either of : segment, paragraph, or full
        doc = '_Summary.csv'
        doclog = ''
        logdoc = ''
        #timestamp = datetime.datetime.now().timestamp()
        sents_pick_desc = '_Selected_Top_' +str(absolute_rank_threshold) + '_informative sentences_'
        encoder_desc = '_Encoder_is_' + text_encoder + '_embeddings_'
        mode_desc = 'Text_Chunk_is_' + mode 
        
        if text_encoder == 'w2v':
            text_encoder = 'glove'
            
        if mode.strip().lower() == 'segment':
            mode = 'segment'
        elif mode.strip().lower() == 'paragraph':
            mode = 'paragraph'
        elif mode.strip().lower() == 'full':
            mode = 'full'
        else:
            mode = 'segment'
            #print('[SUMMARIZER INFO]:: Wrong mode selected. Mode should be either of full (whole document), segment (splitted segemnts), or paragraph (paragraph-tokenized document) \n')
            #print('[SUMMARIZER INFO]:: Switching to MODE:>> [Segment]')
           
        path, filename = os.path.split(filepath)
        summary = ''
        
        if mode == 'segment':
            try:
                if splitter_model == 1:
                    full_text, splitted_segment, para_segment = self.read_document(filepath = filepath, read_mode = 'segment', splitter_model = 1)
                elif splitter_model == 4:
                    full_text, splitted_segment, para_segment = self.read_document(filepath = filepath, read_mode = 'segment', splitter_model = 4)
                elif splitter_model == 6:
                    full_text, splitted_segment, para_segment = self.read_document(filepath = filepath, read_mode = 'segment', splitter_model = 6)
                elif splitter_model not in [1,4,6]:
                    full_text, splitted_segment, para_segment = self.read_document(filepath = filepath, read_mode = 'segment', splitter_model = 1)
                    
            except:
                pass
                #print ('[ERROR INFO]:: Error reading document {} \n'.format(filename))
        
        elif mode == 'full':
            try:
                full_text, splitted_segment, para_segment = self.read_document(filepath = filepath, read_mode = 'full', splitter_model = 1)
            except:
                pass
                #print ('[ERROR INFO]:: Error reading document {} \n'.format(filename))
                
        else:
            try:
                full_text, splitted_segment, para_segment = self.read_document(filepath = filepath, read_mode = 'paragraph', splitter_model = 1)
            except:
                pass
                #print ('[ERROR INFO]:: Error reading document {} \n'.format(filename))
        
        try:
            if mode == 'segment':
                #print('\n[SUMMARIZER INFO]:: Found {} sections in document: {}'.format(len(splitted_segment), filename))   
                #summary = self.summarize(segments = splitted_segment, stemmed = False, text_encoder = 'w2v', segment_rank_threshold = 2, absolute_rank_threshold = 5)
                summary = self.summarize(mode = mode, segments = splitted_segment, stemmed = stemmed, text_encoder = text_encoder, segment_rank_threshold = segment_rank_threshold, absolute_rank_threshold = absolute_rank_threshold)
                
            elif mode == 'paragraph':
                #print('\n[SUMMARIZER INFO]:: Found {} paragraphs in document: {}'.format(len(para_segment), filename))   
                summary = self.summarize(mode = mode,  segments = para_segment, stemmed = stemmed, text_encoder = text_encoder, segment_rank_threshold = segment_rank_threshold, absolute_rank_threshold = absolute_rank_threshold)
        
            elif mode == 'full':
                #print('\n[SUMMARIZER INFO]:: Using whole text from document: {}'.format(filename))   
                summary = self.summarize(mode = mode, segments = full_text, stemmed = stemmed, text_encoder = text_encoder, segment_rank_threshold = segment_rank_threshold, absolute_rank_threshold = absolute_rank_threshold)
            
            #print ('\n[SUMMARIZER INFO]:: Finished summary extraction for document {}.'.format(filename))
            
            summary_ = sent_tokenizer.tokenize(summary)
            #summary_ = sent_tokenize(summary)
            summary_ = '\n'.join(summary_)
            docdict = {"Summary":summary_}
            #docdict = {"Summary":summary}
            index = [0]
            docframe = pd.DataFrame(docdict, columns=['Summary'], index =index)
            fname = filename[:-4].upper()
            #print('fname is', fname)
            logdoc = mode_desc + encoder_desc  + sents_pick_desc + fname + doc
            doclog =  path + "/" + logdoc           
            docframe.to_csv(doclog, index=False)
            
#            try:
#                writer = pd.ExcelWriter(doclog[:-4] + '.xlsx', engine='xlsxwriter')
#                docframe.to_excel(writer, sheet_name='Document Summary Log')
#                writer.save()
#            except:
#                pass
            
        except:
            pass
            #print ('\n[ERROR INFO]:: Error generating summary for document {}. Skipping document.'.format(filename))
                
        #print('\n[SUMMARIZER INFO]:: Summary extraction finished. Summary logged at {}'.format(doclog))
        
        return summary
    
    def glove_ranking(self, text, num_sents = 20, threshold = 0.6):
    
        sentences = self.get_sentences(text)
    
        s_vec = self.encode_with_glove_embeddings(sentences)
        doc_rep =  sum([vec for vec in s_vec])/(len(s_vec)+0.001)
    
        scores = []
        for i in range(len(s_vec)):
            scores.append(cosine_similarity(s_vec[i].reshape(1,-1), doc_rep.reshape(1,-1))[0,0])
    
        pd_dict = {'sentences':sentences, 'scores':scores}
        frame = pd.DataFrame(pd_dict, columns=['sentences', 'scores'])
        scores_ = sorted(scores, reverse=True)
        score_lent = len(scores)
        scores_threshold = scores_[math.ceil(score_lent *threshold)]
        frame_ = frame.loc[frame.scores > scores_threshold, :]
        sents = list(frame_.sentences)
    
        s_vec_ = self.encode_with_glove_embeddings(sents)
        doc_rep_ =  sum([vec for vec in s_vec_])/(len(s_vec_)+0.001)
    
        s_scores = []
        for i in range(len(s_vec_)):
            s_scores.append(cosine_similarity(s_vec_[i].reshape(1,-1), doc_rep_.reshape(1,-1))[0,0])
        
        ranked_sentences = sorted(((s_scores[i],s) for i,s in enumerate(sents)), reverse=True)
        selected = [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences) if i < num_sents]
        #ranked_all =  [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences)] 
        return  ' '.join(selected)
    
    
    
    
    def glove_doc_averaging_hybrid_rank(self, filepath, num_sents = 15, threshold = 0.6):
    
        sentences, _, _ =  self.read_document(filepath = filepath, read_mode = 'raw')
    
        s_vec = self.encode_with_glove_embeddings(sentences)
        doc_rep =  sum([vec for vec in s_vec])/(len(s_vec)+0.001)
    
        scores = []
        for i in range(len(s_vec)):
            scores.append(cosine_similarity(s_vec[i].reshape(1,-1), doc_rep.reshape(1,-1))[0,0])
    
        pd_dict = {'sentences':sentences, 'scores':scores}
        frame = pd.DataFrame(pd_dict, columns=['sentences', 'scores'])
        scores_ = sorted(scores, reverse=True)
        score_lent = len(scores)
        scores_threshold = scores_[math.ceil(score_lent *threshold)]
        frame_ = frame.loc[frame.scores > scores_threshold, :]
        sents = list(frame_.sentences)
    
        s_vec_ = self.encode_with_glove_embeddings(sents)
        doc_rep_ =  sum([vec for vec in s_vec_])/(len(s_vec_)+0.001)
    
        s_scores = []
        for i in range(len(s_vec_)):
            s_scores.append(cosine_similarity(s_vec_[i].reshape(1,-1), doc_rep_.reshape(1,-1))[0,0])
        
        ranked_sentences = sorted(((s_scores[i],s) for i,s in enumerate(sents)), reverse=True)
        selected = [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences) if i < num_sents]
        #ranked_all =  [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences)] 
        return  ' '.join(selected)
    
    def glove_doc_averaging_rank(self, filepath, num_sents = 15):
        sentences, _, _ =  self.read_document(filepath = filepath, read_mode = 'raw')
    
        s_vec = self.encode_with_glove_embeddings(sentences)
        doc_rep =  sum([vec for vec in s_vec])/(len(s_vec)+0.001)
    
        scores = []
        for i in range(len(s_vec)):
            scores.append(cosine_similarity(s_vec[i].reshape(1,-1), doc_rep.reshape(1,-1))[0,0])
        
        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        selected = [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences) if i < num_sents]
        #ranked_all =  [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences)] 
        return ' '.join(selected)
    
    
    def lda_doc_averaging_rank(self, filepath, num_sents = 15):
    
        sentences, _, _ =  self.read_document(filepath = filepath, read_mode = 'raw')
    
        topic_vec = self.topic_encoding(sentence_list = sentences, stemmed = False)
        doc_rep =  sum([vec for vec in topic_vec])/(len(topic_vec)+0.001)
    
        scores = []
        for i in range(len(topic_vec)):
            scores.append(cosine_similarity(topic_vec[i].reshape(1,-1), doc_rep.reshape(1,-1))[0,0])
        
        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        selected = [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences) if i < num_sents]
        #ranked_all =  [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences)] 
        return ' '.join(selected)
    
    def lda_rank(self, filepath, num_sents = 15):
    
        sentences, _, _ =  self.read_document(filepath = filepath, read_mode = 'raw')

        topic_vec = self.topic_encoding(sentence_list = sentences, stemmed = False)
        sim_mat = self.get_cosine_of_vectors(topic_vec)
        nx_graph = nx.from_numpy_array(sim_mat)

        scores = nx.pagerank(nx_graph)   
        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        selected = [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences) if i < num_sents + 1]
        #all_ranked = [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences)]
        return ' '.join(selected)
    
    def glove_rank(self, filepath, num_sents = 15):
    
        sentences, _, _ =  self.read_document(filepath = filepath, read_mode = 'raw')

        s_vec = self.encode_with_glove_embeddings(sentences)
        sim_mat = self.get_cosine_of_vectors(s_vec)
        nx_graph = nx.from_numpy_array(sim_mat)

        scores = nx.pagerank(nx_graph)   
        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        selected = [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences) if i < num_sents + 1]
        #all_ranked = [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences)]
        return ' '.join(selected)
    
    
    def tfidf_rank(self, filepath, num_sents = 15):
    
        sentences, _, _ =  self.read_document(filepath = filepath, read_mode = 'raw')

        vectorizer = TfidfVectorizer()
        tfidf_vec = vectorizer.fit_transform(sentences)
        sim_mat = self.get_cosine_of_vectors(tfidf_vec)
        nx_graph = nx.from_numpy_array(sim_mat)

        scores = nx.pagerank(nx_graph)   
        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        selected = [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences) if i < num_sents ]
        #all_ranked = [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences)]
        return ' '.join(selected)
    
    def cluster_rank(self, filepath, num_sents = 15):
        
        sentences, _, _ =  self.read_document(filepath = filepath, read_mode = 'raw')
    
        s_vec = self.encode_with_glove_embeddings(sentences)
        n_clusters = int(np.ceil(len(s_vec)**0.5))+1
        kmeans = KMeans(init='k-means++',n_clusters=n_clusters, random_state=0).fit(np.asarray(s_vec))
    
        avg = []
        for j in range(n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, s_vec)
        ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    
        summary = ' '.join([sentences[closest[idx]] for idx in ordering])
        summaries = sent_tokenizer.tokenize(summary)  
        if len(summaries) >= num_sents:
            summary_ = ' '.join(summaries[:num_sents])
        else:
            summary_ =summary
        return summary_, summary
    
    def lda_with_doc_averaging_hybrid_rank(self, filepath, num_sents = 15, threshold = 0.6):
    
        sentences, _, _ =  self.read_document(filepath = filepath, read_mode = 'raw')
    
        topic_vec = self.topic_encoding(sentence_list = sentences, stemmed = False)
        sim_mat = self.get_cosine_of_vectors(topic_vec)
        nx_graph = nx.from_numpy_array(sim_mat)

        scores = nx.pagerank(nx_graph) 
        scores = list(scores.values())
    
        pd_dict = {'sentences':sentences, 'scores':scores}
        frame = pd.DataFrame(pd_dict, columns=['sentences', 'scores'])
    
        scores_ = sorted(scores, reverse=True)
        score_lent = len(scores)
        scores_threshold = scores_[math.ceil(score_lent * threshold)]
    
        frame_ = frame.loc[frame.scores > scores_threshold, :]
        sents = list(frame_.sentences)
        #f_score = list(frame_.scores)

        s_vec_ = self.encode_with_glove_embeddings(sents)
        doc_rep_ =  sum([vec for vec in s_vec_])/(len(s_vec_)+0.001)
    
        s_scores = []
        for i in range(len(s_vec_)):
            s_scores.append(cosine_similarity(s_vec_[i].reshape(1,-1), doc_rep_.reshape(1,-1))[0,0])
        
        ranked_sentences = sorted(((s_scores[i],s) for i,s in enumerate(sents)), reverse=True)
        selected = [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences) if i < num_sents+1]
        #ranked_all =  [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences)] 
        return ' '.join(selected)
    
    
    def glove_doc_averaging_with_lda_hybrid_rank(self, filepath, num_sents = 15, threshold = 0.7):
        sentences, _, _ =  self.read_document(filepath = filepath, read_mode = 'raw')
    
        s_vec = self.encode_with_glove_embeddings(sentences)
        doc_rep =  sum([vec for vec in s_vec])/(len(s_vec)+0.001)
    
        scores = []
        for i in range(len(s_vec)):
            scores.append(cosine_similarity(s_vec[i].reshape(1,-1), doc_rep.reshape(1,-1))[0,0])
   
        pd_dict = {'sentences':sentences, 'scores':scores}
    
        frame = pd.DataFrame(pd_dict, columns=['sentences', 'scores'])
   
        scores_ = sorted(scores, reverse=True)
        score_lent = len(scores)
        scores_threshold = scores_[math.ceil(score_lent * threshold)]
    
        frame_ = frame.loc[frame.scores > scores_threshold, :]
        sents = list(frame_.sentences)
        #f_score = list(frame_.scores)
   
        topic_vec = self.topic_encoding(sentence_list = sents, stemmed = False)
        sim_mat = self.get_cosine_of_vectors(topic_vec)
        nx_graph = nx.from_numpy_array(sim_mat)

        scores = nx.pagerank(nx_graph) 
        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sents)), reverse=True)
        selected = [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences) if i < num_sents+1]
        #ranked_all =  [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences)] 
        return ' '.join(selected)
       
    def tfidf_with_glove_doc_averaging_hybrid_rank(self, filepath, num_sents = 15, threshold = 0.6):
    
        sentences, _, _ =  self.read_document(filepath = filepath, read_mode = 'raw')
    
        vectorizer = TfidfVectorizer()
        tfidf_vec = vectorizer.fit_transform(sentences)
        sim_mat = self.get_cosine_of_vectors(tfidf_vec)
        nx_graph = nx.from_numpy_array(sim_mat)

        scores = nx.pagerank(nx_graph) 
        scores = list(scores.values())
   
        pd_dict = {'sentences':sentences, 'scores':scores}
    
        frame = pd.DataFrame(pd_dict, columns=['sentences', 'scores'])
  
        scores_ = sorted(scores, reverse=True)
        score_lent = len(scores)
        scores_threshold = scores_[math.ceil(score_lent * threshold)]
    
        frame_ = frame.loc[frame.scores > scores_threshold, :]
        sents = list(frame_.sentences)
        #f_score = list(frame_.scores)

        s_vec_ = self.encode_with_glove_embeddings(sents)
        doc_rep_ =  sum([vec for vec in s_vec_])/(len(s_vec_)+0.001)
    
        s_scores = []
        for i in range(len(s_vec_)):
            s_scores.append(cosine_similarity(s_vec_[i].reshape(1,-1), doc_rep_.reshape(1,-1))[0,0])
        
        ranked_sentences = sorted(((s_scores[i],s) for i,s in enumerate(sents)), reverse=True)
        selected = [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences) if i < num_sents]
        #ranked_all =  [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences)] 
        return ' '.join(selected)
    
    def glove_doc_averaging_with_tfidf_hybrid_rank(self, filepath, num_sents = 15, threshold = 0.7):
        sentences, _, _ =  self.read_document(filepath = filepath, read_mode = 'raw')
    
        s_vec = self.encode_with_glove_embeddings(sentences)
        doc_rep =  sum([vec for vec in s_vec])/(len(s_vec)+0.001)
    
        scores = []
        for i in range(len(s_vec)):
            scores.append(cosine_similarity(s_vec[i].reshape(1,-1), doc_rep.reshape(1,-1))[0,0])
   
        pd_dict = {'sentences':sentences, 'scores':scores}
    
        frame = pd.DataFrame(pd_dict, columns=['sentences', 'scores'])
    
        scores_ = sorted(scores, reverse=True)
        score_lent = len(scores)
        scores_threshold = scores_[math.ceil(score_lent * threshold)]
    
        frame_ = frame.loc[frame.scores > scores_threshold, :]
        sents = list(frame_.sentences)
        #f_score = list(frame_.scores)
    
        vectorizer = TfidfVectorizer()
        tfidf_vec = vectorizer.fit_transform(sents)
        sim_mat = self.get_cosine_of_vectors(tfidf_vec)
        nx_graph = nx.from_numpy_array(sim_mat)

        scores = nx.pagerank(nx_graph) 
        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sents)), reverse=True)
        selected = [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences) if i < num_sents]
        #ranked_all =  [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences)] 
        return ' '.join(selected)
    
    
    def compute_word_freq(self, text):
        sentences = sent_tokenizer.tokenize(' '.join(text))
        text_ = [i for i in sentences if not i.isdigit() and '---' not in i and '___' not in i and '...' not in i and len(i) >= 3 and i not in stops]
        all_words = list(set(' '.join(text_).lower().split()))
        all_words = [i for i in all_words if not i.isdigit() and len(i) > 2 and self.is_word(i) and '---' not in i and i not in stops]
        vectordim = len(all_words)
        sentences = [i for i in sentences if not '...' in i]
        sentences = [i for i in sentences if not '___' in i]
        sentences = [i for i in sentences if not '---' in i]
    
        sentences = [i for i in sentences if not i.lower().strip().startswith('annex')]
        sentences = [i for i in sentences if not i.lower().strip().startswith('figure')]
        sentences = [i for i in sentences if not i.lower().strip().startswith('annexure')]
        sentences = [i for i in sentences if not i.lower().strip().startswith('appendix')]
    
        all_vec = []
        for i in sentences:
            v = [0] * vectordim
            for w in i.lower().split():
                try:
                    v[all_words.index(w)] += 1
                except:
                    pass
            all_vec.append(v)
        return np.array(all_vec), all_vec, sentences, all_words

    def word_freq_rank(self, filepath, num_sents = 15):
    
        text, _, _ =  self.read_document(filepath = filepath, read_mode = 'raw')
    
        freq_vec, freq_vec_, my_sentences, all_words = self.compute_word_freq(text)
    
        sim_mat = self.get_cosine_of_vectors(freq_vec)
    
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)   
        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(my_sentences)), reverse=True)
        selected = [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences) if i < num_sents + 1]
        #all_ranked = [ranked_sentences[i][1] for i, j in enumerate(ranked_sentences)]
        return ' '.join(selected)