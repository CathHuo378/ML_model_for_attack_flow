import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, porter
import numpy as np
from utils.filter_data import *
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
import os
from tika import parser
import sys
from difflib import SequenceMatcher
from deepl_utils import *

#import dataset with attack technique labels
techniques_df = pd.read_csv("dataset.csv")
#import MLP classifier
ml_model_filenames = ['ml_models/MLP_classifier.sav']

#DATA PRE-PROCESSING
def repl(matchobj):
    return ","+ matchobj.group(1) + ","

def remove_empty_lines(text):
	lines = text.split("\n")
	non_empty_lines = [line for line in lines if line.strip() != ""]

	string_without_empty_lines = ""
	for line in non_empty_lines:
		if line != "\n": 
			string_without_empty_lines += line + "\n"

	return string_without_empty_lines 

def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text

def lemmatize_set(dataset):
    lemmatizer = WordNetLemmatizer()
    lemmatized_list = []
    for sentence in dataset:
        word_list = word_tokenize(sentence)
        lemma_list = [lemmatizer.lemmatize(w) for w in word_list]
        lemmatized_list.append(' '.join(lemma_list))
    return lemmatized_list

def stemmatize_set(dataset):
    ps = porter.PorterStemmer()
    stemmatize_list = []
    for sentence in dataset:
        word_list = word_tokenize(sentence)
        stemma_list = [ps.stem(w) for w in word_list]
        stemmatize_list.append(' '.join(stemma_list))
    return stemmatize_list

def extract_text(file_path):

    parsed_document = parser.from_file(file_path)
    lines = []
    lines.append(parsed_document['content'])
    return lines

#extract possible assets
def get_assets(sentence):

    assets = []
    words = np.array([word.strip() for word in sentence.split()])
    assets_lists = ["email", "server", "servers", "webserver", "webservers", "data", "database", "databases", "information", "address", "addresses", "account", "accounts", "file", "files", "record", "records","credentials", "repository", "repositories", "system", "systems", "share", "workspace", "computer", "computers", "machine", "machines"]
    keywords = ["use", "Use", "using", "Using", "used", "Used", "allow", "harvest", "utilised", "utilized", "compromised", "gained", "access", "exploited", "exfiltrated", "take over", "took over"]

    for asset in assets_lists:
        result = ""
        context_words = 1
        for x, word in enumerate(words):
            if word == asset:
                left_context = " ".join(words[:x][-context_words:])
                right_context = " ".join(words[x+1:context_words+x+1])
                result = " ".join([left_context, asset, right_context])
                assets.append(result)

    if not assets:
        for keyword in keywords:
            result = ""
            context_words = 3
            for x, word in enumerate(words):
                if word == keyword:
                    left_context = " ".join(words[:x][-context_words:])
                    right_context = " ".join(words[x+1:context_words+x+1])
                    result = " ".join([left_context, keyword, right_context])
                    assets.append(result)
                    break

                if result:
                    break

    return assets

#similarity test on possible assets
similarity_threshold = 0.4
sentence_range = 3
def find_related_assets(df):
    related_id = []
    for i, row in df.iterrows():
        related_assets = []
        for j in range(max(0, i-sentence_range-1), min(i+sentence_range, len(df))):
            if i != j:
                if isinstance(row["Possible_Assets"], tuple) and isinstance(df.at[j, "Possible_Assets"], tuple):
                    for asset1 in row["Possible_Assets"]:
                        for asset2 in df.at[j, "Possible_Assets"]:
                            similarity_score = SequenceMatcher(None, asset1, asset2).ratio()
                            if similarity_score > similarity_threshold:
                                related_assets.append(df.at[j, "ID"])
                                break
                else:
                    continue
        related_id.append(related_assets)
    df.insert(df.columns.get_loc("Possible_Assets") + 1, "Related_ID", related_id)
    return df

#CLASSIFICATION
def analyze_all_doc(file_path, model_filenames):
    #extract text from documents
    lines = extract_text(file_path)
    
    #apply regex 
    regex_list = load_regex("utils/regex.yml")

    text = combine_text(lines)
    text = re.sub('(%(\w+)%(\/[^\s]+))', repl, text)
    text = apply_regex_to_string(regex_list, text)
    text = re.sub('\(.*?\)', '', text)
    text = remove_empty_lines(text)
    text = text.strip()
    sentences = sent_tokenize(text)

    num_sen = len(sentences)

    double_sentences = []

    for i in range(1, len(sentences)):
        new_sen = sentences[i-1] + sentences[i]
        double_sentences.append(new_sen)
    
    for model_filename in model_filenames: 
        # load the model from disk
        vectorizer, classifier = pickle.load(open(model_filename, 'rb'))

        stemmatized_set = stemmatize_set(sentences)
        lemmatized_set = lemmatize_set(stemmatized_set)
        x_test_vectors = vectorizer.transform(lemmatized_set)
        predicted = classifier.predict(x_test_vectors)
        #Matrix
        #Vector of vector of probabilities
        predict_proba_scores = classifier.predict_proba(x_test_vectors)
        #Identify the indexes of the top predictions (increasing order so let's take the last 2, highest proba)
        top_k_predictions = np.argsort(predict_proba_scores, axis = 1)[:,-2:]
        #Get classes related to previous indexes
        top_class_v = classifier.classes_[top_k_predictions]

        #result setup
        results = {'Tech_ID': [], 'Tech_Name': [], 'Possible_Assets': [], 'Sentence': [],}
        df = pd.DataFrame(results, dtype=object)

        #generate result
        thresholds = [0.10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        for threshold in thresholds: 

            for i in range(0,len(predict_proba_scores)):
                sorted_indexes = top_k_predictions[i]
                top_classes = top_class_v[i]
                proba_vector = predict_proba_scores[i]
                predicted_labels = []
                label_name = []
                used_sentences = []
                if proba_vector[sorted_indexes[1]] > threshold:
                    predicted_label = top_classes[1]
                    label_name = " ".join(re.findall("[a-zA-Z]+", techniques_df[techniques_df["label_subtec"] == predicted_label]["tec_name"].head(1).to_string()))
                    predicted_labels.append(predicted_label)
                    used_sentences.append(sentences[i])
                    result = {"Tech_ID": predicted_label, "Tech_Name": label_name, "Possible_Assets": tuple(get_assets(sentences[i])), "Sentence": sentences[i],}
                    df.loc[len(df)] = result

    #result clean up
    df.drop_duplicates(inplace=True)

    #issue ID
    df.insert(0, 'ID', range(1, len(df) + 1))
    df = df.reset_index(drop=True)

    #link assets
    df = find_related_assets(df)
    return df


#main
n = len(sys.argv)
print("Start analysing", n-1, "documents...")

for i in range(1, n):
    file_path = sys.argv[i]
    results = analyze_all_doc(file_path, ml_model_filenames)

    index = 0
    for char in file_path:
        if char == '.':
            break
        index+=1

    new_file_path = file_path[:index] + '_result.xlsx'
    results.to_excel(new_file_path, index=False)
    print("Analysis progress: ", i, "/", n-1, sep="")
