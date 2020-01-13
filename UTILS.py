import pdb
import torch
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import gc
import os
import math

import nltk
nltk.download('punkt') #it may take running this twice for it to work properly.

from nltk.tokenize import word_tokenize as wtok

gpu = 0 # may have to reset...?

def reset_gpu(new_loc):
    gpu = new_loc

print("Is this the GPU? "+str(torch.cuda.get_device_name(gpu)))
print("If not, reset it using reset_gpu()")

DEVICE = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

def get_data(path, d=","):
    in_f = open(path)
    rows = [ln.split(d) for ln in in_f.read().split("\n") if d in ln]
    in_f.close()
    return rows

TYPING = {"a": lambda x : x.isalpha() and x.islower(),
          "A": lambda x : x.isalpha() and x.isupper(),
          "n": lambda x : x.isnumeric()}

def get_unique_types(list):
    output = []
    for li in list:
        output += [li] if li not in output else []
    return output

def get_type_freqs(list):
    output = {}
    for li in list:
        if li in output:
            output[li] += 1
        else:
            output[li] = 1
    return output

# Let's how many unique indices are in each hierarchy level?
def get_types_per_level(total_set):
    tpl = []
    the_set = [[line[0], line[1].split(">")] for line in total_set]
    ts_max_depth = max([len(line[1]) for line in the_set])
    for cur_depth in range(0,ts_max_depth):
        tpl += [len(get_unique_types([line[1][cur_depth] for line in the_set if len(line[1]) > cur_depth]))]
    return np.array(tpl)

def is_typed(x):
    return True in [f(x) for f in TYPING.values()]
def ws_ctype(x):
    return x if not is_typed(x) else [k for k,v in TYPING.items() if v(x)][0]

def word_shape(s):
    return "".join([ws_ctype(ch) for ch in s])

def ws_summary(s):
    ws = word_shape(s)
    out, wsi = ws[:1], 1
    while wsi < len(ws):
        if ws[wsi] == ws[wsi-1]:
            out += "*"
            while ws[wsi] == ws[wsi-1] if wsi < len(ws) else False:
                wsi += 1
        else:
            out += ws[wsi]
            wsi += 1
    return out

WSS_TO_EXCLUDE = ["a","a*","Aa","Aa*"]

def extract_ws_feats(tokens):
    toks = [t for t in tokens if ws_summary(t) not in WSS_TO_EXCLUDE]
    return get_unique_types([word_shape(t) for t in toks]), get_unique_types([ws_summary(t) for t in toks])

def to_unigram(x):
    return x[0].lower()+x[1:]

BG_DLM = " "

FLAGS = {"ws":"Word shape features", "wss":"Word shape summary features", "u":"Unigram features", "b":"Bigram features"}
FEAT_TYPE_ORDER = ["ws","wss","u","b"]

# theoretically would also implement trigrams here too and/or attention vectors
#   if we wanted actual accuracy, with smoothing. But this is just a proxy test.
class FeatureExtractor():

    def __init__ (self, build_path = "feat_extractor_task1.txt", thresh = 15):
        self.THRESH = thresh # feature threshold, below which we do not keep info on the feature.

        self.BPATH = build_path
        if build_path not in os.listdir(os.getcwd()):
            print("New feature extractor initialized, but no features yet as no valid path in this directory is specified.\n\t"
                  "-- please submit an X set and if necessary a threshhold to build new features")
            self.feats_determined = False
        else:
            self.build_from_path(build_path)

    def build_from_path(self, path = "", thresh = 15):
        self.THRESH = thresh if thresh != -1 else self.THRESH
        if path != "":
            self.BPATH = path
        inf = open(self.BPATH)
        lines = [ln for ln in inf.read().split("\n") if ln != ""]
        inf.close()

        if lines[0] != FLAGS[0] or False in [ftyi in lines for ftyi in FEAT_TYPE_ORDER[1:]]:
            pdb.set_trace()
            raise ValueError("Invalid format for feature set path!")

        self.init_attributes()
        for ftyi in range(0, len(FEAT_TYPE_ORDER)):
            print("Extracting "+FLAGS[FEAT_TYPE_ORDER[ftyi]])
            self.next_build(ftyi)(
                lines[lines.index(FLAGS[FEAT_TYPE_ORDER[ftyi]])+1 :
                      len(lines)+1 if ftyi == len(FEAT_TYPE_ORDER) else lines.index(FLAGS[FEAT_TYPE_ORDER[ftyi+1]]) ] )
        self.finish_attrs()
        self.feats_determined = True

        print("Features successfully extracted from prior run.")

    def next_build(self, i):
        return [self.build_ws_feats, self.build_wss_feats, self.build_unigram_feats, self.build_bigram_feats][i]

    def init_attributes(self):
        self.FT_FXNS = []
        self.FT_NAMES = []
        self.N_WS_FTS, self.N_WSS_FTS, self.FT_LEN, self.LEN_F1, self.LEN_F2 = 0, 0, 0, 0, 0

    def write_to_path(self):
        outf = open(self.BPATH, mode="w")

        for ftyi in range(0, len(FEAT_TYPE_ORDER)):
            outf.write(FLAGS[FEAT_TYPE_ORDER[ftyi]]+"\n")
            for fti in range(self.BOUNDS[ftyi], self.BOUNDS[ftyi+1]):
                outf.write(self.FT_NAMES[fti]+"\n")
        outf.close()

    def build_ws_feats(self, word_shapes):
        self.FT_NAMES += word_shapes
        self.N_WS_FTS = len(word_shapes)
        self.LEN_F1 += len(word_shapes)
        self.FT_LEN += len(word_shapes)
        self.FT_FXNS += [lambda x : shape == word_shape(x) for shape in word_shapes]

    def build_wss_feats(self, word_shape_summaries):
        self.FT_NAMES += word_shape_summaries
        self.N_WSS_FTS += len(word_shape_summaries)
        self.LEN_F1 += len(word_shape_summaries)
        self.FT_LEN += len(word_shape_summaries)
        self.FT_FXNS += [lambda x : summary == ws_summary(x) for summary in word_shape_summaries]

    def build_unigram_feats(self, unigrams):
        self.FT_NAMES += unigrams
        self.FT_FXNS += [lambda x: unigram == to_unigram(x) for unigram in unigrams]
        self.N_U_FTS = len(unigrams)
        self.LEN_F1 += len(unigrams)
        self.FT_LEN += len(unigrams)

    # also builds positional features
    def build_bigram_feats(self, bigrams):
        for bg in bigrams:
            self.new_bigram_feature(bg)

    def finish_attrs(self):
        self.FT_LEN = len(self.FT_NAMES)
        self.LEN_F2 = self.FT_FXNS - self.LEN_F1

        self.FT_NAMES = np.array(self.FT_NAMES)
        self.FT_FXNS = np.array(self.FT_FXNS)

        self.BOUNDS = [0, self.N_WS_FTS, self.N_WS_FTS+self.N_WSS_FTS, self.LEN_F1, self.FT_LEN]


    def build_anew (self, X_for_counts, thresh = -1, ignore_bigrams=False):
        self.THRESH = thresh if thresh != -1 else self.THRESH

        self.init_attributes()

        WS_FRQS, WSS_FRQS = {} , {}
        START_WSS_FRQS, END_WSS_FRQS = {} , {}
            # values are of form [# at start, # at end]

        print("Determining word shape features... (uft = " + str(self.THRESH)+")")
        count = 0

        for desc in X_for_counts:
            ws_feats, wss_feats = extract_ws_feats(wtok(desc))
            del desc
            for ft in ws_feats:
                WS_FRQS[ft] = (WS_FRQS[ft] if ft in WS_FRQS else 0) + 1
            for sft in wss_feats:
                WSS_FRQS[sft] = (WSS_FRQS[sft] if sft in WSS_FRQS else 0) + 1

            if len(wss_feats) > 1:
                start, end = wss_feats[0], wss_feats[-1]

                if start not in WSS_TO_EXCLUDE:
                    START_WSS_FRQS[start] = (START_WSS_FRQS[start] if start in START_WSS_FRQS else 0 ) + 1
                if end not in WSS_TO_EXCLUDE:
                    END_WSS_FRQS[end] = (END_WSS_FRQS[end] if end in END_WSS_FRQS else 0 ) + 1

            count += 1
            if count % 50000 == 0:
                print("On "+str(count)+"th description...")

        self.build_ws_feats([name for name, freq in WS_FRQS.items() if freq > self.THRESH])
        del WS_FRQS; gc.collect()

        self.build_wss_feats([name for name, freq in WSS_FRQS.items() if freq > self.THRESH])
        del WSS_FRQS; gc.collect()

        print("Determining unigram features")
        unigram_cts = {}
        count = 0
        for desc in X_for_counts:
            unigrams_here = get_unique_types(np.vectorize(to_unigram)(wtok(desc)))
            for utyp in unigrams_here:
                unigram_cts[utyp] = (unigram_cts[utyp] if utyp in unigram_cts else 0 ) + 1

            count += 1
            if count % 50000 == 0:
                print("On " + str(count) + "th description...")

        self.build_unigram_feats([name for name, freq in unigram_cts.items() if freq > self.THRESH and word_shape(name) != "a"])
        del unigram_cts; gc.collect()

        #now implement bigram features using self.FT1_NAMES as indexer for unigrams
        # for word shape and word summary features, we only make "bigrams" where the other element is the beginning or
            # end of the phrase -- i.e. these are positional features.
        # the beginning and end of the sentence have the unigram "index" -1

        print("Extracting boundary positional features for word shape and word summary...")

        #block below abrogated and replaced with with this simple method that just draws from exsting info, and only for wss -- time constraints.
        for swssfti in [k for k,fr in START_WSS_FRQS.items() if fr > self.THRESH ]:
            self.new_bigram_feature("-1"+BG_DLM+swssfti)
        del START_WSS_FRQS; gc.collect()
        print("Start wss features extracted")
        for ewssfti in [k for k, fr in END_WSS_FRQS.items() if fr > self.THRESH]:
            self.new_bigram_feature(ewssfti+BG_DLM+"-1")
        del END_WSS_FRQS; gc.collect()
        print("Coda wss features extracted")


        """
        # first extract the boundary positional factors for ws_shape and wss_shape:
        fti = 0
        while fti < self.N_WS_FTS + self.N_WSS_FTS:
            BOUND_FRQS = [0,0] # start of desc, end of desc
            for desc in X_for_counts:
                toks = wtok(desc)
                BOUND_FRQS = [BOUND_FRQS[p] + int(self.FT_FXNS[fti](toks[-1 * p])) for p in [0, 1]]
 
            if BOUND_FRQS[0] > self.THRESH:
                self.new_bigram_feature("-1"+BG_DLM+str(fti))
            if BOUND_FRQS[1] > self.THRESH:
                self.new_bigram_feature(str(fti)+BG_DLM+"-1")

            fti += 1

            if fti % 50 == 0:
                print("on "+str(fti)+"th feature...")
        """

        pdb.set_trace()

        # now unigram-based bigram and boundary positional features
        fti = self.N_WSS_FTS + self.N_WS_FTS
        while fti < self.LEN_F1 and not ignore_bigrams:
            print("Extracting bigram and boundary features based on unigrams... (tot unigrams = " + str(self.N_U_FTS))

            BOUND_FRQS = [0,0]
            PRIOR_FRQS = torch.zeros(self.LEN_F1)
            POSTR_FRQS = torch.zeros(self.LEN_F1)

            print("Collecting second level features based on unigram feature "+str(fti)+" '"+self.FT_NAMES[fti]+"'")

            toks =""
            for desc in [d for d in X_for_counts ]:
                toks = wtok(desc)
                del desc
                for occ in [j for j in range(0,len(toks)) if self.FT_FXNS[fti](toks[j])]:
                    if occ == 0:
                        BOUND_FRQS[0] += 1
                    else:
                        for prior_match in [prii for prii in range(0, self.LEN_F1) if self.FT_FXNS[prii](toks[j-1])]:
                            PRIOR_FRQS[prior_match] += 1

                    if occ == len(toks) - 1:
                        BOUND_FRQS[1] += 1
                    else:
                        for postr_match in [pstri for pstri in range(0, self.LEN_F1) if self.FT_FXNS[pstri](toks[j+1])]:
                            POSTR_FRQS[postr_match] += 1
            del toks; gc.collect()

            if BOUND_FRQS[0] > self.THRESH:
                self.new_bigram_feature("-1"+BG_DLM+str(fti))
            if BOUND_FRQS[1] > self.THRESH:
                self.new_bigram_feature(str(fti)+BG_DLM+"-1")
            del BOUND_FRQS

            for prii in [j for j in range(0,self.LEN_F1) if PRIOR_FRQS[j] > self.THRESH]:
                bg_name = str(prii) + BG_DLM + str(fti)
                if bg_name not in self.FT_NAMES[self.LEN_F1:]:
                    self.new_bigram_feature(bg_name)
            del PRIOR_FRQS; gc.collect()

            for pstri in [j for j in range(0, self.LEN_F1) if POSTR_FRQS[j] > self.THRESH]:
                bg_name = str(fti) + BG_DLM + str(pstri)
                if bg_name not in self.FT_NAMES[self.LEN_F1:]:
                    self.new_bigram_feature(bg_name)

            del POSTR_FRQS; gc.collect()

            fti += 1
            if fti % 50 == 0:
                print("On "+str(fti)+"th feature...")

        self.feats_determined = True
        print("Feature extraction complete!")

        self.write_to_path()

    def new_bigram_feature(self, bigram_name):
        self.FT_NAMES += [bigram_name]
        self.FT_FXNS += [lambda x: self.check_bigram(bigram_name, x)]

    # -1 before BG_DLM -- start of phrase; after -- end
    # remember, the frequency of the bigram within one description doesn't matter -- all that matters is if it occurs.
    def check_bigram(self, bg_str, dsc_toks):
        gram_inds = bg_str.split(BG_DLM)
        gram_inds = [int(gi) for gi in gram_inds if gi != -1]
        return False not in [self.FT_FXNS[gi] for gi in gram_inds]

    # returns a pytorch tensor
    def extract(self, phrase):
        out_f2 = torch.tensor([int(self.FT_FXNS[ft2i](phrase)) for ft2i in range(self.LEN_F1,self.FT_LEN)])

        toks = wtok(phrase)
        out_f1 = torch.tensor([int(True in [self.FT_FXNS[ft1i](tok) for tok in toks]) for ft1i in range(0, self.LEN_F1) ])

        return torch.cat((out_f1, out_f2), 0)

def get_task1_class(ylab):
    return ylab.split(">")[0]

def get_task2_class(ylab):
    return ylab if ">" not in ylab else ylab.split(">")[1]


# TODO below is abrogated
"""
class RFModel():

    # n = number of trees/"estimators"
    def __init__(self, X_for_freq_counts, lvl_transform = get_task1_lab, uft = 2):
        self.LT = lvl_transform
        print("initializing feature extractor with uft = "+str(uft)+"... ")
        self.FE = FeatureExtractor(X_for_freq_counts, uft) # initialization includes necessary preprocessing
        print("initialized RFModel with uft = "+str(uft)+"... n_feats = "+str(self.FE.LEN))

    def fit_with_n_trees(self,n,X,Y, xtrd = False, lvl_transformed = False):
        self.RF = RandomForestClassifier(n_estimators = n, random_state=42)
        print("Fitting random forest with "+str(n)+" trees...")
        tr_inputs = X if xtrd else np.vectorize(lambda desc : self.FE.extract(desc))(X)
        tr_labels = Y if lvl_transformed else np.array(list(map(lambda nest : self.LT(nest),Y)))
        self.RF.fit(tr_inputs, tr_labels)
        print("...done.")

    def predict(self, prediction_inputs, xtrd = False):
        print("Generating predictions...")
        return self.RF.predict(prediction_inputs if xtrd else np.array(list(map(lambda dsc : self.FE.extract(dsc),prediction_inputs))))

    def evaluate(self, Y_inp, Y_gold, objective=F1, xtrd=False, lvl_transformed = False):
        gold_labs = Y_gold if lvl_transformed else np.vectorize(lambda yi : self.LT(yi))(Y_gold)
        predictions = self.predict(Y_inp, xtrd=xtrd)
        return objective(gold_labs, predictions) 
"""