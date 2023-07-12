import pandas as pd
import os
import sys
from javalang.ast import Node
import javalang
import random
sys.setrecursionlimit(10000)
import time
class Pipeline:
    def __init__(self,  ratio, root):
        self.ratio = ratio
        self.root = root
        self.sources = None
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.size = None

    # parse source code
    def parse_source(self, output_file, option):
        path = self.root + output_file
        if os.path.exists(path) and option is 'existing':
            source = pd.read_pickle(path)
        else:
            import javalang
            def parse_program(func):
                tree = javalang.parse.parse(func)
                return tree

            source = pd.read_pickle(self.root+'programs.pkl')
            source.columns = ['id', 'code', 'label']
            source['code'] = source['code'].apply(parse_program)

            source.to_pickle(path)
        self.sources = source
        return source

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, size):
        self.size = size
        trees = self.sources
        trees['code'] = trees['code'].apply(parse_ast)
        sequences = trees['code'].values.tolist()
        corpus = []
        n = len(sequences)
        for i in range(n):
            seq = str(sequences[i])
            line = seq.split()
            corpus.append(line)
        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(sentences=corpus, size=size, workers=16, sg=1, min_count=1)
        if not os.path.exists(self.root+'embedding'):
            os.mkdir(self.root+'embedding')
        w2v.save(self.root+'embedding/node_w2v_' + str(size)) 

        # 生成特征
        max_token = len(w2v.wv.index2word)
        vocab = w2v.wv.vocab

        import numpy as np
        def trans2feature(sequence):
            
            result = []
            sequence = str(sequence).split()
            for seq in sequence:
                if seq not in vocab:
                    result.append(w2v.wv.__getitem__(vocab[max_token]))
                else:
                    vector = w2v.wv.__getitem__(seq)
                    result.append(vector)
            return result

        trees['code'] = trees['code'].apply(trans2feature)
        trees.to_pickle(self.root+'/vectors.pkl')

    # split data for training, developing and testing
    def split_data(self):

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)

        def save_flod_data(i, part, data):
            data_folder = self.root + '/' + str(i)
            check_or_create(data_folder)
            test_folder = data_folder +'/'+part
            check_or_create(test_folder)
            test_path = test_folder + '/features.pkl'
            data.to_pickle(test_path)

        data = pd.read_pickle(self.root+'/vectors.pkl')
        data_num = len(data)
        k = 10 
        ratio = data_num // k
        data = data.sample(frac=1, random_state=666+random.randint(1,1000))

        for i in range(k):
            if i== k-1:
                test = data.iloc[ratio*i : ratio*(i+1)]
                dev = data.iloc[:ratio]
                train = data.iloc[ratio:ratio*i]
            else:
                test = data.iloc[ratio*i : ratio*(i+1)]
                dev = data.iloc[ratio*(i+1):ratio*(i+2)]
                train = data.iloc[:ratio*i].append(data.iloc[ratio*(i+2):])
             
            save_flod_data(i, 'test', test)
            save_flod_data(i, 'dev', dev)
            save_flod_data(i, 'train', train)       

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source(output_file='ast.pkl',option='existing')
        self.dictionary_and_embedding(128)
        print('split data...')
        self.split_data()
        print('end')

def parse_ast(tree):
    res = []
    for path, node in tree:
        pattern = javalang.tree.ReferenceType
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('ReferenceType_' + node.name)
        pattern = javalang.tree.MethodInvocation
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('MethodInvocation_' + node.member)
        pattern = javalang.tree.MethodDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('MethodDeclaration_' + node.name)
        pattern = javalang.tree.TypeDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('TypeDeclaration_' + node.name)
        pattern = javalang.tree.ClassDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('ClassDeclaration_' + node.name)
        pattern = javalang.tree.EnumDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('EnumDeclaration_' + node.name)
        pattern = javalang.tree.IfStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("ifstatement")
        pattern = javalang.tree.WhileStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("whilestatement")
        pattern = javalang.tree.DoStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("dostatement")
        pattern = javalang.tree.ForStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("forstatement")
        pattern = javalang.tree.AssertStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("assertstatement")
        pattern = javalang.tree.BreakStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("breakstatement")
        pattern = javalang.tree.ContinueStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("continuestatement")
        pattern = javalang.tree.ReturnStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("returnstatement")
        pattern = javalang.tree.ThrowStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("throwstatement")
        pattern = javalang.tree.SynchronizedStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("synchronizedstatement")
        pattern = javalang.tree.TryStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("trystatement")
        pattern = javalang.tree.SwitchStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("switchstatement")
        pattern = javalang.tree.BlockStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("blockstatement")
        pattern = javalang.tree.StatementExpression
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("statementexpression")
        pattern = javalang.tree.TryResource
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("tryresource")
        pattern = javalang.tree.CatchClause
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("catchclause")
        pattern = javalang.tree.CatchClauseParameter
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("catchclauseparameter")
        pattern = javalang.tree.SwitchStatementCase
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("switchstatementcase")
        pattern = javalang.tree.ForControl
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("forcontrol")
        pattern = javalang.tree.EnhancedForControl
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("enhancedforcontrol")
    return ' '.join(res)

projects = ['ant']
for project in projects:
    for i in range(1):
        print("-------------",project,"   ", i,"--------------------")
        ppl = Pipeline('7:1:2','data_cross/'+project+'/'+project+'_'+str(i)+'/')
        ppl.run()


