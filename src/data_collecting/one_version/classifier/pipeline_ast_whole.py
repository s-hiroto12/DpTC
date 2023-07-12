import pandas as pd
import os
import sys
from javalang.ast import Node
import javalang
sys.setrecursionlimit(10000)

class Pipeline:
    def __init__(self, root):
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

    # split data for training
    def split_data(self):
        train = self.sources

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)
        train_path = self.root+'train/'
        check_or_create(train_path)
        self.train_file_path = train_path + 'train_.pkl'
        train.to_pickle(self.train_file_path)        

    # generate block sequences with index representations
    def generate_ast_seqs(self,data_path):
        trees = pd.read_pickle(data_path)
        trees['code'] = trees['code'].apply(parse_ast)
        trees.to_pickle(self.root+'vectors.pkl') 

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, size):
        self.size = size
        
        sequences = pd.read_pickle(self.root+input_file)
        sequences = sequences['code'].values
        # print(sequences.shape)

        corpus = []
        n = len(sequences)
        for i in range(n):
            seq = str(sequences[i])
            line = seq.split()
            corpus.append(line)
        # print(corpus)

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(sentences=corpus, size=size, workers=16, sg=1, min_count=1)
        if not os.path.exists(self.root+'train/embedding'):
            os.mkdir(self.root+'train/embedding')
        w2v.save(self.root+'train/embedding/node_w2v_' + str(size))  

    # 生成类的特征，保存为features.pkl
    def save_ast_feature(self, input_file, size):
        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec.load(self.root+'train/embedding/node_w2v_'+str(size))

        import numpy as np
        def trans2feature(sequence):
            result = np.array([0.0]*128)
            sequence = str(sequence).split()
            for seq in sequence:
                # 获得词对应的词向量
                vector = w2v.wv.__getitem__(seq)
                result += vector
            return result

        trees_seq = pd.read_pickle(self.root+input_file)
        trees_seq['code'] = trees_seq['code'].apply(trans2feature)
        features = trees_seq['code'].values.tolist()
        features = np.array(features)
        np.save(self.root+'/features.npy', features)

        labels = trees_seq['label'].values.tolist()
        np.save(self.root+'/labels.npy', labels)

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source(output_file='ast.pkl',option='existing')
        print('split data...')
        self.split_data()
        print('generate ast sequences...')
        self.generate_ast_seqs(self.train_file_path)
        # print('train word embedding...')
        self.dictionary_and_embedding('vectors.pkl', 128)
        # print('get and save features...')
        self.save_ast_feature('vectors.pkl', 128)
        print('end')

def parse_ast(tree):
    res = []
    for path, node in tree:
        # res.append(node)
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


projects = ["ambari"]
for project in projects:
    print("-------------",project,"--------------------")
    ppl = Pipeline('data/ast-whole/'+project+'/')
    ppl.run()
