# VAE_RNN
## train
python main.py
## test_file
python main.py --mode test  
file : [data_dir]/source_test
## test_stdin
python main.py --mode stdin
## parameters
All settings in flags
## data
[data_dir]/:  
 * source_train  
   * one chinese sentence a line  
   * splitted  
 * source_test  
   * one chinese sentence a line  
   * splitted  
 * dict  
   * json file (word to id)  
   * {"word0": 0, "word1": 1, ...}  
 * word  
   * all words  
   * one word a line  
 * word_embd  
   * all word embeddings (in the same order with id in dict)  
   * one word embedding a line  
   * "word" d0 d1 ... dn  
   * the number of dimension in model.py  
## Environment
python2.7  
tensorflow-gpu(1.1.0)  

