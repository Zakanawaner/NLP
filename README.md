# NLP

<p>Here I tested three approaches for Text Generation.</p>

<h3>1st_approach_torch.py</h3>

<p>Using Pytorch and word embedding. Original post: https://machinetalk.org/2019/02/08/text-generation-with-pytorch/</p>
<p>I added Spacy library for testing their embedding algorythm.</p>
<p>To use it, modify the train_file hyper paraeter to point to the *.txt file you wanna train the model with. You can find 
in Source/TXT/merged.txt a file made with 178 books in spanish. Once you have defined your train_file (and any other hyper 
parameter that you want to modify), you can execute the script</p>

<h3>2nd_approach_tf.py</h3>

<p>Using Tensorflow and one hot encoding. Original post: https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/</p>
<p>To use it, modify the path to the *.txt file you wanna train the model with, in line 9. You can find 
in Source/TXT/merged.txt a file made with 178 books in spanish. Once you have defined your train file, you can execute the script</p>

<h3>3rd_approach_transformers.py</h3>

<p>Using Transformers under Pytorch. Original post:  https://towardsdatascience.com/train-a-gpt-2-transformer-to-write-harry-potter-books-edf8b2e3f3db</p>
<p>To use it, call it to train the model with the following command:</p>
<p>!python /content/run_lm_finetuning.py 
    --output_dir=output 
    --model_type=gpt2 
    --model_name_or_path=gpt2-medium 
    --do_train 
    --train_data_file='/content/TXT/merged.txt' 
    --do_eval 
    --eval_data_file='/content/TXT/merged copia.txt' 
    --overwrite_output_dir 
    --block_size=200 
    --per_gpu_train_batch_size=1 
    --save_steps 4000 
    --num_train_epochs=4</p>
<p>If you want to know more about the arguments you can use, please go to the original post</p>
<p>For testing, call the test script with the following command:</p>
<p>!python /content/run_generation.py 
    --model_type=gpt2 
    --model_name_or_path=output 
    --length 300 
    --prompt "Todo era caos, y nadie alcanzaba a comprender." 
    --temperature=1.0</p>
<p>You can fin in Source/ two scripts to convert a pdf file to text for data processing</p>
