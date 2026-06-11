#### These are all the materials I am using to take the course AI Research fundamentals by Google Deepmind: https://www.skills.google/paths/3135 

**Something to Note:** This is just where I note down Important stuff, NO using AI, just my own reflections learning in just my own way no matter how messy it is 😅

```py
# Install the custom package for this course.
#  after initializing the project with uv and python 3.12. In pyproject.toml edit the following line: requires-python = ">=3.12,<3.13"
# then run
# uv add "git+https://github.com/google-deepmind/ai-foundations.git@main"
```

#### Some Papers & docs & blogs mentioned to go through later:

1. Tiny stories: https://arxiv.org/pdf/2305.07759 
2. Probabilities: https://www.deeplearningbook.org/contents/prob.html 
3. Conditional predictability: https://www.probabilitycourse.com/chapter1/1_4_0_conditional_probability.php
4. Language model alignment: https://arxiv.o: rg/pdf/2407.02273 
5. Why algorithms aare not moral agents: https://link.springer.com/article/10.1007/s00146-021-01189-x
6. gemma models impacthttps://deepmind.google/models/gemma/
7. uniprot: https://www.uniprot.org/help/ProtNLM
8. 

Reflection is an important thing in research, goes beyond methodology.
we have contextual cues, real world impact and stereotpes(generalizing features/aspects based on group of people, i.e I am a Kalenjin so I should be good in running 😂) & biases(unfair assumptions) - My reflections of this: context matters it makes a language model predict the next thing alaso depending on the data its trained on no wonder based on my experience with LLMS they are bad at generating good ideas, usually more generic. 

Probabilities must add up to 1 and must not be negative. Between 0 and 1.

Conditional probability - it expresses probability of next word given the previous one 
(How to write latex in markdown \mid - vertical bar, \frac{}{} - fraction, \cap AND sign(n), \mathcal{n} - caligraphic n)
$P(B \mid A) = \frac{A \cap B}{P(A)}$

n-grams --> sequence of *n words* that appear together in a text
take an example sentence - I was curious so
Unigrams($\mathcal{n}$=1) - Individual words --> I, was, curious, so
Bigrams($\mathcal{n}$=2) - pairs of words --> I was, was curious, curious so
Trigrams($\mathcal{n}$=3) - three word sequence --> I was curious, was curious so

Context window - part of preceding information that influences model's prediction of the next word
Tokenization - splitting sequences of text into words

Doing the coding challenges without AI is hard. I admit I had to look for a little help understanding the problem in lab_1_2 challenge 2

Trigram model generates more senssible continuations than bigram. even though to find a valid continuation starting sentence is haarder on the trigram than bigram

Trolley problem 

Comparing ngram and transformer, qualities:
1. Fluency - does it read naturally
2. Coherence - does it make logical sense and stay on the topic
3. Relevance - does it fit the context or prompt
4. Bias - does the output promote inequalities

Greedy sampling - making model output more deterministic to ensure token with higher probability is always chosen

inference - process of using ml model to make predictions
Reminder; lab1-> manually providing probabilities for inferencing, lab2-> using ngram model that performed inference based on last n tokens, lab3-> using transformer model gemma1B to perform inferencing(predicting the next token)

Model training - teaching a model to recognize patterns in data
Training process loop - Predict(look at input, generate a prediction) --> compare(prediction and actual target; the diff is called loss) --> Adjust(parameters based on the loss to improve next prediction; optimization)

ML Pipeline
1. Data - preprocessing
2. Train - pretraining
3. Finetune - SFT & RLHF
4. Evaluate - automated & human evaluation, benchmarks, 
5. Deploy - real world use and continuous monitoring

case study - google learnlm, protoNLM, gemma models

got an "InternalError: {{function_node __wrapped__Equal_device_/job:localhost/replica:0/task:0/device:GPU:0}} 'cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, 0, reinterpret_cast<CUstream>(stream), params, nullptr)' failed with 'CUDA_ERROR_INVALID_HANDLE' [Op:Equal] name: " error. effects of using diff environment other than the tutorials suggested. its good for debugging though.  

so I did some debugging. the obvious going to chatgpt(I had depleted my codex credits 😅) and try to ask for help. wrong idea in teh first place. so it gave me some not working solutions overcomplicating things. you know create this virtual environment and  this specific version of smoke_test code. after being unsuccessful with chatgpt's help. I decided to go the oldway google search with -ai(no ai searches) going through medium blog, tensorflow github past issues next is the documentation I had to follow the documentation that the error suggested 😂: https://www.tensorflow.org/install/gpu but it didn't work turns out I was missing something small that was just out in the open but my eyes didn't seem to catch it. I then stumbled upon this post in nvidia forum: https://forums.developer.nvidia.com/t/how-can-i-use-my-rtx-5060ti-to-training-model-by-using-tensorflow/350054 so the solution was a bit long and suggested conda(I am a uv guy) so decided to try one commad which is to install the tensorflow nightly version and it worked. I am here celebrating and I haven't even trained my own SML 😅. 
As I suspected another error: "LLVM ERROR: PTX version 8.5 does not support target 'sm_120'. Minimum required PTX version is 8.7. Either remove the PTX version to use the default, or increase it to at least 8.7."  

I had to create my own local notebook.
Solution another specific tensorflow version ```uv pip install tf-nightly[and-cuda]==2.21.0.dev20251017``` 
solving one error getting another is going to be my new normality in this course  "Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence" got this when trying to shuffle the dataset. 
its weird its like the warning occurs randomly. when I rerun the cell it sometimes it occurs other times not so a bit confused

Its getting tougher - just realized that the training function is form the ai_foundations the library that was causing the mismatching issues so the solution now is to either fork the repo and fix it manually or copy/donwload only the necessary code I need 
So the solution was copying the codes I need and editing it on the fly. forking the repo and updating the dependencies introduced even more bugs 
one bug I encountered had to do with this new library "tensorflow.python.framework.ops.EagerTensor" so jax doesn't recognize this.
fyi I had to dump all the code I needed in one file for instance the training file has 769 LOC multiple classes and functions.
All this debugging led me to forget that I came to this course to learn. as I speedrun the rest of the codes in *gdm_lab_1.5_local.ipynb* to ensure all run. I have to go back and see what I missed 