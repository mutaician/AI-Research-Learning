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