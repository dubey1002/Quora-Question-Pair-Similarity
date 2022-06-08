# Quora-Question-Pair-Similarity

<h1> 1. Business Problem </h1>
<h2> 1.1 Description </h2>
<p>Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.</p>
<p>
Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.
</p>
<br>
> Credits: Kaggle 


__ Problem Statement __
- Identify which questions asked on Quora are duplicates of questions that have already been asked. 
- This could be useful to instantly provide answers to questions that have already been answered. 
- We are tasked with predicting whether a pair of questions are duplicates or not. 

<h2> 1.2 Sources/Useful Links</h2>
- Source : https://www.kaggle.com/c/quora-question-pairs
<br><br>
____ Useful Links ____<br>
- Discussions : https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb/comments<br>
- Kaggle Winning Solution and other approaches: https://www.dropbox.com/sh/93968nfnrzh8bp5/AACZdtsApc1QSTQc7X0H3QZ5a?dl=0<br>
- Blog 1 : https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning<br>
- Blog 2 : https://towardsdatascience.com/identifying-duplicate-questions-on-quora-top-12-on-kaggle-4c1cf93f1c30<br>

<h2>1.3 Real world/Business Objectives and Constraints </h2>
1. The cost of a mis-classification can be very high.<br>
2. You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.<br>
3. No strict latency concerns.<br>
4. Interpretability is partially important.<br>

<h1>2. Machine Learning Probelm </h1>
<h2> 2.1 Data </h2>
<h3> 2.1.1 Data Overview </h3>
<p> 
- Data will be in a file Train.csv <br>
- Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate <br>
- Size of Train.csv - 60MB <br>
- Number of rows in Train.csv = 404,290
</p>
<h3> 2.1.2 Example Data point </h3>
<pre>
"id","qid1","qid2","question1","question2","is_duplicate"
"0","1","2","What is the step by step guide to invest in share market in india?","What is the step by step guide to invest in share market?","0"
"1","3","4","What is the story of Kohinoor (Koh-i-Noor) Diamond?","What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?","0"
"7","15","16","How can I be a good geologist?","What should I do to be a great geologist?","1"
"11","23","24","How do I read and find my YouTube comments?","How can I see all my Youtube comments?","1"
</pre>
<h2> 2.2 Mapping the real world problem to an ML problem </h2>
<h3> 2.2.1 Type of Machine Leaning Problem </h3>
<p> It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not. </p>
<h3> 2.2.2 Performance Metric </h3>
Source: https://www.kaggle.com/c/quora-question-pairs#evaluation

Metric(s): 
* log-loss : https://www.kaggle.com/wiki/LogarithmicLoss
* Binary Confusion Matrix

<h2>3.3 Basic Feature Extraction (before cleaning) </h2>
Let us now construct a few features like:<br>
<b>freq_qid1<b> = Frequency of qid1's<br>
<b>freq_qid2<b> = Frequency of qid2's <br>
<b>q1len<b> = Length of q1<br>
<b>q2len<b> = Length of q2<br>
<b>q1_n_words<b> = Number of words in Question 1<br>
<b>q2_n_words<b> = Number of words in Question 2<br>
<b>word_Common<b> = (Number of common unique words in Question 1 and Question 2)<br>
<b>word_Total<b> = (Total num of words in Question 1 + Total num of words in Question 2)<br>
<b>word_share<b> = (word_common)/(word_Total)<br>
<b>freq_q1+freq_q2<b> = sum total of frequency of qid1 and qid2 <br>
<b>freq_q1-freq_q2<b> = absolute difference of frequency of qid1 and qid2 <br>
 


