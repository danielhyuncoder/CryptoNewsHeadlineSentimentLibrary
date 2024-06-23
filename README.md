<h1>CryptoNewsHeadlineSentimentLibrary</h1>
<br />
<h2>Requirements: </h2>
<ul>
  <li>Python 3+</li>
  <li>Beautiful Soup</li>
  <li>Tensorflow 2</li>
  <li>Keras 3</li>
  <li>Numpy</li>
  <li>Requests</li>
</ul>
<br />
<h2>Introduction: </h2>
<br />
<p>The CryptoNewsHeadlineSentimentLibrary (or cnhs for short) is a small Python library built off of BeautifulSoup and Tensorflow. It scrapes news headlines from "coindesk.com" and delivers an overall sentiment of all the news headlines that it collected via a Tensorflow Neural Network.</p>
<br/>
<h2>Start: </h2>
<p>First install all the files from this repo (Specifically "cnhs.py" and "sentiment_tokenizer.txt" are both necessary). The file titled "training.py", however, is unecessary, as it only reveals how the sample model for this project was trained.</p>
<br />
<p>For quickstart, you can install the custom neural network that was trained for this project <a href="https://drive.google.com/drive/folders/1MMhvPmNXXVQ9BPB-7-Hj5ZmdOZpXUlGj?usp=sharing">here</a> and utilize it for your own purposes. However, if you want to use your own Neural Network model, you can.</p>
<h2>Documentation: </h2>
<br />
<h3>SentimentModel class: </h3>
<ol>
  <li>Class declaration: __init__(model_file_path: str, tokenizer_file_path: str) - both arguments are required and is used to load in both the specific Tensorflow model (based on the file path) along with the saved tokenizer data (In this case, is "sentiment_tokenizer.txt")</li>
  <li>Set Function: set_prediction_outputs(output_array) - configures the labels to how the model identifies its predictions (after the numpy.argmax function is applied) as either neutral, negative, or positive. The default value for this is [0, 1, 2] if not called.</li>
  <li>Get Function: get_prediction(txt: str) - through the provided string, txt, the SentimentModel class by default will convert the string into a sequence, after of which, will be inputed into the model and will return the translated(Indexes the label of the previously set predicted outputs) numpy.argmax result of the predicted array.</li>
</ol>
<h3>CryptoSentimentHeadlineScanner class: </h3>
<ol>
  <li>Class declaration: __init__(selected_model: SentimentModel) - declaration of the class requires an input parameter of type SentimentModel</li>
  <li>Get Function: converted_date(month: int, day: int, year: int) - gets the current month, day, and year. Returns the hashed date of the arguments.</li>
  <li>Get Function: getOverallSentimentOfCryptoByPage(coin_name:str, start_page:int, end_page:int) - gets the coin of which needs to be scanned, and the start and end page (on coindesk) of which the user wishes to scan. It returns a percentage distribution (similar to a what a softmax activation function returns) of how many news headlines were negative, positive, or neutral.</li>
  <li>Get Function: getOverallSentimentOfCryptoByDate(coin_name: str, start_date: int, end_date: int) - gets the coin of which needs to be scanned, and the start and end date range of the news headlines the user wishes to scan (note both the start and end date have to be hashed by the converted_date function in order to work properly). It returns a percentage distribution (similar to a what a softmax activation function returns) of how many news headlines were negative, positive, or neutral.</li>
</ol>
