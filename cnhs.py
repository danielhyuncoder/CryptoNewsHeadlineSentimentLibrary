import requests
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf

class TextBasedTokenizer:
  def decode_file(self, file_name):
    lines=open(file_name, "r").readlines()
    for line in lines:
      s=line.split(":")
      if len(s) < 2:
        continue
      self.word_index["".join(s[0].split(" "))]=int("".join("".join(s[1].split(" ")).split("\n")))
  def __init__(self, file_name):
    self.word_index={}
    self.decode_file(file_name)
  def text_to_sequence(self, text):
    words=text.split(" ")
    sequence = []
    for word in words:
      if word not in self.word_index:
        sequence.append(0)
      else:
        sequence.append(self.word_index[word])
    return sequence


class SentimentModel:
  def __init__(self, model_file_path: str, tokenizer_file_path: str):
    self.model=tf.keras.models.load_model(model_file_path)
    self.tokenizer=TextBasedTokenizer(tokenizer_file_path)
    self.pred_outputs=[0,1,2]
  def set_prediction_outputs(self, output_array):
    self.pred_outputs=output_array
  def get_prediction(self, txt: str):
    X=[self.tokenizer.text_to_sequence(txt)]
    predictions=self.model.predict(X, verbose=0)
    return self.pred_outputs[np.argmax(predictions[0])]
 
class CryptoSentimentHeadlineScanner:
  def __init__(self, selected_model: SentimentModel):
    self.sentimentModel=selected_model
    self.months_map_abbr = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12
    }
    self.abbreviated_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
  def converted_date(self, month: int, day: int, year: int):
    return (year*1000)+(month*100)+day
  def getOverallSentimentOfCryptoByPage(self,coin_name:str, start_page:int, end_page:int):
    coin_name=coin_name.lower()
    if start_page<1:
      print("Start Page has to be greater than or equal 1")
      return
    page=start_page
    text_X=[]
    while page<=end_page:
      req=requests.get("https://www.coindesk.com/tag/"+coin_name+"/"+str(page)+'/')
      parser=BeautifulSoup(req.text, "html.parser") 
      all_titles=parser.find_all("a",class_="card-title")
      all_contents=parser.find_all("span",class_="content-text")
      for i in range(len(all_titles)):
        text_X.append(all_titles[i].text+". "+all_contents[i].text)
      page+=1 

    neutral=0
    pos=0
    neg=0
    for txt in text_X:
      pred=self.sentimentModel.get_prediction(str(txt))
      if pred == 0:
        neg+=1
      elif pred==1:
        neutral+=1
      elif pred==2:
        pos+=1
    return [neg/len(text_X), neutral/len(text_X), pos/len(text_X)]
  def getOverallSentimentOfCryptoByDate(self, coin_name: str, start_date: int, end_date: int):
    coin_name=coin_name.lower()
    if start_date>end_date:
      print("Error: Cannot have start date greater than the ending date")
      return
    iteration_count=1
    end=False
    while end==False:
      req=requests.get("https://www.coindesk.com/tag/"+coin_name+"/"+str(iteration_count)+'/')
      parser=BeautifulSoup(req.text, "html.parser") 
      all_dates=parser.find_all("span", class_="iOUkmj")
      for date in all_dates:
         d=" ".join(" ".join(date.text.split(" ")).split(",")).split(" ")
         if d[0] not in self.abbreviated_months or "at" in date.text:
          continue
         conv_date=self.converted_date(self.months_map_abbr[d[0]], int(d[1]), int(d[3]))
         if conv_date<=end_date:
           end=True
           break
      if end:
        break
      iteration_count+=1
    text_X=[]
    end=False
    while end==False:
      req=requests.get("https://www.coindesk.com/tag/"+coin_name+"/"+str(iteration_count)+'/')
      parser=BeautifulSoup(req.text, "html.parser") 
      all_dates=parser.find_all("span", class_="iOUkmj")
      date_begin_index=0
      date_index=-1
      ind=0
      for date in all_dates:
         d=" ".join(" ".join(date.text.split(" ")).split(",")).split(" ")
         if d[0] not in self.abbreviated_months or "at" in date.text:
          continue
         conv_date=self.converted_date(self.months_map_abbr[d[0]], int(d[1]), int(d[3]))
         if conv_date<start_date:
           date_index=ind
           end=True
           break
         if conv_date>end_date:
           date_begin_index=ind+1
         ind+=1
      all_titles=parser.find_all("a",class_="card-title")
      all_contents=parser.find_all("span",class_="content-text")
      while date_begin_index<len(all_titles):
        if date_index==date_begin_index:
           break
        text_X.append(all_titles[date_begin_index].text+". "+all_contents[date_begin_index].text)
        date_begin_index+=1
      iteration_count+=1
    #get predictions
    neutral=0
    pos=0
    neg=0
    for txt in text_X:
      pred=self.sentimentModel.get_prediction(str(txt))
      if pred == 0:
        neg+=1
      elif pred==1:
        neutral+=1
      elif pred==2:
        pos+=1
    return [neg/len(text_X), neutral/len(text_X), pos/len(text_X)]
