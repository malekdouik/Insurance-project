# On fait le scraping de WIKISTRIKE parce que on a pas trouvé son flux RSS 
# On fait le scraping avec scrapy

import scrapy
from scrapy.crawler import CrawlerProcess
import json
from newspaper import Article
import pandas as pd
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import torch
from nltk.tokenize import sent_tokenize
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd

# https://www.wikistrike.com/

####################################################################################
## Scraping 
####################################################################################
class ArticleScraper(scrapy.Spider):
    name = 'wikistrike'

    headers = {
        'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
    }
    
   

    def start_requests(self):
        liste_sujet=["https://www.wikistrike.com/tag/ecologie%20-%20conso%20-%20biodiversite%20-%20energie/",
        	      "https://www.wikistrike.com/tag/astronomie%20-%20espace/",
        	      "https://www.wikistrike.com/tag/terre%20et%20climat/"]
        for i in liste_sujet:	      
            yield scrapy.Request(
                    url=i,
                    headers=self.headers,
                    callback=self.parse_cards                 
                    )

    # parse article cards
    def parse_cards(self, response):
         

        # loop over article cards
        for card in response.css('#home_featured2 li'):
            features = {
               'titre': card.css('.post-title a::text').get(),
               'link': card.css('.post-title a::attr(href)').get(),
               
            }
            print(json.dumps(features,indent=2))
            url = json.dumps(features['link'])
            print(url)
            
            
            with open('list_lien_wikistrikeEnv.txt', 'a') as f:
                f.write(url+ '\n')


####################################################################################
## Get Article   
####################################################################################     
def GetArticle ():
    with open('list_lien_wikistrikeEnv.txt') as f:
        lines = f.read().splitlines()
    
    # nettoyer la liste
    lines = [x for x in lines if x != 'null']
    for x in range(len(lines)):
        lines[x]=lines[x].replace('"','')
        
    # Get texte Article
    article_text=[]
    article_titre=[]
    for url in lines:
        article = Article(url)
        article.download()
        article.html
        try:
            article.parse()
        except ValueError:
            print("Erreure dans le parsing !")    
        
        article_text.append(article.text)
        article_titre.append(article.title)
    
    
    # Nettoyer le texte des articles
    for i in range(len(article_text)):
        article_text[i]= re.sub(r"[-()\"#/@;:<>{}|.?,«»<<>>]", "", article_text[i])
    
    ################################# 10 premiers texte du texte 
    # Prendre les 10 premiers mots et supprimer les \n du texte
    newliste_article=[]
    for i in article_text:
        newliste_article.append(i.strip())

    newliste_article = [i.split()[:10] for i in newliste_article]
    # Rq : split nous donne un liste of liste donc => on convertit liste of list à list
    newliste_article2=[]
    for sublist in newliste_article:
      sublist = " ".join(sublist)
      newliste_article2.append(sublist)
   
    ################################# Traduction Texte en anglais pour faire un résumé aprés 
    def clean_text(text):
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\n\n", " ", text)
        text = text.strip(" ")
        text = re.sub(' +',' ', text).strip() # gets rid of multiple spaces and replace with a single
        return text

    def translate(text):
        if text is None or text == "":
            return "Error",

        #batch input + sentence tokenization
        batch = tokenizer.prepare_seq2seq_batch(sent_tokenize(text))
        #run model
        translated = model.generate(**batch)
        tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return " ".join(tgt_text)


    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")   
    
    article_textEng=article_text
    k=0
    for i in range(len(article_textEng)):
        article_textEng[i]=clean_text(article_textEng[i])
        article_textEng[i]=translate(article_textEng[i])
        k+=1
        print("Traduction fr -> Eng terminée : "+str(k))
    

    ################################# Résumé Texte par t5
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')
    
    textRezumEng=[]
    k=0
    for i in article_textEng:
        preprocess_text = i.strip().replace("\n","")
        t5_prepared_Text = "summarize: "+preprocess_text
        tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    
        # summmarize 
        summary_ids = model.generate(tokenized_text,
                                      num_beams=4,
                                      no_repeat_ngram_size=2,
                                      min_length=15,
                                      max_length=100,
                                      early_stopping=True)
        
        k+=1
        print("résumé terminé : "+str(k))                              
        output_rezumFr = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        textRezumEng.append(output_rezumFr) 
    
    ################################################## traduire texte de l'en -> fr 
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")   
     
    k=0
    for i in range(len(textRezumEng)):
        textRezumEng[i]=clean_text(textRezumEng[i])
        textRezumEng[i]=translate(textRezumEng[i])
        k+=1
        print("Traduction Eng -> Fr terminée : "+str(k))    
    
    ################################################### Diviser le résumé en
    textRezumEng2 = textRezumEng
    for i in range(len(textRezumEng2)):
        textRezumEng2[i]=textRezumEng2[i].split(".")
       
    ################################################### Mettre le tout dans un dataframe
    d = {'Url':lines,'Titre':article_titre,'Texte':article_text,'10 1ers mots':newliste_article2,'Résumé':textRezumEng,'phrases de resumé':textRezumEng2}
    
    df = pd.DataFrame(d)
    df.to_csv("wikistrike.csv")

    

# main driver
if __name__ == '__main__':
    # run scraper
    open('list_lien_wikistrikeEnv.txt', 'w').close()
    process = CrawlerProcess()
    process.crawl(ArticleScraper)
    process.start()
    
    GetArticle()	


# article_text 
# article_titre
# lines :url
# newliste_article2 : 10 mots 
# liste_outputFr : rezumé en fr direct
# article_textEng : rezumé en fr en se passant par en ang 
# textRezumEng : contient le text résumé en ang 
# textRezumEng : traduire texte de eng en fr 
# textRezumEng2 : Diviser le résumé en plusieurs phrases 
