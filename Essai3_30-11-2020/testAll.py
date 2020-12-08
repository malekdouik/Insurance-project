from aitextgen import aitextgen
from aitextgen.colab import mount_gdrive, copy_file_from_gdrive
from pandas_ods_reader import read_ods #pip install pandas-ods-reader

### Pour la traduction
import nltk
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
from transformers import MarianMTModel, MarianTokenizer
nltk.download('punkt')
import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import time
start_time = time.time()

# Load model
ai = aitextgen()


# convert column dataframe to list

#df = read_ods('debut_phrases_fr.ods',1)
#liste_input = df.iloc[:,0]


df = pd.read_csv('debut_phrases_fr_Article.csv')
liste_input = df.iloc[:,1]
print(liste_input)

# Traduire liste input en anglais 
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
 
liste_inputEng=liste_input 

for i in range(len(liste_inputEng)):
  liste_inputEng[i]=clean_text(liste_inputEng[i])
  liste_inputEng[i]=translate(liste_inputEng[i])


print("Input après la traduction en anglais")
#print(liste_inputEng)


#####################################################################################################
# Règles génération

from nltk import tokenize
from nltk.corpus import wordnet as wn

nltk.download('averaged_perceptron_tagger')


# Pour la condition 4 
def print_sentences(text):
    test = tokenize.punkt.PunktSentenceTokenizer()
    return test.sentences_from_text(text)

def re_generationRegles(text,input_text):

  # supprimer les débuts de phrases du texte généré car ils sont nos débuts de phrases
  text = text.replace(input_text,"")

  liste_stopfirst_sentence=['United States', 'governor', 'American','Donald Trump','the U.S','US','american','M. Trump','Trump','he said']
  # Condition 1:
  for i in list(liste_stopfirst_sentence):
    if i in text:
      print("Condition 1 n'est pas validée donc text non valide régenration : il existe États-Unis ou bien gouverneur ou bien américain") 
      return True

  # Condition 2:
  # » => " 
  if '''"''' in text:

    # Prend la sous chaine qui contient le caractère »
    string=text.split('''"''')[0]
    # Dans cette nouvelle sous chaine il faut trouver caractère 
    if '''"''' not in string:
    #  print("Condition 2 n'est pas validee donc text non valide régenration : on trouve le caractère » et qu'il n'est pas précédé par «") 
       print("Condition 2 n'est pas validee donc text non valide régenration : on trouve le caractère » et qu'il n'est pas précédé par «")
       return True

  # Condition 3:
  # Avoir le 1er mot
  
  print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
  print(text)
  Firstword = text.split()[0]
  
  # re.sub("\s\s+", " ", s) # supprimer plus que 2 espaces car cela causes des problems pour la detection des majiscule
  print(Firstword)
  
  if Firstword.istitle() == True:
    print("Condition 3 n'est pas validée donc text non valide régenration : ça ne commence pas par une majuscule")
    return True

 # Condition 4: La 1ere phrase doit avoir:
 # Avoir la 1ere phrase
  first_sentence = print_sentences(text)[0]
  
  text = nltk.word_tokenize(first_sentence)
  nltk.pos_tag(text)

  k=0
  for i in list(nltk.pos_tag(text)):
    if ('VB' or 'VBP' or 'VBG')  in i :
      k+=1
  if k == 0 :

    print("Condition 4 n'est pas validée la première phrase n'a pas de verbe ")
    return True
  
  
  
  return False      
  #if k!=0 :
  #  print("cette phrase contient des verbe ")

            
######################################################################################################

# Génération du texte par aitextgen
liste_outputEng=[]
k=0
for i in range(len(liste_inputEng)):
    
    #liste_input2.append(liste_input[i])
    generate = True
    while generate is True:
        print("---------------------------:")
        print("Traitement génération phrase : "+str(k))
        print("Début de phrase : "+liste_inputEng[i])
    
        output=ai.generate_one(batch_size=100,
            prompt=liste_inputEng[i],
            max_length=25,
            temperature=1.0,
            top_p=0.9)
            
        print("La phrase générée :" + output)      
        print("------- :")
         
        if(re_generationRegles(output,liste_inputEng[i]) == False):
            generate = False
            break
            
     
    k+=1


    """
    # nettoyer l'output du modele
    for j in range(len(output)):
        output[j]=output[j].replace("\x1b[1m","")
        output[j]=output[j].replace("\x1b[0m","")
        liste_output.append(output[j])
     """  
    liste_outputEng.append(output)

print("Texte généré en anglais:")
print(liste_outputEng)

# Traduire le texte généré en francais et le mettre dans un dataFrame

def translate(text):
    if text is None or text == "":
        return "Error",

    #batch input + sentence tokenization
    batch = tokenizer.prepare_seq2seq_batch(sent_tokenize(text))

    #run model
    translated = model.generate(**batch)
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    return " ".join(tgt_text)


model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

liste_outputFr = liste_outputEng

k=0
for i in range(len(liste_outputFr)):
  liste_outputFr[i]=clean_text(liste_outputFr[i])
  liste_outputFr[i]=translate(liste_outputFr[i])
  k+=1
  print("traduction : "+str(k)+" treminée")    


#liste_input = df.iloc[:,0]
df = pd.read_csv('debut_phrases_fr_Article.csv')
liste_input = df.iloc[:,1]

dictt = {'inputFr':liste_input,'TextFr':liste_outputFr}
df = pd.DataFrame(dictt)


df.to_csv("Resultat_Generation/resultatScraping13Rezum_10mots.ods") 

print("--- %s seconds ---" % (time.time() - start_time)) 

