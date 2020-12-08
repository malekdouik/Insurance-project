# Freeze versions of dependencies for now
import logging
logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

from aitextgen import aitextgen
from aitextgen.colab import mount_gdrive, copy_file_from_gdrive

'''
ai = aitextgen(tf_gpt2="124M", to_gpu=True)

file_name = "DataSet/EnvironementText.txt"


ai.train(file_name,
         line_by_line=False,
         from_cache=False,
         num_steps=5000,
         generate_every=1000,
         save_every=1000,
         save_gdrive=False,
         learning_rate=1e-4,
         batch_size=1, 
         )
         
ai.save()         
'''
# Load model
ai = aitextgen()

#ai.generate()



# Use the model
ai.generate(n=1,
            batch_size=100,
            prompt="I'am a legend ",
            max_length=200,
            temperature=3.0,
            top_p=0.9)
            
#'''           
