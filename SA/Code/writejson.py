# from distutils.file_util import write_file
# import json

# # Write list from text file 
# negative = open("D:\DCCNTT13.10.5\bot_tkinter\ChatBox/ChatBot/negative_text.txt","r").read().splitlines()
# positive = open("D:\DCCNTT13.10.5\bot_tkinter\ChatBox/ChatBot/positive_text.txt","r").read().splitlines()

# # Writing to json file
# with open("D:\DCCNTT13.10.5\bot_tkinter\ChatBox/ChatBot/intents004.json", "w") as outfile:
#     outfile.write('{"intents":[{"tag":"negative","patterns":[')
#     for item_neg in negative:
#         outfile.write('"' + item_neg.rstrip() + '",\n')
#     outfile.write('"0000"],"responses":[" this is negative"],"context":[""]},')
#     outfile.write('{"tag":"positive","patterns":[')
#     for item_pos in positive:
#         outfile.write('"' + item_pos.rstrip()  + '",\n')
#     outfile.write('"1111"],"responses":[" this is positive"],"context":[""]}]}')

import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load negation words
negation_words = ["not","no","never", "none","nobody","nothing","neither","nor", "nowhere","hardly","scarcely","barely","don't","doesn't","didn't","isn't","wasn't","aren't","weren't","cannot","couldn't", "won't","wouldn't","don't","doesn't","didn't", "shouldn't","wouldn't","couldn't","hasn't","hadn't","haven't","ain't"]

# Load positive and negative sentences
negative = open(os.path.join(BASE_DIR, "Data", "negative.txt"),"r").read().splitlines()
positive = open(os.path.join(BASE_DIR, "Data", "positive.txt"),"r").read().splitlines()

intents = {
    "intents": [
        {
            "tag": "negative",
            "patterns": [],
            "responses": [" This is negative"],
            "context": [""]
        },
        {
            "tag": "positive",
            "patterns": [],  
            "responses": [" This is positive"],
            "context": [""]
        }
    ]
}

for sentence in negative:
    if any(word in sentence for word in negation_words):
        intents["intents"][0]["patterns"].append(f'"{sentence}",')
    else:
        intents["intents"][1]["patterns"].append(f'"{sentence}",')
        
for sentence in positive:
    intents["intents"][1]["patterns"].append(f'"{sentence}",')

with open(os.path.join(BASE_DIR, "Data", "intent", "intents002.json"), "w") as outfile:
    json.dump(intents, outfile, indent=4)

