# Create GUI
import os
import sys
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk

# Thiết lập đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'DOTA'))

# Chỉ sử dụng BERT
from DOTA.predict_bert import BertPredictor
from RunProgram import but_scan # Vẫn giữ but_scan để làm sạch văn bản

try:
    bert_predictor = BertPredictor()
    HAS_BERT = True
except Exception as e:
    print(f"Lỗi: Không thể tải mô hình BERT. {e}")
    HAS_BERT = False

def sendText():
    per_msg = box_send.get(1.0, END).strip()
    box_send.delete(0.0, END)
    box_rel.config(state=NORMAL)
    
    if not per_msg:
        res = "Please try sending again ! Thank you !"
    elif per_msg.lower() == "hello":
        res = "Hello there! I am powered by BERT."
    else:
        # Làm sạch câu (loại bỏ ký tự đặc biệt, chuyển lowercase)
        clean_msg = but_scan(per_msg)
        
        if HAS_BERT:
            # Dự đoán bằng BERT
            res = bert_predictor.get_response(clean_msg)
        else:
            res = "Mô hình BERT chưa được tải. Vui lòng kiểm tra lại!"
        
    CSU_color = "#FF3300"
    box_rel.insert(END, " You: " + per_msg + "\n", "You_chat")
    box_rel.insert(END, " Our Bot (BERT): ")
    box_rel.insert(END, res + "\n\n", "CSU_chat")
    box_rel.tag_config("CSU_chat", justify=LEFT, foreground=CSU_color)
    box_rel.config(state=DISABLED)
    box_rel.yview(END)

# Cấu hình giao diện
Bot_name = "BERT BOT"
BG_GUI = os.path.join(BASE_DIR, "images", "Chatbot_bg.png")
BG_send = os.path.join(BASE_DIR, "images", "sendbtn.png")

chatbot = tk.Tk()
chatbot.title("CHAT SYSTEM ")
chatbot.iconbitmap(os.path.join(BASE_DIR, "images", "Chatbot_icon.ico"))
chatbot.resizable(width=False, height=False)
chatbot.configure(width=500, height=650)

# Load background
try:
    load_bg = Image.open(BG_GUI)
    img_bg = ImageTk.PhotoImage(load_bg)
    img = Label(chatbot, image=img_bg)
    img.place(x=0, y=0)
except:
    chatbot.configure(bg="#DFFFFF")

# Head label
name_sys = Label(chatbot, text=Bot_name, fg="#000000", bd=0, bg="#DFFFFF")
name_sys.config(font=("Bookman Old Style", 25, "bold"))
name_sys.place(y=10, x=200)

# Label trạng thái mô hình
status_label = Label(chatbot, text="Engine: BERT Intelligence v1.0", fg="#008000", bg="#DFFFFF", font=("Arial", 9, "italic"))
status_label.place(y=50, x=175)

# Box message
box_rel = Text(chatbot, width=43, height=21, font=("Arial", 14), bd=1, bg="#FAFAFA")
box_rel.place(y=80, x=11)
box_rel.config(state=DISABLED)

# Box send
box_send = Text(chatbot, width=35, height=4, font=("Arial", 14), bd=1, bg="#FAFAFA")
box_send.place(y=540, x=11)

# Send button
try:
    send_bg = Image.open(BG_send)
    send = ImageTk.PhotoImage(send_bg)
    sendbtn = Button(chatbot, bg="#DFFFFF", bd=0, image=send, width=75, height=75, command=sendText)
except:
    sendbtn = Button(chatbot, text="SEND", bg="#FF3300", fg="white", font=("Arial", 12, "bold"), command=sendText)
    
sendbtn.place(y=547, x=410)
box_send.bind("<Return>", lambda x: sendText())

chatbot.mainloop()