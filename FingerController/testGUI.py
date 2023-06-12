from tkinter import *
import subprocess

tk = Tk()
tk.title("GUI 프로그램")
tk.geometry("300x200")

process = None

def start():
    global process
    process = subprocess.Popen(["python", "main_program.py"])
    status_label.config(text="실행 중...")

def stop():
    global process
    if process is not None:
        process.terminate()
        status_label.config(text="종료됨")


status_label = Label(tk, text="대기 중", font=("Arial", 14))
status_label.pack(pady=20)


button_frame = Frame(tk)
button_frame.pack()

start_button = Button(button_frame, text="START", command=start, width=10, font=("Arial", 12))
start_button.pack(side=LEFT, padx=10)

stop_button = Button(button_frame, text="STOP", command=stop, width=10, font=("Arial", 12))
stop_button.pack(side=LEFT, padx=10)

tk.mainloop()
