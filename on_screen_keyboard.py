from pyautogui import press
from threading import Thread
import tkinter
from tkinter import Label, ttk

window = tkinter.Tk()

window.title('OnScreen Keyboard')
window.call('wm', 'attributes', '.', '-topmost', '1')
unit = 76
winHeight = unit*8
winWidth = unit*22
window.geometry(f'{winHeight}x{winWidth}')
window.maxsize(width=winWidth, height=winHeight)
window.minsize(width=winWidth, height=winHeight)

# * Style Setup

style = ttk.Style()
window.configure(bg='gray27')
style.configure('TButton', background='gray21')
style.configure('TButton', foreground='white')
_displayFont = ("Arial", 18, "bold")

# * Display Setup

input = tkinter.StringVar(window, "")
display = tkinter.Entry(window, state='readonly',
                        textvariable=input, font=_displayFont)
display.grid(row=0, column=0, rowspan=1, columnspan=99, ipadx=900, ipady=34)


# * Num Pad
numPad = tkinter.LabelFrame(window, padx=0, pady=0,
                            background='gray27', border=0)
numPad.grid(row=2, column=3, pady=10, padx=10)

# * Nav Pad
navPad = tkinter.LabelFrame(window, padx=0, pady=0,
                            background='gray27', border=0)
navPad.grid(row=2, column=2, pady=10, padx=10)

# * Alpha Pad
alphaPad = tkinter.LabelFrame(
    window, padx=0, pady=0, background='gray27', border=0)
alphaPad.grid(row=2, column=1, pady=10, padx=10)

# * Func Pad
funcPad = tkinter.LabelFrame(
    window, padx=0, pady=0, background='gray27', border=0)
funcPad.grid(row=1, column=0, columnspan=6, pady=10, padx=10, sticky="W")


capsLock=False
def keyPressed(key):
    global capsLock

    if(key=='capslock'):
        press('capslock')
        if(capsLock):
            capsLock=False
        else:
            capsLock=True
    if(key=='capslock'):
        pass
    elif(key=='space'):
        input.set(str(input.get())+" ")
    else:
        if(capsLock):
            input.set(str(input.get())+str(key).upper())
        else:
            input.set(str(input.get())+key)
    
    

def createButton(panel, r, c, context, key, font, pad, size, span):
    btn = tkinter.Button(panel, text=context, anchor="nw", width=size[0], height=size[1],
                         command=lambda: keyPressed(key), font=font)
    btn.configure(background='gray21', highlightbackground='gray27',
                  foreground="white", highlightcolor="gray27")
    btn.grid(row=r, column=c,
             ipadx=pad[0], ipady=pad[1], rowspan=span[0], columnspan=span[1])


def createAlphanumericPad(panel):

    # * Create layers

    layerOne = tkinter.LabelFrame(
        panel, padx=0, pady=0, background='gray27', border=0)
    layerOne.grid(row=1, column=0, pady=0, padx=0, sticky="W")

    layerTwo = tkinter.LabelFrame(
        panel, padx=0, pady=0, background='gray27', border=0)
    layerTwo.grid(row=2, column=0, pady=0, padx=0, sticky="W")

    layerThree = tkinter.LabelFrame(
        panel, padx=0, pady=0, background='gray27', border=0)
    layerThree.grid(row=3, column=0, pady=0, padx=0, sticky="W")

    layerFour = tkinter.LabelFrame(
        panel, padx=0, pady=0, background='gray27', border=0)
    layerFour.grid(row=4, column=0, pady=0, padx=0, sticky="W")

    layerFive = tkinter.LabelFrame(
        panel, padx=0, pady=0, background='gray27', border=0)
    layerFive.grid(row=5, column=0, pady=0, padx=0, sticky="W")

    createButton(layerOne, 0, 0, "~\n`", "`",
                 ("Arial", 12, "bold"), [8, 10], [3, 0], [1, 1])
    createButton(layerOne, 0, 1, "!\n1", "!1",
                 ("Arial", 12, "bold"), [8, 10], [3, 0], [1, 1])
    createButton(layerOne, 0, 2, "@\n2", "@2",
                 ("Arial", 12, "bold"), [8, 10], [3, 0], [1, 1])
    createButton(layerOne, 0, 3, "#\n3", "#3",
                 ("Arial", 12, "bold"), [8, 10], [3, 0], [1, 1])
    createButton(layerOne, 0, 4, "$\n4", "$4",
                 ("Arial", 12, "bold"), [8, 10], [3, 0], [1, 1])
    createButton(layerOne, 0, 5, "%\n5", "%5",
                 ("Arial", 12, "bold"), [8, 10], [3, 0], [1, 1])
    createButton(layerOne, 0, 6, "^\n6", "^6",
                 ("Arial", 12, "bold"), [8, 10], [3, 0], [1, 1])
    createButton(layerOne, 0, 7, "&\n7", "&7",
                 ("Arial", 12, "bold"), [8, 10], [3, 0], [1, 1])
    createButton(layerOne, 0, 8, "*\n8", "*8",
                 ("Arial", 12, "bold"), [8, 10], [3, 0], [1, 1])
    createButton(layerOne, 0, 9, "(\n9",
                 "(9", ("Arial", 12, "bold"), [8, 10], [3, 0], [1, 1])
    createButton(layerOne, 0, 10, ")\n0", ")0", ("Arial",
                 12, "bold"), [8, 10], [3, 0], [1, 1])
    createButton(layerOne, 0, 11, "_\n-", "`_-",
                 ("Arial", 12, "bold"), [8, 10], [3, 0], [1, 1])
    createButton(layerOne, 0, 12, "+\n=", "+=",
                 ("Arial", 12, "bold"), [8, 10], [3, 0], [1, 1])
    createButton(layerOne, 0, 13, "Backspace", "bse",
                 ("Arial", 12, "bold"), [4, 19], [12, 0], [1, 1])

    createButton(layerTwo, 0, 0, "Tab\n", "tab",
                 ("Arial", 12, "bold"), [8, 10], [7, 0], [1, 1])
    createButton(layerTwo, 0, 1, "Q", "q",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerTwo, 0, 2, "W", "w",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerTwo, 0, 3, "E", "e",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerTwo, 0, 4, "R", "r",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerTwo, 0, 5, "T", "t",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerTwo, 0, 6, "Y", "y",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerTwo, 0, 7, "U", "u",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerTwo, 0, 8, "I", "i",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerTwo, 0, 9, "O", "o",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerTwo, 0, 10, "P", "p",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerTwo, 0, 11, "{\n[", "{[", ("Arial", 12, "bold"), [
                 8, 10], [3, 0], [1, 1])
    createButton(layerTwo, 0, 12, "}\n]", "}]", ("Arial", 12, "bold"), [
                 8, 10], [3, 0], [1, 1])
    createButton(layerTwo, 0, 13, "|\n\\", "|\\",
                 ("Arial", 12, "bold"), [8, 10], [6, 0], [1, 1])

    createButton(layerThree, 0, 0, "Caps\nLock", "capslock",
                 ("Arial", 12, "bold"), [8, 9], [9, 0], [1, 1])
    createButton(layerThree, 0, 1, "A", "a",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerThree, 0, 2, "S", "s",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerThree, 0, 3, "D", "d",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerThree, 0, 4, "F", "f",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerThree, 0, 5, "G", "g",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerThree, 0, 6, "H", "h",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerThree, 0, 7, "J", "j",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerThree, 0, 8, "K", "k",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerThree, 0, 9, "L", "l",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerThree, 0, 10, ":\n;", ":;",
                 ("Arial", 12, "bold"), [8, 10], [3, 0], [1, 1])
    createButton(layerThree, 0, 11, "\"\n\'", "\"\'",
                 ("Arial", 12, "bold"), [8, 10], [3, 0], [1, 1])
    createButton(layerThree, 0, 12, "Enter \n", "en",
                 ("Arial", 12, "bold"), [8, 9], [12, 0], [1, 1])

    createButton(layerFour, 0, 0, "Shift", "shift",
                 ("Arial", 12, "bold"), [4, 19], [13, 0], [1, 1])
    createButton(layerFour, 0, 1, "Z", "z",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerFour, 0, 2, "X", "x",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerFour, 0, 3, "C", "c",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerFour, 0, 4, "V", "v",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerFour, 0, 5, "B", "b",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerFour, 0, 6, "N", "n",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerFour, 0, 7, "M", "m",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(layerFour, 0, 8, "<\n,", "<,",
                 ("Arial", 12, "bold"), [8, 10], [3, 0], [1, 1])
    createButton(layerFour, 0, 9, ">\n.", ">.",
                 ("Arial", 12, "bold"), [8, 10], [3, 0], [1, 1])
    createButton(layerFour, 0, 10, "?\n/", "?/",
                 ("Arial", 12, "bold"), [8, 10], [3, 0], [1, 1])
    createButton(layerFour, 0, 11, "Shift", "shift",
                 ("Arial", 12, "bold"), [4, 19], [18, 0], [1, 1])

    createButton(layerFive, 0, 0, "Ctrl\n", "crl",
                 ("Arial", 12, "bold"), [8, 10], [7, 0], [1, 1])
    createButton(layerFive, 0, 1, "⊞\n", "start",
                 ("Arial", 12, "bold"), [8, 10], [5, 0], [1, 1])
    createButton(layerFive, 0, 2, "Alt\n", "alt",
                 ("Arial", 12, "bold"), [8, 10], [5, 0], [1, 1])
    createButton(layerFive, 0, 3, " ", "space",
                 ("Arial", 16, "bold"), [8, 16], [32, 0], [1, 1])
    createButton(layerFive, 0, 4, "Alt\n", "alt",
                 ("Arial", 12, "bold"), [8, 10], [5, 0], [1, 1])
    createButton(layerFive, 0, 5, "⊞\n", "start",
                 ("Arial", 12, "bold"), [8, 10], [5, 0], [1, 1])
    createButton(layerFive, 0, 6, "☰\n", "opt",
                 ("Arial", 12, "bold"), [8, 10], [5, 0], [1, 1])
    createButton(layerFive, 0, 7, "Ctrl\n", "crl",
                 ("Arial", 12, "bold"), [8, 10], [5, 0], [1, 1])


def createNavigationPad(panel):

    createButton(panel, 0, 0, "Insert\n", "ist",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(panel, 0, 1, "Home\n", "home",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(panel, 0, 2, "Page\nUp", "pup",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(panel, 1, 0, "Delete\n", "del",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(panel, 1, 1, "End\n", "end",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(panel, 1, 2, "Page\nDown", "pdown",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    tkinter.Label(panel, text="", width=8, height=4, background='gray27',
                  border=0, padx=6).grid(row=2, columnspan=3, column=0)
    createButton(panel, 3, 1, "↑", "up", ("Arial", 16, "bold"),
                 [4, 16], [3, 0], [1, 1])
    createButton(panel, 4, 1, "↓", "down",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(panel, 4, 0, "←", "left",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])
    createButton(panel, 4, 2, "→", "right",
                 ("Arial", 16, "bold"), [4, 16], [3, 0], [1, 1])


def createNumericPad(panel):

    # * Create Numpad

    createButton(panel, 0, 0, "Num\nLock", "nl",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(panel, 0, 1, "/", "/", ("Arial", 16, "bold"),
                 [4, 16], [3, 0], [1, 1])
    createButton(panel, 0, 2, "*", "*", ("Arial", 16, "bold"),
                 [4, 16], [3, 0], [1, 1])
    createButton(panel, 0, 3, "-", "-", ("Arial", 16, "bold"),
                 [4, 16], [3, 0], [1, 1])
    createButton(panel, 1, 0, "7", "7", ("Arial", 16, "bold"),
                 [4, 16], [3, 0], [1, 1])
    createButton(panel, 1, 1, "8", "8", ("Arial", 16, "bold"),
                 [4, 16], [3, 0], [1, 1])
    createButton(panel, 1, 2, "9", "9", ("Arial", 16, "bold"),
                 [4, 16], [3, 0], [1, 1])
    createButton(panel, 1, 3, "+", "+", ("Arial", 12, "bold"),
                 [4, 16], [4, 5], [2, 1])
    createButton(panel, 2, 0, "4", "4", ("Arial", 16, "bold"),
                 [4, 16], [3, 0], [1, 1])
    createButton(panel, 2, 1, "5", "5", ("Arial", 16, "bold"),
                 [4, 16], [3, 0], [1, 1])
    createButton(panel, 2, 2, "6", "6", ("Arial", 16, "bold"),
                 [4, 16], [3, 0], [1, 1])
    createButton(panel, 3, 0, "1", "1", ("Arial", 16, "bold"),
                 [4, 16], [3, 0], [1, 1])
    createButton(panel, 3, 1, "2", "2", ("Arial", 16, "bold"),
                 [4, 16], [3, 0], [1, 1])
    createButton(panel, 3, 2, "3", "3", ("Arial", 16, "bold"),
                 [4, 16], [3, 0], [1, 1])
    createButton(panel, 3, 3, "Enter", "en",
                 ("Arial", 12, "bold"), [4, 16], [4, 5], [2, 1])
    createButton(panel, 4, 0, "0", "0", ("Arial", 16, "bold"),
                 [4, 16], [9, 0], [1, 2])
    createButton(panel, 4, 2, ".", ".", ("Arial", 16, "bold"),
                 [4, 16], [3, 0], [1, 1])


def createFunctionPad(panel):

    # * Create layers

    section1 = tkinter.LabelFrame(
        panel, padx=0, pady=0, background='gray27', border=0)
    section1.grid(row=0, column=0, pady=10, padx=10, sticky="W")

    section2 = tkinter.LabelFrame(
        panel, padx=0, pady=0, background='gray27', border=0)
    section2.grid(row=0, column=1, pady=10, padx=10, sticky="W")

    section3 = tkinter.LabelFrame(
        panel, padx=0, pady=0, background='gray27', border=0)
    section3.grid(row=0, column=2, pady=10, padx=10, sticky="W")

    section4 = tkinter.LabelFrame(
        panel, padx=0, pady=0, background='gray27', border=0)
    section4.grid(row=0, column=3, pady=10, padx=10, sticky="W")

    section5 = tkinter.LabelFrame(
        panel, padx=0, pady=0, background='gray27', border=0)
    section5.grid(row=0, column=4, pady=10, padx=10, sticky="W")


    createButton(section1, 0, 0, "Esc\n", "esc",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    tkinter.Label(section1, text="", width=9, height=4, background='gray27',
                  border=0, padx=6).grid(row=0, columnspan=3, column=1)
    createButton(section2, 0, 0, "F1\n", "f1",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(section2, 0, 1, "F2\n", "f2",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(section2, 0, 2, "F3\n", "f3",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(section2, 0, 3, "F4\n", "f4",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(section3, 0, 0, "F5\n", "f5",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(section3, 0, 1, "F6\n", "f6",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(section3, 0, 2, "F7\n", "f7",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(section3, 0, 3, "F8\n", "f8",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(section4, 0, 0, "F9\n", "f9",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(section4, 0, 1, "F10\n", "f10",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(section4, 0, 2, "F11\n", "f11",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(section4, 0, 3, "F12\n", "f12",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(section5, 0, 0, "Print\nScreen", "prtsrc",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(section5, 0, 1, "Scroll\nLock", "srllock",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])
    createButton(section5, 0, 2, "Pause\nBreak", "pb",
                 ("Arial", 10, "bold"), [4, 12], [5, 0], [1, 1])

    


def onScreenKeyboard(numPad, navPad, alphaPad):

    Thread(target=createNumericPad(numPad)).start()
    Thread(target=createNavigationPad(navPad)).start()
    Thread(target=createAlphanumericPad(alphaPad)).start()
    Thread(target=createFunctionPad(funcPad)).start()
    window.mainloop()


onScreenKeyboard(numPad, navPad, alphaPad)
