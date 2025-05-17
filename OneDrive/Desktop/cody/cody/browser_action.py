# browser_actions.py
import os
import webbrowser
import subprocess
import pyautogui
import pytesseract
import speech_recognition as sr
from googletrans import Translator
import pygetwindow as gw

# -------------------- Configuration --------------------
#Path to Tesseract executable if not in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

LISTEN_TIMEOUT = 5       # seconds to listen for voice input
ICON_SEARCH_CONF = 0.8   # confidence for image matching

translator = Translator()
recognizer = sr.Recognizer()

# -------------------- OCR Utilities --------------------
def ocr_near_cursor(width=300, height=100):
    x, y = pyautogui.position()
    region = (x - width//2, y - height//2, width, height)
    img = pyautogui.screenshot(region=region)
    return pytesseract.image_to_string(img).strip()

# -------------------- UI/Window Utilities --------------------
def activate_window(title_substring):
    for w in gw.getAllWindows():
        if title_substring.lower() in w.title.lower():
            w.activate()
            return True
    return False

# -------------------- Action Implementations --------------------
def search_this(_=None):
    query = ocr_near_cursor()
    if query:
        webbrowser.open(f"https://www.google.com/search?q={query}")

def open_this(_=None):
    pyautogui.click()

def scroll(direction='down', amount=500):
    pyautogui.scroll(-amount if direction=='down' else amount)

def navigate_back(_=None):
    pyautogui.hotkey('alt', 'left')

def navigate_forward(_=None):
    pyautogui.hotkey('alt', 'right')

def new_tab(_=None):
    pyautogui.hotkey('ctrl', 't')

def close_tab(_=None):
    pyautogui.hotkey('ctrl', 'w')

def translate_this(dest='en'):
    text = ocr_near_cursor()
    if text:
        webbrowser.open(
            f"https://translate.google.com/?sl=auto&tl={dest}&text={webbrowser.quote(text)}&op=translate"
        )

def open_folder(cmd):
    name = cmd.replace('open folder', '').strip()
    path = os.path.expanduser(f"~\\{name}")
    if os.path.isdir(path):
        os.startfile(path)

def open_app(cmd):
    name = cmd.replace('open app', '').strip()
    subprocess.Popen(['start', '', name], shell=True)

def click_icon(cmd):
    img = cmd.replace('click icon', '').strip() + '.png'
    loc = pyautogui.locateCenterOnScreen(img, confidence=ICON_SEARCH_CONF)
    if loc:
        pyautogui.click(loc)

# -------------------- Command Mapping --------------------
COMMAND_MAP = {
    'search this': search_this,
    'open this': open_this,
    'scroll down': lambda cmd=None: scroll('down'),
    'scroll up': lambda cmd=None: scroll('up'),
    'go back': navigate_back,
    'go forward': navigate_forward,
    'new tab': new_tab,
    'close tab': close_tab,
    'translate this': translate_this,
    'open folder': open_folder,
    'open app': open_app,
    'click icon': click_icon,
}

