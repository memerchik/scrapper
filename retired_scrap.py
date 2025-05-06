import time
import os
import subprocess


def install(name):
    subprocess.call(['pip', 'install', name])


try:
    import termcolor
except ImportError as e:
    install("termcolor")


try:
    from selenium import webdriver
except ImportError as e:
    install("selenium")


if(os.name == 'posix'):#Clearing the terminal. Remove this if causes errors
   os.system('clear')
else:
   os.system('cls')


from termcolor import cprint
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import json
os.system('color')


APP_VERSION="1.2"
ser = Service("chromedriver")#Change the driver extension for macOS here. Go to https://chromedriver.chromium.org/downloads for the driver you need (check your Chrome browser version or update it to the latest one). The driver used here is for 106.0.5249.119 Chrome version on Windows, but it works for 107.0.5304.63 as well
options = Options()
options.add_experimental_option("detach", True)
options.add_experimental_option('excludeSwitches', ['enable-logging'])
browser = webdriver.Chrome(service=ser, options=options)


def loadNew(curs):
    errorr=False
    browser.get("https://lexica.art/api/trpc/prompts.infinitePrompts?batch=1&input=%7B%220%22%3A%7B%22json%22%3A%7B%22text%22%3A%22"+SEARCH_REQUEST+"%22%2C%22searchMode%22%3A%22images%22%2C%22source%22%3A%22search%22%2C%22cursor%22%3A"+str(curs)+"%7D%7D%7D")
    try:
        heading = browser.find_element(By.TAG_NAME, "pre").text
    except:
        errorr=True
    if(errorr==False):
        with open('temp.json', 'wb') as f:
            heading=heading.encode('utf8')
            f.write(heading)
        f = open('temp.json', encoding="utf8")
        data = json.load(f)
        if "error" in data[0]:
            cprint("Website returned an error, trying again in 100ms. You have "+str(curs)+" prompts by now", "yellow")
            time.sleep(0.1)
            loadNew(curs)
        else:
            j=0
            for i in data[0]['result']['data']['json']['prompts']:
                data=i['prompt']+"\n"
                with open(str(SEARCH_REQUEST+'.txt'), 'ab') as m:
                    data=data.encode('utf8')
                    m.write(data)
                if(curs+j>=Counterr-1):
                    print("Successfully added new prompts from "+str(curs)+" to "+str(Counterr))
                    cprint("Done! Check "+str(SEARCH_REQUEST)+".txt in current directory for the result", "green")
                    with open('temp.json', 'wb') as f:
                        clearr=" "
                        clearr=clearr.encode('utf8')
                        f.write(clearr)
                    browser.quit()
                    exit()
                j=j+1
            print("Successfully added new prompts from "+str(curs)+" to "+str(curs+50))
            time.sleep(0.1)
    else:
        cprint("Unexpected error, trying again in 100ms", "red")
        time.sleep(0.1)
        loadNew(curs)

def getMaxResults():
    numm=0
    nums=['0','1','2','3','4','5','6','7','8','9']
    print("You entered -1 as a prompt results number. Detecting the maximum result number...")
    browser.get("https://lexica.art/?q="+str(SEARCH_REQUEST))
    try:
        cellValue = browser.find_element(By.CSS_SELECTOR, "div.text-center:nth-child(7)").text
        for i in range(0,len(cellValue),1):
            if cellValue[i] in nums:
                numm=numm*10+int(cellValue[i])
        global Counterr
        Counterr = numm
        cprint("Maximum number of results detected: "+str(Counterr), "green")
    except:
        cprint("Error when looking for the maximum result number; Trying again in 250ms", "yellow")
        time.sleep(0.25)
        getMaxResults()
    
def startFunction():
    browser.get("https://lexica.art/api/trpc/prompts.infinitePrompts?batch=1&input=%7B%220%22%3A%7B%22json%22%3A%7B%22text%22%3A%22"+SEARCH_REQUEST+"%22%2C%22searchMode%22%3A%22images%22%2C%22source%22%3A%22search%22%2C%22cursor%22%3Anull%7D%2C%22meta%22%3A%7B%22values%22%3A%7B%22cursor%22%3A%5B%22undefined%22%5D%7D%7D%7D%7D")
    heading = browser.find_element(By.TAG_NAME, "pre").text
    with open('temp.json', 'wb') as f:
        heading=heading.encode('utf8')
        f.write(heading)
    f = open('temp.json', encoding="utf8")
    data = json.load(f)
    try:
        global cursor
        cursor=data[0]['result']['data']['json']['nextCursor']
        for i in data[0]['result']['data']['json']['prompts']:
            data=i['prompt']+"\n"
            with open(str(SEARCH_REQUEST+'.txt'), 'ab') as m:
                data=data.encode('utf8')
                m.write(data)
    except:
        cprint("Website returned an error, restaring the process automatically in 0.5s", "red")
        time.sleep(0.5)
        startFunction()

#MAIN
SEARCH_REQUEST=input("Enter the search request: ")
Counterr=int(input("Enter the number of prompts needed(minimum=50, put -1 if you want ALL results): "))

if(Counterr!=-1 and Counterr<50):
    Counterr=50
    print("The number you entered is less than 50, setting it to 50 automatically")

if(Counterr==-1):
    getMaxResults()
cprint("Request = '"+str(SEARCH_REQUEST)+"'; Prompts Number = "+str(Counterr), "green")
cprint("In case the process freezes for >15 seconds press the down arrow on the keyboard", "yellow")
cprint("Website API can return lots of errors, script will handle them automatically", "yellow")
cprint("Starting...", "green")
print("/////////////////////////////////////////////////////////////////////")

with open(str(SEARCH_REQUEST+'.txt'), 'w') as g:
    g.write('#####FIRST LINE#####'+"\n")

startFunction()

print("Successfully added first 50 lines of prompts")
while(cursor<Counterr):
    loadNew(cursor)
    cursor=cursor+50
with open('temp.json', 'wb') as f:
    clearr=" "
    clearr=clearr.encode('utf8')
    f.write(clearr)
browser.quit()
cprint("Done! Check "+str(SEARCH_REQUEST)+".txt in current directory for the result", "green")