# Lexica.art prompt scrapper & AI-powered classifier of relations to the past within sets of prompts launching guide

## New Lexica.art scrapper

### MacOS/Linux:
```
python3 scrap2025.py
```
### Windows:
```
python scrap2025.py
```


### Retired scrapper (doesn't work for the moment):

### 1. Download the webdriver for the Chrome browser
Webdriver download link: https://chromedriver.chromium.org/downloads
Check your chrome driver: 
![image](https://user-images.githubusercontent.com/73663808/198590911-afefde95-9f5d-4998-8969-64358e3aedb8.png)

### 2. Place it in the same folder with the script and run the following command (for macOS users):
```
xattr -d com.apple.quarantine chromedriver
```
### 4. Start the script
```
pip install selenium
pip install termcolor
python scrap.py
```

## AI-powered past references detector launching guide

### Installing required libraries

In order to be able to launch the model, you need to first install the following libraries

```
pip install pandas datasets transformers scikit-learn torch numpy matplotlib seaborn
cd past_reference_classifier
```

### Testing if CUDA is available for faster perfomance

Run the following command to determine if your GPU is correctly detected:
```
python test_cuda.py
```
If your GPU has been displayed there then everything is alright, you are good to go.

### Preparing the data for classification

In order for the classifier to work the data must be stored either as a .csv file, or as a .txt file in the according format.

#### CSV File formatting

There are 2 main requirements for the csv files:

1. The prompts must be stores in the `text` column, the other columns don't matter
2. The separator of the csv file must be a comma `,`

#### Txt file formatting

For the txt files, there are almost no requirements except that every prompt has to have a separate line. You can easily get an example of the needed file by running `scrap2025.py`. Both 'array-formatted' and 'plain' txt files are accepted by the program.

#### File placement

In order to make the app work place your files into the same directory where `testing.py` is, or lower.

### Running the classifier

Once you have your data prepared, run the `testing.py` file and follow the instuctions provided there. You classified file will be saved in the same directory with the timestamp.