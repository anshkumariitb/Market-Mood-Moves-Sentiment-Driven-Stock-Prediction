This repository contains all the resources and code files for the WiDS Market Mood and Moves project.

Market Mood and Moves: Project Setup Guide 

The guide below details how to set up an environment on Windows, Mac, or Linux using the Anaconda Prompt (or terminal).
Install the suitable Anaconda distribution available for your system at https://www.anaconda.com/download.

### **Step 1: Open Anaconda Prompt**
Mac/Linux: Open your standard terminal.

### **Step 2: Create the Virtual Environment**
We will use Python 3.10, which is currently the most stable version for the latest PyTorch and Transformers libraries (Python 3.11/3.12 can sometimes cause compatibility issues with specific CUDA builds).

Run the following command to create an environment named market_mood:


```bash
conda create -n market_mood python=3.10
```

*Type `y` and press Enter when prompted to proceed.*

Once created, activate the environment using the command:
```bash
conda activate market_mood
```

Note: All work for your project should be done in this environment only. Whenever you have to execute any code, you must activate the environment using the command above.

### **Step 3: Install PyTorch with CUDA (GPU Support)**
This is the most critical step. We will install PyTorch along with the CUDA toolkit.
Note: This assumes you have an NVIDIA GPU. If you do not, this command will still work but will default to CPU mode, or you can remove pytorch-cuda.

Run this single command to install PyTorch, TorchVision, and the CUDA 12.4 toolkit:
```bash
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
```
  * `-c pytorch -c nvidia` tells conda to pull specifically from the official PyTorch and NVIDIA channels to ensure you get the correct GPU-enabled versions.

### **Step 4: Install Project Libraries**
Now we install the libraries for NLP (Transformers, spaCy), Data Science (pandas, numpy), and our specific project needs (yfinance, newsapi).

Run this command:
```bash
conda install -c conda-forge transformers numpy pandas matplotlib seaborn scikit-learn statsmodels pyarrow sqlalchemy spacy nltk jupyter notebook
```

Next, install the specific API wrappers and utilities using pip:
```bash
pip install yfinance newsapi-python praw python-dotenv tqdm
```

### **Step 5: Download NLP Models**
We need to download the language dictionaries for NLTK and spaCy. Run these commands inside your environment:

For NLTK:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

For spaCy:
```bash
python -m spacy download en_core_web_sm
```

Above libraries are sufficient to get you through Week 1 and most of the stuff ahead. If any need arises new libraries or packages will be installed explicitly in the environment.
