import glob as fileSelector, nltk as naturalLanguageToolkit, string
from sklearn.feature_extraction.text import TfidfVectorizer as TIV

stemmer = naturalLanguageToolkit.stem.porter.PorterStemmer()

punctuationRemover = dict((ord(char), None) for char in string.punctuation)

def stemmeriser(tokens):
    return [stemmer.stem(item) for item in tokens]

def normaliser(text):
    return stemmeriser(naturalLanguageToolkit.word_tokenize(text.lower().translate(punctuationRemover)))


tfIdVectorized = TIV(analyzer=normaliser, min_df=0, stop_words='english', sublinear_tf=True)

directoryFiles = sorted(fileSelector.glob("*.txt"))

print("The files to be evaluated are : {}\n".format(directoryFiles))

setOfFiles = [open(file, encoding="utf8").read() for file in directoryFiles]

tfIdMatrix = tfIdVectorized.fit_transform(setOfFiles)

similarityMatrix = (tfIdMatrix*tfIdMatrix.T).A

#print(sm)

print("The files found to be plagiarized are : \n")

ind = 1

for row in range(0, len(setOfFiles)):
    for col in range(row+1, len(setOfFiles)):
        if similarityMatrix[row,col] > 0.60:
            print("\t{} -> {} and {} with similarity {}%".format(ind, directoryFiles[row], directoryFiles[col], round(similarityMatrix[row, col]*100, 2))); ind=ind+1