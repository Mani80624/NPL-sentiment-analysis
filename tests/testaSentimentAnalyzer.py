from NLP.sentiment_analyzer import SentimentAnalyzer
# create a Object
sentiment = SentimentAnalyzer()

tokens = ['well', 'right', 'i', 'woke', 'midday', 'nap', 'sort', 'weird', 'but', 'ever', 
'since', 'i', 'moved', 'texas', 'i', 'problems', 'concentrating', 'things', 'i', 
'remember', 'starting', 'my', 'homework', 'th', 'grade', 'soon', 'clock', 'struck', 
'not', 'stopping', 'done', 'course', 'easier', 'but', 'i', 'still', 'but', 'i', 'moved', 
'homework', 'got', 'little', 'challenging', 'lot', 'busy', 'work', 'so', 'i', 'decided', 
'not', 'spend', 'hours', 'getting', 'but', 'thing', 'i', 'always', 'paid', 'attention', 
'class', 'plain', 'knew', 'stuff', 'i', 'look', 'back', 'i', 'really', 'worked', 'hard', 
'stayed', 'track', 'last', 'two', 'years', 'without', 'getting', 'lazy', 'i', 'would', 
'genius', 'but', 'hey', 'thats', 'good', 'too', 'late', 'correct', 'past', 'but', 'i', 
'dont', 'really', 'know', 'stay', 'focused', 'n', 'future', 'one', 'thing', 'i', 'know', 
'people', 'say', 'bc', 'they', 'live', 'campus', 'they', 'cant', 'concentrate', 'b', 'me', 
'would', 'easier', 'but', 'alas', 'im', 'living', 'home', 'watchful', 'eye', 'my', 'parents', 
'little', 'nagging', 'sister', 'nags', 'nags', 'nags', 'you', 'get', 'my', 'point', 'another', 
'thing', 'hassle', 'go', 'way', 'back', 'school', 'go', 'library', 'study', 'i', 'need', 'move', 
'but', 'i', 'dont', 'know', 'tell', 'them', 'dont', 'get', 'me', 'wrong', 'i', 'see', 'theyre', 
'coming', 'they', 'dont', 'want', 'me', 'move', 'but', 'i', 'need', 'get', 'away', 'my', 'theyve', 
'sheltered', 'me', 'so', 'much', 'i', 'dont', 'worry', 'world', 'thing', 'they', 'ask', 'me', 
'keep', 'my', 'room', 'clean', 'help', 'business', 'but', 'i', 'cant', 'even', 'but', 'i', 'need', 
'but', 'i', 'got', 'enough', 'money', 'ut', 'live', 'dorm', 'apartment', 'next', 'semester', 'i',
'think', 'ill', 'take', 'advantage', 'but', 'topic', 'i', 'went', 'sixth', 'street', 'last', 
'night', 'blast', 'i', 'havent', 'so', 'long', 'i', 'know', 'i', 'love', 'austin', 
'so', 'much', 'i', 'lived', 'va', 'i', 'used', 'go', 'dc', 'time', 'blast', 'but', 
'so', 'many', 'students', 'running', 'around', 'night', 'i', 'want', 'fun', 'i', 
'know', 'i', 'responsible', 'enough', 'able', 'fun', 'but', 'keep', 'my', 'priorities', 
'straight', 'living', 'home', 'i', 'cant', 'go', 'without', 'them', 'asking', 'you', 
'coming', 'back', 'questions', 'i', 'wish', 'i', 'could', 'treated', 'like', 'responsible', 
'person', 'but', 'my', 'sister', 'screwed', 'me', 'went', 'crazy', 'second', 'moved', 
'college', 'messed', 'whole', 'college', 'career', 'partying', 'too', 'much', 'thats', 
'ultimate', 'reason', 'they', 'dont', 'want', 'me', 'go', 'fun', 'but', 'im', 'not', 
'little', 'anymore', 'they', 'need', 'let', 'me', 'go', 'explore', 'world', 'but', 
'im', 'indian', 'indian', 'culture', 'indian', 'values', 'they', 'go', 'fun', 'i', 
'mean', 'sense', 'meeting', 'people', 'going', 'people', 'partying', 'plain', 'fun', 
'my', 'school', 'difficult', 'already', 'but', 'somehow', 'i', 'think', 'freedom', 'put', 
'pressure', 'me', 'better', 'school', 'bc', 'thats', 'my', 'parents', 'ultimately', 'i', 
'expect', 'myself', 'well', 'fun', 'writing', 'i', 'dont', 'know', 'you', 'go', 'anything', 
'writing', 'but', 'helped', 'me', 'get', 'my', 'thoughts', 'order', 'so', 'i', 'hope', 'you', 
'fun', 'reading', 'good', 'luck', 'tas']

# Tets map_emotion method
# clasificación en emotion_lexicon

#lexicon = sentiment.load_nrc_lexicon("C:/Users/ma-nu/Downloads/sentiment-analysis/NPL-sentiment-analysis/Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")


word = sentiment.normalize("playing")
print(word)

# Test analyze method
#dominant_sentiment, counter_sentiment = sentiment.analyze(tokens)

#print(f'Emoción dominante: {dominant_sentiment}\nConteo de sentimientos: {counter_sentiment}')