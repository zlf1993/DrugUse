from PredictionModel import Predictor


W2V_MODEL_PATH = 'dataset/word2vec_twitter_model.bin'
CNN_MODEL_PATH = 'saved_model-best/model.ckpt'


predictor = Predictor(W2V_MODEL_PATH, CNN_MODEL_PATH)





print(predictor.make_prediction('hello spongebob i am your neighbor squidward'))
print(predictor.make_prediction('hello spongebob i am your neighbor patric'))
print(predictor.make_prediction('hello spongebob i am your boss mr crab'))
print(predictor.make_prediction('hello spongebob i am yourself'))
print(predictor.make_prediction('hello spongebob i want to smoke weed'))
