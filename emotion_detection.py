import watson_nlp

def detect_emotion(text_to_analyze):
    # Load the pre-trained emotion workflow model for English
    emotion_model = watson_nlp.load('emotion_aggregated-workflow_en_stock')
    
    # Run the emotion prediction model on the input text
    emotion_result = emotion_model.run(text_to_analyze)
    
    # Extract the emotion prediction dictionary
    # The output typically includes scores for anger, disgust, fear, joy, and sadness
    formatted_output = emotion_result.to_dict()
    
    return formatted_output
