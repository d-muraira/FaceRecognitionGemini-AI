#@author David Arturo Martinez Muraira 


import os
from PIL import Image
import google.generativeai as genai




def imageStrings(path): #Reads all the files inside a folder and makes an array of the photos.
    dir_list = os.listdir(path)
    images_Path = []
    
    for file_name in dir_list:
        file_path = os.path.join(path, file_name)
        images_Path.append(file_path)
        
    return images_Path


def upload_to_gemini(path, mime_type=None): #Uploads image to gemini
 
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

# Configure the API key
genai.configure(api_key=os.environ["GEMINI"])


def faceDetection(images, photoPath):
   
    # Uploads all images in the folder
    uploaded_images = [upload_to_gemini(image, mime_type="image/jpeg") for image in images]
    
    # Uploads the single photo to analyze
    analyze_image = upload_to_gemini(photoPath, mime_type="image/jpeg")
    
    # Create the model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
    ]
    
    model = genai.GenerativeModel( #Model definition
        model_name="gemini-1.5-flash",
        safety_settings=safety_settings,
        generation_config=generation_config,
        system_instruction="You're an excellent detective that utilizes pictures from a person to verify their identity. You will upload a photo and based on the other pictures you have, you will decide if they are the same person or not. You can only say these 2 answers: 'Face Match Detected' or 'Face Match Not Detected'.",
    )

    # Create the initial chat session with the uploaded images
    chat_history = [
        {
            "role": "user",
            "parts": uploaded_images + ["You are going to receive n amount of photos that the user has. Now that you have the photos, save the photos."],
        },
        {
            "role": "model",
            "parts": ["Face Match Detected \n"],
        },
        {
            "role": "user",
            "parts": [analyze_image, "Analyze this photo and tell me if there's a face match with the n amount of photos you were given before."],
        },
        {
            "role": "model",
            "parts": ["Face Match Detected \n"],
        },
    ]

    chat_session = model.start_chat(history=chat_history) #Start chat

    # Send a message to the model
    response = chat_session.send_message("Remember that you are an excellent detective. Now tell me if there's a face match based on the instructions you were given.")
    answer=response.text
    return answer


def main(): #Main Function
    path = "C://Users//David//Desktop//FotosAI"
    images=imageStrings(path)
    photoPath="Angela.jpeg"
    print(faceDetection(images,photoPath))
    


main()
